import json
import time
import uuid
import asyncio
import os
import traceback
import logging
from typing import List, Optional, Union, AsyncGenerator, Dict, Any
from contextlib import asynccontextmanager
from concurrent.futures import ProcessPoolExecutor
import multiprocessing

import httpx
import uvicorn
from fastapi import FastAPI, Request, HTTPException, Depends, status
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.exceptions import RequestValidationError
from pydantic import BaseModel, Field
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import anyio

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("deepseek-proxy")

try:
    from .config import get_config, get_token_manager, build_headers
    from .pow import DeepSeekPoW
    from .constants import BASE_URL, X_HIF_LEIM
except ImportError:
    from config import get_config, get_token_manager, build_headers
    from pow import DeepSeekPoW
    from constants import BASE_URL, X_HIF_LEIM

# Global process pool for PoW calculations
_process_pool: Optional[ProcessPoolExecutor] = None

def _solve_pow_worker(algorithm: str, challenge: str, salt: str, difficulty: int, expire_at: int) -> Optional[int]:
    """Worker function that runs in a separate process."""
    try:
        # Each process creates its own WASM instance to avoid sharing state
        solver = DeepSeekPoW()
        return solver.solve_challenge(algorithm, challenge, salt, difficulty, expire_at)
    except Exception as e:
        print(f"Process worker error: {e}")
        return None

# --- Models ---

class ChatMessage(BaseModel):
    role: str
    content: str

class ChatCompletionRequest(BaseModel):
    model: str = "deepseek-chat"
    messages: List[ChatMessage]
    stream: bool = False
    temperature: Optional[float] = 0.7
    max_tokens: Optional[int] = None

class ChatCompletionResponseChoice(BaseModel):
    index: int
    message: ChatMessage
    finish_reason: Optional[str] = "stop"

class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4()}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: List[ChatCompletionResponseChoice]
    usage: Dict[str, int] = {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}

class ModelResponse(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "deepseek"

class ModelListResponse(BaseModel):
    object: str = "list"
    data: List[ModelResponse]

# --- Client ---

class AsyncDeepSeekClient:
    def __init__(self, client: httpx.AsyncClient):
        self.client = client
        self.config = get_config()
        # No need for a local solver here anymore, we'll use the process pool
        self.pow_semaphore = asyncio.Semaphore(10) # Safe to increase again with multiprocess

    async def solve_pow_async(self, challenge_data: dict) -> Optional[str]:
        biz_data = challenge_data.get("data", {}).get("biz_data", {}).get("challenge", {})
        algorithm = biz_data.get("algorithm")
        challenge = biz_data.get("challenge")
        salt = biz_data.get("salt")
        difficulty = biz_data.get("difficulty")
        expire_at = biz_data.get("expire_at")
        signature = biz_data.get("signature")
        target_path = biz_data.get("target_path")

        if algorithm != "DeepSeekHashV1":
            return None

        async with self.pow_semaphore:
            loop = asyncio.get_running_loop()
            # Run the CPU-bound task in a separate process
            answer = await loop.run_in_executor(
                _process_pool,
                _solve_pow_worker,
                algorithm, challenge, salt, difficulty, expire_at
            )

        if answer is not None:
            import base64
            resp_obj = {
                "algorithm": algorithm,
                "challenge": challenge,
                "salt": salt,
                "answer": answer,
                "signature": signature,
                "target_path": target_path
            }
            resp_str = json.dumps(resp_obj)
            return base64.b64encode(resp_str.encode('utf-8')).decode('utf-8')
        return None

    async def create_session(self, headers: dict, model: str = "deepseek-chat") -> Optional[str]:
        url = f"{BASE_URL}/chat_session/create"
        
        # Determine character_id based on model
        character_id = "v3"
        if "reasoner" in model.lower() or "r1" in model.lower():
            character_id = "r1"
            
        payload = {
            "character_id": character_id,
            "device_id": str(uuid.uuid4())
        }

        max_retries = 3
        for attempt in range(max_retries):
            try:
                resp = await self.client.post(url, json=payload, headers=headers, timeout=30.0)
                
                # If 202 Accepted, DeepSeek is rate limiting or queuing, wait and retry
                if resp.status_code == 202:
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"DeepSeek returned 202 Accepted (Queued). Body: {resp.text[:200]}. Retrying in {wait_time}s... (Attempt {attempt+1}/{max_retries})")
                    await asyncio.sleep(wait_time)
                    continue

                resp.raise_for_status()
                data = resp.json()
                if data.get("code") == 0:
                    return data.get("data", {}).get("biz_data", {}).get("id")
                
                error_msg = data.get("msg") or "Unknown error"
                logger.error(f"Session creation code error: {data}. Full body: {resp.text}")
                return f"Error: Session creation failed: {error_msg}"
            except json.JSONDecodeError as e:
                logger.error(f"Failed to decode session JSON: {str(e)}. Full body: {resp.text}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return "Error: Invalid JSON response from DeepSeek"
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 401:
                    logger.error(f"Unauthorized (401). Body: {e.response.text}")
                    return "Error: Unauthorized (Invalid Token)"
                logger.error(f"HTTP {e.response.status_code} during session creation. Body: {e.response.text}")
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return f"Error: HTTP {e.response.status_code} during session creation"
            except Exception as e:
                if attempt < max_retries - 1:
                    await asyncio.sleep(1)
                    continue
                return f"Error: {str(e)}"

        return "Error: Failed to create session after multiple retries (DeepSeek busy/202)"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        retry=retry_if_exception_type((httpx.NetworkError, httpx.TimeoutException)),
        reraise=True
    )
    async def get_pow_challenge(self, headers: dict) -> Optional[dict]:
        url = f"{BASE_URL}/chat/create_pow_challenge"
        data = {"target_path": "/api/v0/chat/completion"}
        try:
            resp = await self.client.post(url, json=data, headers=headers)
            resp.raise_for_status()
            return resp.json()
        except Exception:
            return None

    async def stream_chat(self, token: str, prompt: str, model: str = "deepseek-chat") -> AsyncGenerator[str, None]:
        headers = build_headers(token)
        logger.info(f"Starting stream_chat for model: {model}")
        
        try:
            session_result = await self.create_session(headers, model=model)
            if not session_result or session_result.startswith("Error:"):
                logger.error(f"Session creation failed: {session_result}")
                yield session_result or "Error: Failed to create session"
                return
            
            session_id = session_result
            logger.info(f"Session created: {session_id}")

            challenge = await self.get_pow_challenge(headers)
            if not challenge:
                logger.error("Failed to get PoW challenge")
                yield "Error: Failed to get PoW challenge"
                return

            logger.info("Solving PoW challenge...")
            pow_response = await self.solve_pow_async(challenge)
            if not pow_response:
                logger.warning("PoW solving failed or skipped")
            else:
                logger.info("PoW solved successfully")
            
            url = f"{BASE_URL}/chat/completion"
            req_headers = headers.copy()
            if pow_response:
                req_headers["x-ds-pow-response"] = pow_response
            req_headers["x-hif-leim"] = X_HIF_LEIM

            model_lower = model.lower()
            thinking_enabled = "reasoner" in model_lower or "r1" in model_lower
            search_enabled = "search" in model_lower

            payload = {
                "chat_session_id": session_id,
                "parent_message_id": None,
                "prompt": prompt,
                "ref_file_ids": [],
                "thinking_enabled": thinking_enabled,
                "search_enabled": search_enabled,
                "preempt": False,
            }

            async with self.client.stream("POST", url, json=payload, headers=req_headers, timeout=60.0) as resp:
                if resp.status_code != 200:
                    error_text = await resp.aread()
                    logger.error(f"DeepSeek API error {resp.status_code}: {error_text.decode()}")
                    yield f"Error: DeepSeek API returned {resp.status_code} - {error_text.decode()}"
                    return

                async for line in resp.aiter_lines():
                    if line:
                        yield line
        except Exception as e:
            logger.error(f"Unexpected error in stream_chat: {str(e)}")
            logger.error(traceback.format_exc())
            yield f"Error: {str(e)}"

# --- App & Dependencies ---

async def rap_fetch_loop():
    config = get_config()
    while True:
        try:
            await asyncio.sleep(config.rap_update_time * 60)
            print(f"Periodic RAP update triggered (every {config.rap_update_time} min)...")
            # Run sync load in thread to avoid blocking event loop
            await anyio.to_thread.run_sync(config.load)
        except asyncio.CancelledError:
            break
        except Exception as e:
            print(f"Error in RAP fetch loop: {e}")
            await asyncio.sleep(60) # Wait a bit before retrying on error

@asynccontextmanager
async def lifespan(app: FastAPI):
    global _process_pool
    # Initialize process pool for PoW
    # Use 2-4 workers depending on CPU to keep it responsive
    worker_count = min(os.cpu_count() or 1, 4)
    _process_pool = ProcessPoolExecutor(max_workers=worker_count)
    
    # Initialize global HTTP client with optimized pool settings
    limits = httpx.Limits(
        max_connections=200,          # Increase total connections
        max_keepalive_connections=50, # Increase keepalive connections
        keepalive_expiry=30.0         # 30s keepalive
    )
    # Use a faster, more robust async client
    async with httpx.AsyncClient(
        timeout=httpx.Timeout(60.0, connect=10.0), 
        limits=limits,
        follow_redirects=True
    ) as client:
        app.state.ds_client = AsyncDeepSeekClient(client)
        
        config = get_config()
        if config.rap_update_time > 0:
            print(f"Starting RAP fetch loop (interval: {config.rap_update_time} min)")
            update_task = asyncio.create_task(rap_fetch_loop())
            try:
                yield
            finally:
                update_task.cancel()
                try:
                    await update_task
                except asyncio.CancelledError:
                    pass
        else:
            yield
    
    # Shutdown process pool
    if _process_pool:
        _process_pool.shutdown(wait=True)

app = FastAPI(title="DeepSeek to OpenAI Proxy", lifespan=lifespan)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return JSONResponse(
        status_code=exc.status_code,
        content=openai_error(str(exc.detail), code=str(exc.status_code))
    )

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    return JSONResponse(
        status_code=422,
        content=openai_error(str(exc), error_type="validation_error")
    )

security = HTTPBearer()

def get_proxy_key(auth: HTTPAuthorizationCredentials = Depends(security)):
    config = get_config()
    if not config.keys: # No keys defined, allow all
        return auth.credentials
    if auth.credentials in config.keys:
        return auth.credentials
    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Invalid proxy key",
        headers={"WWW-Authenticate": "Bearer"},
    )

# --- Endpoints ---

def openai_error(message: str, error_type: str = "invalid_request_error", code: Optional[str] = None):
    return {
        "error": {
            "message": message,
            "type": error_type,
            "param": None,
            "code": code
        }
    }

@app.get("/v1/models")
async def list_models(_key: str = Depends(get_proxy_key)):
    models = [
        "deepseek-chat",
        "deepseek-chat-search",
        "deepseek-reasoner",
        "deepseek-reasoner-search",
        "deepseek-r1",
        "deepseek-r1-search",
        "deepseek-search"
    ]
    return ModelListResponse(
        data=[ModelResponse(id=m) for m in models]
    )

@app.post("/v1/chat/completions")
async def chat_completions(
    request: ChatCompletionRequest,
    _key: str = Depends(get_proxy_key)
):
    token_manager = get_token_manager()
    ds_token = token_manager.get_next_token()
    if not ds_token:
        raise HTTPException(status_code=500, detail="No DeepSeek tokens available")

    # Convert messages to formatted prompt with <ROLE> tags
    formatted_prompt = ""
    for msg in request.messages:
        role_tag = msg.role.capitalize()
        formatted_prompt += f"<{role_tag}>\n{msg.content}\n</{role_tag}>\n"
    
    # If last message is user, we want assistant to respond
    if request.messages and request.messages[-1].role == "user":
        formatted_prompt += "<Assistant>\n"
    
    prompt = formatted_prompt.strip()
    ds_client: AsyncDeepSeekClient = app.state.ds_client

    if not request.stream:
        # Non-streaming implementation
        full_text = ""
        try:
            async for line in ds_client.stream_chat(ds_token, prompt, model=request.model):
                if line.startswith("Error:"):
                    raise HTTPException(status_code=500, detail=line)
                
                # DeepSeek web API returns specific JSON structure
                content = ""
                try:
                    clean_line = line.strip()
                    if clean_line.startswith("data: "):
                        line_data = clean_line[6:]
                    else:
                        line_data = clean_line
                    
                    if not line_data or line_data == "[DONE]":
                        continue

                    data = json.loads(line_data)
                    # Case 1: Initial fragment block
                    if isinstance(data.get("v"), dict):
                        fragments = data["v"].get("response", {}).get("fragments", [])
                        if fragments:
                            content = fragments[0].get("content", "")
                    # Case 2: Append operation
                    elif data.get("p") and data.get("o") == "APPEND" and "v" in data:
                        content = data["v"]
                    # Case 3: Direct value string (only if not a control message with 'p')
                    elif "v" in data and isinstance(data["v"], str) and "p" not in data:
                        content = data["v"]
                    
                    full_text += content
                except json.JSONDecodeError as e:
                    logger.warning(f"Non-critical JSON parse error: {str(e)} for line: {line[:50]}")
                    continue
            
            return ChatCompletionResponse(
                model=request.model,
                choices=[
                    ChatCompletionResponseChoice(
                        index=0,
                        message=ChatMessage(role="assistant", content=full_text)
                    )
                ]
            )
        except Exception as e:
            logger.error(f"Non-stream chat execution error: {str(e)}")
            logger.error(traceback.format_exc())
            raise HTTPException(status_code=500, detail=str(e))
    else:
        # Streaming implementation
        async def response_generator():
            request_id = f"chatcmpl-{uuid.uuid4()}"
            created_time = int(time.time())
            
            try:
                async for line in ds_client.stream_chat(ds_token, prompt, model=request.model):
                    if line.startswith("Error:"):
                        yield f"data: {json.dumps(openai_error(line))}\n\n"
                        break
                    
                    try:
                        clean_line = line.strip()
                        if clean_line.startswith("data: "):
                            line_data = clean_line[6:]
                        else:
                            line_data = clean_line
                        
                        if not line_data or line_data == "[DONE]":
                            continue

                        # Robust JSON parsing
                        try:
                            ds_data = json.loads(line_data)
                        except json.JSONDecodeError:
                            # If it's not JSON, it might be a raw message or a partial chunk
                            # Let's log it for debugging and skip to avoid crashing
                            print(f"DEBUG: Skipping invalid JSON line: {line_data[:100]}...")
                            continue
                        
                        content = ""
                        finish_reason = None

                        # Case 1: Initial fragment block
                        if isinstance(ds_data.get("v"), dict):
                            fragments = ds_data["v"].get("response", {}).get("fragments", [])
                            if fragments:
                                content = fragments[0].get("content", "")
                        # Case 2: Append operation
                        elif ds_data.get("p") and ds_data.get("o") == "APPEND" and "v" in ds_data:
                            content = ds_data["v"]
                        # Case 3: Direct value string (only if not a control message with 'p')
                        elif "v" in ds_data and isinstance(ds_data["v"], str) and "p" not in ds_data:
                            content = ds_data["v"]
                        
                        # Check for finish reason
                        if ds_data.get("p") == "response/status" and ds_data.get("v") == "FINISHED":
                            finish_reason = "stop"

                        if content or finish_reason:
                            chunk = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_time,
                                "model": request.model,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": content} if content else {},
                                        "finish_reason": finish_reason
                                    }
                                ]
                            }
                            yield f"data: {json.dumps(chunk)}\n\n"
                    except json.JSONDecodeError:
                        continue
                yield "data: [DONE]\n\n"
            except Exception as e:
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(response_generator(), media_type="text/event-stream")

@app.get("/health")
async def health():
    return {"status": "healthy"}

if __name__ == "__main__":
    import uvicorn
    config = get_config()
    # High-performance settings but avoiding some experimental flags that might cause instability
    uvicorn.run(
        app, 
        host=config.host, 
        port=config.port,
        workers=1           # Must be 1 to maintain Wasmtime safety and shared state
    )
