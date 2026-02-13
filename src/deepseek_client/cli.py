import json
import time
import base64
from curl_cffi import requests
from .config import build_headers, get_auth_token
from .network import create_session, create_pow_challenge, chat_completion
from .pow import DeepSeekPoW

def main():
    token = get_auth_token()
    if not token:
        print("Warning: DEEPSEEK_AUTH_TOKEN is not set. Proceeding anyway.")
    with requests.Session(impersonate="chrome120") as session:
        session.headers.update(build_headers(token))
        session_info = create_session(session)
        if not session_info:
            print("Failed to create session.")
            return
        session_id = session_info.get("data", {}).get("biz_data", {}).get("id")
        print(f"Session ID: {session_id}")
        challenge_info = create_pow_challenge(session)
        if not challenge_info:
            print("Failed to get PoW challenge.")
            return
        biz = challenge_info.get("data", {}).get("biz_data", {}).get("challenge", {})
        algorithm = biz.get("algorithm")
        challenge = biz.get("challenge")
        salt = biz.get("salt")
        difficulty = biz.get("difficulty")
        expire_at = biz.get("expire_at")
        target_path = biz.get("target_path")
        signature = biz.get("signature")
        print(f"Solving PoW: algo={algorithm}, diff={difficulty}, expire_at={expire_at}")
        pow_resp_b64 = None
        try:
            wasm = DeepSeekPoW()
            start_time = time.time()
            ans = wasm.solve_challenge(algorithm, challenge, salt, difficulty, expire_at)
            duration = time.time() - start_time
            print(f"WASM PoW calculation took {duration:.2f}s")
            if isinstance(ans, int):
                print(f"WASM PoW Solved: Answer={ans}")
                resp_obj = {
                    "algorithm": algorithm,
                    "challenge": challenge,
                    "salt": salt,
                    "answer": ans,
                    "signature": signature,
                    "target_path": target_path,
                }
                resp_str = json.dumps(resp_obj)
                pow_resp_b64 = base64.b64encode(resp_str.encode("utf-8")).decode("utf-8")
        except Exception as e:
            print(f"WASM solver error: {e}")
        chat_completion(session, session_id, "Hello", pow_resp_b64)

