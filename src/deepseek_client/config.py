import os
import json
import re
from itertools import cycle
from typing import List, Optional, Any, Union
import httpx
from cryptography.fernet import Fernet
from .constants import DEFAULT_HEADERS

class Config:
    def __init__(self, config_path: str = "config.json"):
        self.config_path = config_path
        self.tokens: List[str] = []
        self.keys: List[str] = []
        self.retry_times: int = 3
        self.max_pow_concurrency: int = 2
        self.port: int = 8000
        self.host: str = "0.0.0.0"
        self.rap_update_time: int = 0
        self.load()

    def load(self):
        # 1. Load local config if exists
        if os.path.exists(self.config_path):
            with open(self.config_path, "r", encoding="utf-8") as f:
                self._apply_config_data(json.load(f))
        
        # 2. Load RAP (Remote Account Pool) if configured
        self._load_rap()

        # 3. Fallback to env if tokens empty
        if not self.tokens:
            env_token = os.getenv("DEEPSEEK_AUTH_TOKEN")
            if env_token:
                self.tokens = [env_token]
        
        # 4. Override with env vars
        self.rap_update_time = int(os.getenv("RAP_UPDATE_TIME", os.getenv("RPA_UPDATE_TIME", self.rap_update_time)))

    def _apply_config_data(self, data: dict):
        data = self._replace_env_vars(data)
        self.tokens = data.get("tokens", self.tokens)
        self.keys = data.get("keys", self.keys)
        self.retry_times = int(data.get("retry_times", self.retry_times))
        self.max_pow_concurrency = int(data.get("max_pow_concurrency", self.max_pow_concurrency))
        self.port = int(data.get("port", self.port))
        self.host = data.get("host", self.host)
        self.rap_update_time = int(data.get("rap_update_time", data.get("rpa_update_time", self.rap_update_time)))

    def _replace_env_vars(self, data: Any) -> Any:
        """Recursively replace ${ENV_VAR} in config data."""
        if isinstance(data, dict):
            return {k: self._replace_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._replace_env_vars(i) for i in data]
        elif isinstance(data, str):
            # Match ${VAR_NAME}
            pattern = re.compile(r'\$\{([^}]+)\}')
            
            def replacer(match):
                env_var = match.group(1)
                return os.getenv(env_var, match.group(0)) # Fallback to original string if not found
            
            return pattern.sub(replacer, data)
        return data

    def _load_rap(self):
        rap_url = os.getenv("RAP_URL")
        rap_key = os.getenv("RAP_KEY")

        if not rap_url:
            return

        print(f"RAP activated: Fetching remote config from {rap_url}...")
        if not rap_key:
            print("Error: RAP_URL is set but RAP_KEY is missing. Skipping RAP.")
            return

        try:
            rap_headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
                "Accept": "*/*",
            }
            with httpx.Client(timeout=10.0, follow_redirects=True) as client:
                resp = client.get(rap_url, headers=rap_headers)
                resp.raise_for_status()
                encrypted_content = resp.content
            
            cipher_suite = Fernet(rap_key.encode())
            decrypted_content = cipher_suite.decrypt(encrypted_content)
            remote_data = json.loads(decrypted_content)
            
            print("RAP config successfully loaded and decrypted.")
            self._apply_config_data(remote_data)

            # Refresh token manager if it exists
            global _token_manager_instance
            if _token_manager_instance:
                _token_manager_instance.set_tokens(self.tokens)
        except Exception as e:
            print(f"Error loading RAP config: {e}")

_config_instance = None

def get_config() -> Config:
    global _config_instance
    if _config_instance is None:
        _config_instance = Config()
    return _config_instance

class TokenManager:
    def __init__(self, tokens: List[str]):
        self.set_tokens(tokens)

    def set_tokens(self, tokens: List[str]):
        self.tokens = tokens
        self._cycle = cycle(tokens) if tokens else None

    def get_next_token(self) -> Optional[str]:
        if not self._cycle:
            return None
        return next(self._cycle)

_token_manager_instance = None

def get_token_manager() -> TokenManager:
    global _token_manager_instance
    if _token_manager_instance is None:
        config = get_config()
        _token_manager_instance = TokenManager(config.tokens)
    return _token_manager_instance

def get_auth_token() -> Optional[str]:
    """Helper for legacy CLI usage."""
    return get_token_manager().get_next_token()

def build_headers(token: Optional[str] = None) -> dict:
    headers = DEFAULT_HEADERS.copy()
    if token:
        headers["authorization"] = f"Bearer {token}"
    return headers

