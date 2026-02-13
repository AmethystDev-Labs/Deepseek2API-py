from curl_cffi import requests
from .constants import BASE_URL, X_HIF_LEIM
from .exceptions import NetworkError

def create_pow_challenge(session: requests.Session) -> dict | None:
    url = f"{BASE_URL}/chat/create_pow_challenge"
    data = {"target_path": "/api/v0/chat/completion"}
    try:
        resp = session.post(url, json=data)
        resp.raise_for_status()
        return resp.json()
    except Exception as e:
        print(f"Error creating POW challenge: {e}")
        return None

def create_session(session: requests.Session) -> dict | None:
    url = f"{BASE_URL}/chat_session/create"
    try:
        resp = session.post(url, json={})
        resp.raise_for_status()
        resp_json = resp.json()
        if resp_json.get("code") != 0:
            print(f"Server returned error: {resp_json}")
            return None
        return resp_json
    except Exception as e:
        print(f"Error creating session: {e}")
        return None

def chat_completion(session: requests.Session, session_id: str, prompt: str, pow_response: str | None) -> None:
    url = f"{BASE_URL}/chat/completion"
    headers = session.headers.copy()
    if pow_response:
        headers["x-ds-pow-response"] = pow_response
    headers["x-hif-leim"] = X_HIF_LEIM
    data = {
        "chat_session_id": session_id,
        "parent_message_id": None,
        "prompt": prompt,
        "ref_file_ids": [],
        "thinking_enabled": False,
        "search_enabled": False,
        "preempt": False,
    }
    try:
        resp = session.post(url, json=data, headers=headers, stream=True)
        if resp.status_code != 200:
            print(f"Error: Received status code {resp.status_code}")
            print(resp.text)
            return
        for line in resp.iter_lines():
            if line:
                print(line.decode("utf-8"))
    except Exception as e:
        print(f"Error during chat completion: {e}")

