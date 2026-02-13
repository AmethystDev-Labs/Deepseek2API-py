import asyncio
import httpx
import json
import time

async def test_chat_completion(stream=False):
    url = "http://localhost:8000/v1/chat/completions"
    headers = {
        "Authorization": "Bearer sk-deepseek-proxy-admin",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "user", "content": "Hello, who are you?"}
        ],
        "stream": stream
    }

    print(f"\nTesting {'Streaming' if stream else 'Non-streaming'} Chat Completion...")
    async with httpx.AsyncClient(timeout=60.0) as client:
        if not stream:
            resp = await client.post(url, json=payload, headers=headers)
            print(f"Status: {resp.status_code}")
            if resp.status_code == 200:
                data = resp.json()
                with open("tests/response_non_stream.json", "w", encoding="utf-8") as f:
                    json.dump(data, f, indent=2, ensure_ascii=False)
                print("Response saved to tests/response_non_stream.json")
            else:
                print("Error:", resp.text)
        else:
            async with client.stream("POST", url, json=payload, headers=headers) as resp:
                print(f"Status: {resp.status_code}")
                with open("tests/response_stream.txt", "w", encoding="utf-8") as f:
                    async for line in resp.aiter_lines():
                        if line.startswith("data: "):
                            data_str = line[6:]
                            if data_str == "[DONE]":
                                f.write("\n[DONE]\n")
                                break
                            try:
                                data = json.loads(data_str)
                                content = data["choices"][0]["delta"].get("content", "")
                                f.write(content)
                                f.flush()
                            except:
                                f.write(f"\nFailed to parse: {line}\n")
                print("Streaming response saved to tests/response_stream.txt")

async def main():
    # Test non-streaming
    await test_chat_completion(stream=False)
    # Test streaming
    await test_chat_completion(stream=True)

if __name__ == "__main__":
    asyncio.run(main())
