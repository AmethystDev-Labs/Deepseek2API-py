import asyncio
import time
import argparse
import statistics
import json
import httpx
from typing import List, Dict, Any

class LoadTester:
    def __init__(self, base_url: str, api_key: str, model: str, concurrency: int, duration: int, stream: bool):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.model = model
        self.concurrency = concurrency
        self.duration = duration
        self.stream = stream
        
        self.results: List[Dict[str, Any]] = []
        self.start_time = 0
        self.total_requests = 0
        self.successful_requests = 0
        self.failed_requests = 0

    async def single_request(self, client: httpx.AsyncClient):
        payload = {
            "model": self.model,
            "messages": [{"role": "user", "content": "你好，请简单介绍一下你自己。"}],
            "stream": self.stream
        }
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        start = time.perf_counter()
        try:
            if self.stream:
                async with client.stream("POST", f"{self.base_url}/v1/chat/completions", json=payload, headers=headers, timeout=60.0) as response:
                    if response.status_code == 200:
                        async for line in response.aiter_lines():
                            pass # Just consume the stream
                        status_code = 200
                    else:
                        status_code = response.status_code
            else:
                response = await client.post(f"{self.base_url}/v1/chat/completions", json=payload, headers=headers, timeout=60.0)
                status_code = response.status_code
            
            latency = (time.perf_counter() - start) * 1000
            self.results.append({"latency": latency, "status": status_code})
            
            if status_code == 200:
                self.successful_requests += 1
            else:
                self.failed_requests += 1
                
        except Exception as e:
            latency = (time.perf_counter() - start) * 1000
            self.results.append({"latency": latency, "status": "error", "error": str(e)})
            self.failed_requests += 1
        
        self.total_requests += 1

    async def worker(self):
        async with httpx.AsyncClient(limits=httpx.Limits(max_connections=self.concurrency)) as client:
            while time.perf_counter() - self.start_time < self.duration:
                await self.single_request(client)

    async def run(self):
        print(f"🚀 开始压测...")
        print(f"目标: {self.base_url} | 模型: {self.model} | 并发: {self.concurrency} | 持续时间: {self.duration}s | 流式: {self.stream}")
        
        self.start_time = time.perf_counter()
        workers = [asyncio.create_task(self.worker()) for _ in range(self.concurrency)]
        
        # Display progress
        try:
            while any(not w.done() for w in workers):
                elapsed = time.perf_counter() - self.start_time
                if elapsed >= self.duration:
                    break
                print(f"进度: {elapsed:.1f}s / {self.duration}s | 已请求: {self.total_requests} | 成功: {self.successful_requests} | 失败: {self.failed_requests}", end='\r')
                await asyncio.sleep(1)
        finally:
            for w in workers:
                w.cancel()
        
        print("\n\n📊 压测结果统计:")
        self.print_stats()

    def print_stats(self):
        total_time = time.perf_counter() - self.start_time
        latencies = [r["latency"] for r in self.results if r["status"] == 200]
        
        if not latencies:
            print("❌ 没有成功的请求")
            return

        print(f"- 总请求数: {self.total_requests}")
        print(f"- 成功数: {self.successful_requests}")
        print(f"- 失败数: {self.failed_requests}")
        print(f"- 成功率: {(self.successful_requests/self.total_requests)*100:.2f}%")
        print(f"- 平均吞吐量 (QPS): {self.successful_requests/total_time:.2f}")
        print(f"- 延迟统计 (仅成功请求):")
        print(f"  - 平均: {statistics.mean(latencies):.2f} ms")
        print(f"  - 中位数: {statistics.median(latencies):.2f} ms")
        print(f"  - P95: {statistics.quantiles(latencies, n=20)[18]:.2f} ms")
        print(f"  - 最小: {min(latencies):.2f} ms")
        print(f"  - 最大: {max(latencies):.2f} ms")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek Proxy 压测工具")
    parser.add_argument("--url", default="http://localhost:8000", help="API Base URL")
    parser.add_argument("--key", default="sk-deepseek-proxy-admin", help="API Key")
    parser.add_argument("--model", default="deepseek-chat", help="测试模型")
    parser.add_argument("--c", type=int, default=5, help="并发请求数")
    parser.add_argument("--d", type=int, default=30, help="持续时间 (秒)")
    parser.add_argument("--stream", action="store_true", help="启用流式测试")

    args = parser.parse_args()

    asyncio.run(LoadTester(args.url, args.key, args.model, args.c, args.d, args.stream).run())
