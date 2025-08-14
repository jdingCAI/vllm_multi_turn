#!/usr/bin/env python3
"""Quick test to verify KV cache metrics collection."""

import asyncio
import aiohttp
from prometheus_client.parser import text_string_to_metric_families

async def test_kv_metrics():
    async with aiohttp.ClientSession() as session:
        # Get metrics
        metrics_url = "http://localhost:8000/metrics"
        async with session.get(metrics_url) as response:
            if response.status != 200:
                print("Failed to get metrics")
                return
            
            metrics_text = await response.text()
            
            queries = None
            hits = None
            
            for family in text_string_to_metric_families(metrics_text):
                if family.name in ["vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries"]:
                    for sample in family.samples:
                        if sample.name.endswith("_total"):
                            queries = sample.value
                            print(f"Found queries: {queries}")
                elif family.name in ["vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits"]:
                    for sample in family.samples:
                        if sample.name.endswith("_total"):
                            hits = sample.value
                            print(f"Found hits: {hits}")
            
            if queries and hits and queries > 0:
                hit_rate = (hits / queries) * 100
                print(f"\nKV Cache Hit Rate: {hit_rate:.2f}%")
                print(f"Total Queries: {queries}")
                print(f"Total Hits: {hits}")
            else:
                print("No KV cache metrics found")

if __name__ == "__main__":
    asyncio.run(test_kv_metrics())