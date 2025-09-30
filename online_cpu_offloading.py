#!/usr/bin/env python3
"""
Simple online inference example with CPU offloading using LMCache and vLLM.

Prerequisites:
- pip install lmcache openai
"""

import os
import re
import time
import asyncio
import aiohttp
import subprocess
from typing import Tuple, Optional, Dict
from openai import AsyncOpenAI
from prometheus_client.parser import text_string_to_metric_families
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME


# Global variable for log file path
LOG_FILE_PATH = None

def get_lmcache_cpu_hit_rate(log_file: Optional[str] = None) -> Dict[str, float]:
    """Parse LMCache logs from server output to calculate CPU KV hit rate.
    
    Args:
        log_file: Optional path to log file. If not provided, will use /tmp/vllm_server_live.log
    
    Returns dict with 'total_tokens', 'total_hits', 'hit_rate'
    """
    # Default to the live server log file
    if not log_file:
        log_file = "/tmp/vllm_server_live.log"
    
    log_content = ""
    
    if os.path.exists(log_file):
        # Read the log file
        with open(log_file, 'r') as f:
            log_content = f.read()
    else:
        print(f"  ⚠️  Warning: Log file {log_file} not found!")
        print(f"  Make sure to start vLLM server with: 2>&1 | tee {log_file}")
        return {'total_tokens': 0, 'total_hits': 0, 'hit_rate': 0}
    
    # Parse LMCache hit information
    # Pattern: "Reqid: X, Total tokens Y, LMCache hit tokens: Z"
    pattern = re.compile(r'Total tokens\s*(\d+),\s*LMCache hit tokens:\s*(\d+)')
    
    total_tokens = 0
    total_hits = 0
    
    for match in pattern.finditer(log_content):
        tokens = int(match.group(1))
        hits = int(match.group(2))
        total_tokens += tokens
        total_hits += hits
    
    hit_rate = (total_hits / total_tokens * 100) if total_tokens > 0 else 0
    
    return {
        'total_tokens': total_tokens,
        'total_hits': total_hits,
        'hit_rate': hit_rate
    }

async def get_kv_cache_metrics(session: aiohttp.ClientSession, metrics_url: str) -> Dict:
    """Fetch KV cache and token metrics from vLLM server.
    
    Returns:
        Dict with queries, hits, hit_rate, prompt_tokens_total
    """
    try:
        async with session.get(metrics_url) as response:
            if response.status != 200:
                return {'queries': None, 'hits': None, 'hit_rate': None, 'prompt_tokens_total': None}
            
            metrics_text = await response.text()
            
            queries = None
            hits = None
            prompt_tokens = None
            
            for family in text_string_to_metric_families(metrics_text):
                # Try both new and deprecated metric names
                if family.name in ["vllm:prefix_cache_queries", "vllm:gpu_prefix_cache_queries"]:
                    for sample in family.samples:
                        queries = sample.value
                elif family.name in ["vllm:prefix_cache_hits", "vllm:gpu_prefix_cache_hits"]:
                    for sample in family.samples:
                        hits = sample.value
                elif family.name == "vllm:prompt_tokens_total":
                    for sample in family.samples:
                        # The metric has labels, we need to get the value from the first sample
                        if sample.name == "vllm:prompt_tokens_total":
                            prompt_tokens = sample.value
            
            if queries is not None and hits is not None and queries > 0:
                hit_rate = (hits / queries) * 100
            else:
                hit_rate = None
                
            return {
                'queries': queries,
                'hits': hits, 
                'hit_rate': hit_rate,
                'prompt_tokens_total': prompt_tokens
            }
    except Exception as e:
        print(f"Failed to fetch metrics: {e}")
        return {'queries': None, 'hits': None, 'hit_rate': None, 'prompt_tokens_total': None}

async def test_with_shared_prefix():
    """Test online inference with shared prefix to demonstrate cache benefits."""
    
    # Connect to existing vLLM server
    client = AsyncOpenAI(
        api_key="EMPTY",
        base_url="http://localhost:8000/v1"
    )
    
    # Create prompts with shared prefix (smaller for clearer demonstration)
    shared_prefix = "3 AI assistant. Please answer the following questions accurately. " * 10000
    prompts = [
        shared_prefix + f"\nQuestion: What is {i} + {i}?"
        for i in range(1, 4)  # Create 3 prompts
    ]
    
    print("Testing with shared prefix...")
    print("=" * 60)
    
    # First run - populate cache
    print("\nFirst run (populating cache):")
    first_times = []
    
    # Create aiohttp session for metrics
    async with aiohttp.ClientSession() as session:
        for i, prompt in enumerate(prompts, 1):
            # Get metrics before request
            metrics_before = await get_kv_cache_metrics(session, "http://localhost:8000/metrics")
            
            start = time.time()
            response = await client.chat.completions.create(
                model="meta-llama/Llama-3.1-8B-Instruct",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=10,
                temperature=0
            )
            elapsed = time.time() - start
            first_times.append(elapsed)
            
            # Get metrics after request
            metrics_after = await get_kv_cache_metrics(session, "http://localhost:8000/metrics")
            
            # Calculate deltas for this request
            if metrics_before['queries'] is not None and metrics_after['queries'] is not None:
                queries_delta = metrics_after['queries'] - metrics_before['queries']
                hits_delta = metrics_after['hits'] - metrics_before['hits'] if metrics_before['hits'] and metrics_after['hits'] else 0
                tokens_delta = metrics_after['prompt_tokens_total'] - metrics_before['prompt_tokens_total'] if metrics_before['prompt_tokens_total'] and metrics_after['prompt_tokens_total'] else 0
                request_hit_rate = (hits_delta / queries_delta * 100) if queries_delta > 0 else 0
                
                print(f"\n  📝 Request {i} completed in {elapsed:.2f}s")
                print(f"     • prefix_cache_queries delta: {queries_delta:.0f}")
                print(f"     • prefix_cache_hits delta: {hits_delta:.0f}")
                print(f"     • prompt_tokens_total delta: {tokens_delta:.0f}")
                print(f"     • Request hit rate: {request_hit_rate:.1f}%")
                print(f"     • Overall hit rate: {metrics_after['hit_rate']:.1f}%" if metrics_after['hit_rate'] else "     • Overall hit rate: N/A")
                
                # Key comparison
                print(f"\n  🔍 KEY INSIGHT:")
                print(f"     • Tokens processed: {tokens_delta:.0f}")
                print(f"     • Cache queries made: {queries_delta:.0f}")
                print(f"     • Ratio (tokens/query): {tokens_delta/queries_delta:.1f}" if queries_delta > 0 else "     • Ratio: N/A")
                
                # Show cumulative totals
                print(f"\n  📊 Cumulative totals:")
                print(f"     • Total prefix_cache_queries: {metrics_after['queries']:.0f}")
                if metrics_after['prompt_tokens_total'] is not None:
                    print(f"     • Total prompt_tokens: {metrics_after['prompt_tokens_total']:.0f}")
                    print(f"     • Ratio (tokens/queries): {metrics_after['prompt_tokens_total']/metrics_after['queries']:.1f}" if metrics_after['queries'] > 0 else "     • Ratio: N/A")
                else:
                    print(f"     • Total prompt_tokens: Not available (metric might not be exposed)")
            else:
                print(f"  Prompt {i}: {elapsed:.2f}s (metrics unavailable)")
            
            # Short sleep between prompts
            await asyncio.sleep(2)
    
    # # Second run - use cache
    # print("\nSecond run (using cache):")
    # second_times = []
    # for i, prompt in enumerate(prompts, 1):
    #     start = time.time()
    #     response = await client.chat.completions.create(
    #         model="meta-llama/Llama-3.1-8B-Instruct",
    #         messages=[{"role": "user", "content": prompt}],
    #         max_tokens=50,
    #         temperature=0
    #     )
    #     elapsed = time.time() - start
    #     second_times.append(elapsed)
    #     print(f"  Prompt {i}: {elapsed:.2f}s")
    
    # Show speedup between first and subsequent requests
    if len(first_times) >= 2:
        speedup = first_times[0] / first_times[-1]
        
        print("\n" + "=" * 60)
        print("Summary:")
        print(f"Speedup from first to last request: {speedup:.2f}x")
        print("=" * 60)

def main():
    import sys
    
    # Check for log file argument, default to /tmp/vllm_server_live.log
    log_file = sys.argv[1] if len(sys.argv) > 1 else "/tmp/vllm_server_live.log"
    
    print("""
To start vLLM server with CPU offloading and logging, run:

LMCACHE_CONFIG_FILE=/home/jding/vllm_multi_turn/lmcache_config.yaml vllm serve \\
    meta-llama/Llama-3.1-8B-Instruct \\
    --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1", "kv_role":"kv_both"}' \\
    --no-enable-prefix-caching --disable-log-requests 2>&1 | tee /tmp/vllm_server_live.log

Then run this script: python online_cpu_offloading.py
Optional: Pass a custom log file path: python online_cpu_offloading.py /path/to/your/log
    """)
    
    print(f"📁 Will read logs from: {log_file}\n")
    
    # Store log file path globally for use in the test
    global LOG_FILE_PATH
    LOG_FILE_PATH = log_file
    
    # Run the test
    asyncio.run(test_with_shared_prefix())
    # LMCacheEngineBuilder.destroy(ENGINE_NAME)  # Not needed for prefix caching test

if __name__ == "__main__":
    main()