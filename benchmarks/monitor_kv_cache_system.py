#!/usr/bin/env python3
"""
Monitor KV cache operations through system metrics and vLLM's API.
"""

import asyncio
import aiohttp
import time
from typing import Dict, Any

async def get_vllm_metrics(base_url: str = "http://localhost:8000") -> Dict[str, Any]:
    """Fetch metrics from vLLM's /metrics endpoint."""
    async with aiohttp.ClientSession() as session:
        async with session.get(f"{base_url}/metrics") as response:
            if response.status == 200:
                text = await response.text()
                return parse_prometheus_metrics(text)
    return {}

def parse_prometheus_metrics(metrics_text: str) -> Dict[str, float]:
    """Parse Prometheus format metrics into a dictionary."""
    metrics = {}
    for line in metrics_text.split('\n'):
        if line and not line.startswith('#'):
            # Parse lines like: vllm:gpu_cache_usage_perc{model_name="..."} 0.15
            if ' ' in line:
                key_part, value = line.rsplit(' ', 1)
                # Extract metric name
                if '{' in key_part:
                    metric_name = key_part.split('{')[0]
                else:
                    metric_name = key_part
                
                try:
                    metrics[metric_name] = float(value)
                except ValueError:
                    pass
    return metrics

async def monitor_kv_cache_metrics(
    base_url: str = "http://localhost:8000",
    interval: float = 1.0,
    duration: float = 60.0
):
    """Monitor KV cache metrics over time."""
    print("Starting KV cache monitoring...")
    print("Metrics will be collected every", interval, "seconds")
    print("-" * 80)
    
    start_time = time.time()
    
    # Track changes in key metrics
    prev_metrics = {}
    
    while time.time() - start_time < duration:
        try:
            metrics = await get_vllm_metrics(base_url)
            
            # Extract relevant cache metrics
            cache_metrics = {
                "gpu_cache_usage": metrics.get("vllm:gpu_cache_usage_perc", 0) * 100,
                "cpu_cache_usage": metrics.get("vllm:cpu_cache_usage_perc", 0) * 100,
                "num_swapped": metrics.get("vllm:num_requests_swapped", 0),
                "gpu_prefix_cache_hit_rate": metrics.get("vllm:gpu_prefix_cache_hit_rate", 0) * 100,
            }
            
            # Calculate swap rate (swaps per second)
            if prev_metrics and "num_swapped" in prev_metrics:
                swap_rate = (cache_metrics["num_swapped"] - prev_metrics["num_swapped"]) / interval
            else:
                swap_rate = 0
            
            # Print current status
            print(f"\rGPU Cache: {cache_metrics['gpu_cache_usage']:.1f}% | "
                  f"CPU Cache: {cache_metrics['cpu_cache_usage']:.1f}% | "
                  f"Swapped: {int(cache_metrics['num_swapped'])} | "
                  f"Swap Rate: {swap_rate:.1f}/s | "
                  f"Cache Hit: {cache_metrics['gpu_prefix_cache_hit_rate']:.1f}%", 
                  end='', flush=True)
            
            prev_metrics = cache_metrics
            
        except Exception as e:
            print(f"\nError fetching metrics: {e}")
        
        await asyncio.sleep(interval)
    
    print("\n\nMonitoring complete!")

def estimate_swap_time(num_blocks: int, block_size_mb: float = 1.0) -> Dict[str, float]:
    """
    Estimate KV cache swap time based on PCIe bandwidth.
    
    Args:
        num_blocks: Number of blocks to swap
        block_size_mb: Size of each block in MB (depends on model)
    
    Returns:
        Dictionary with time estimates
    """
    # Typical PCIe 3.0 x16 bandwidth: ~15 GB/s
    # Typical PCIe 4.0 x16 bandwidth: ~30 GB/s
    pcie_3_bandwidth_gb_s = 15
    pcie_4_bandwidth_gb_s = 30
    
    data_size_gb = (num_blocks * block_size_mb) / 1024
    
    estimates = {
        "data_size_mb": num_blocks * block_size_mb,
        "pcie_3_time_ms": (data_size_gb / pcie_3_bandwidth_gb_s) * 1000,
        "pcie_4_time_ms": (data_size_gb / pcie_4_bandwidth_gb_s) * 1000,
    }
    
    return estimates

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "estimate":
        # Example: Estimate swap time for 100 blocks of 1MB each
        num_blocks = int(sys.argv[2]) if len(sys.argv) > 2 else 100
        block_size = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0
        
        estimates = estimate_swap_time(num_blocks, block_size)
        print(f"Swap time estimates for {num_blocks} blocks ({block_size}MB each):")
        print(f"  Total data: {estimates['data_size_mb']:.1f} MB")
        print(f"  PCIe 3.0: ~{estimates['pcie_3_time_ms']:.1f} ms")
        print(f"  PCIe 4.0: ~{estimates['pcie_4_time_ms']:.1f} ms")
    else:
        # Run the monitoring
        asyncio.run(monitor_kv_cache_metrics())