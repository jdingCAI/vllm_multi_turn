#!/usr/bin/env python3
"""
Example of how to measure KV cache swap timing by patching the CacheEngine.

This demonstrates how to add timing measurements to KV cache operations.
"""

import time
import torch
from typing import Dict, List
from vllm.worker.cache_engine import CacheEngine

# Store timing statistics
swap_timing_stats: Dict[str, List[float]] = {
    "swap_in_times": [],
    "swap_out_times": [],
    "copy_times": []
}

# Save the original methods
original_swap_in = CacheEngine.swap_in
original_swap_out = CacheEngine.swap_out
original_copy = CacheEngine.copy

def timed_swap_in(self, src_to_dst: torch.Tensor) -> None:
    """Wrapped swap_in method with timing."""
    start_time = time.perf_counter()
    original_swap_in(self, src_to_dst)
    torch.cuda.synchronize()  # Ensure GPU operations complete
    end_time = time.perf_counter()
    
    swap_time_ms = (end_time - start_time) * 1000
    swap_timing_stats["swap_in_times"].append(swap_time_ms)
    
    if len(src_to_dst) > 0:
        print(f"[KV Cache] Swap IN: {len(src_to_dst)} blocks in {swap_time_ms:.2f}ms")

def timed_swap_out(self, src_to_dst: torch.Tensor) -> None:
    """Wrapped swap_out method with timing."""
    start_time = time.perf_counter()
    original_swap_out(self, src_to_dst)
    torch.cuda.synchronize()  # Ensure GPU operations complete
    end_time = time.perf_counter()
    
    swap_time_ms = (end_time - start_time) * 1000
    swap_timing_stats["swap_out_times"].append(swap_time_ms)
    
    if len(src_to_dst) > 0:
        print(f"[KV Cache] Swap OUT: {len(src_to_dst)} blocks in {swap_time_ms:.2f}ms")

def timed_copy(self, src_to_dsts: torch.Tensor) -> None:
    """Wrapped copy method with timing."""
    start_time = time.perf_counter()
    original_copy(self, src_to_dsts)
    torch.cuda.synchronize()  # Ensure GPU operations complete
    end_time = time.perf_counter()
    
    copy_time_ms = (end_time - start_time) * 1000
    swap_timing_stats["copy_times"].append(copy_time_ms)
    
    if len(src_to_dsts) > 0:
        print(f"[KV Cache] Copy: {len(src_to_dsts)} blocks in {copy_time_ms:.2f}ms")

def print_timing_summary():
    """Print summary statistics of KV cache operations."""
    print("\n=== KV Cache Operation Timing Summary ===")
    
    for op_name, times in swap_timing_stats.items():
        if times:
            avg_time = sum(times) / len(times)
            max_time = max(times)
            min_time = min(times)
            print(f"\n{op_name}:")
            print(f"  Count: {len(times)}")
            print(f"  Average: {avg_time:.2f}ms")
            print(f"  Min: {min_time:.2f}ms")
            print(f"  Max: {max_time:.2f}ms")

# Monkey patch the CacheEngine methods
def enable_kv_cache_timing():
    """Enable KV cache timing by patching CacheEngine methods."""
    CacheEngine.swap_in = timed_swap_in
    CacheEngine.swap_out = timed_swap_out
    CacheEngine.copy = timed_copy
    print("KV cache timing enabled!")

def disable_kv_cache_timing():
    """Restore original CacheEngine methods."""
    CacheEngine.swap_in = original_swap_in
    CacheEngine.swap_out = original_swap_out
    CacheEngine.copy = original_copy
    print("KV cache timing disabled!")

if __name__ == "__main__":
    # Example usage
    print("This module provides functions to measure KV cache operation timing.")
    print("To use it in your vLLM application:")
    print("1. Import this module")
    print("2. Call enable_kv_cache_timing() before starting vLLM")
    print("3. Run your workload")
    print("4. Call print_timing_summary() to see results")