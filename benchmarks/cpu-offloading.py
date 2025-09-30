# SPDX-License-Identifier: Apache-2.0
"""
This file demonstrates the example usage of cpu offloading
with LMCache in vLLM v1.

Note that lmcache needs to be installed to run this example.
Learn more about LMCache in https://github.com/LMCache/LMCache.
"""
import os
import torch
import argparse
import time
import json
from lmcache.v1.cache_engine import LMCacheEngineBuilder
from lmcache.integration.vllm.utils import ENGINE_NAME
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="CPU offloading example with LMCache")
    parser.add_argument("--num-prompts", type=int, default=10,
                      help="Number of prompts to generate (default: 10)")
    parser.add_argument("--num-tokens", type=int, default=10000,
                      help="Number of tokens per prompt (default: 10000)")
    parser.add_argument("--enable-lmcache", action="store_true",
                      help="Enable LMCache for CPU offloading (default: True)")
    parser.add_argument("--enable-lmcache-ssd", action="store_true",
                      help="Enable LMCache for SSD offloading (default: False)")
    parser.add_argument("--bypass-method", type=str, default="odirect",
                      choices=["odirect", "mmap_dontneed", "posix_fadvise", "drop_caches", "none"],
                      help="Method to bypass page cache (default: odirect)")
    parser.add_argument("--drop-caches-between-runs", action="store_true",
                      help="Drop page caches between runs (requires sudo)")
    return parser.parse_args()

def setup_lmcache_environment(num_prompts, num_tokens):
    """
    Configure LMCache environment variables.
    Args:
        num_prompts: Number of prompts to process
        num_tokens: Number of tokens per prompt
    """
    cpu_size = num_prompts * num_tokens * 1.5 / 10000  # 1.5GB per 10000 tokens

    env_vars = {
        "LMCACHE_CHUNK_SIZE": "256",         # Set tokens per chunk
        "LMCACHE_LOCAL_CPU": "True",         # Enable local CPU backend
        "LMCACHE_MAX_LOCAL_CPU_SIZE": str(cpu_size)  # Dynamic CPU memory limit (GB)
    }
    for key, value in env_vars.items():
        os.environ[key] = value

def setup_lmcache_ssd_environment(num_prompts, num_tokens, bypass_method="odirect"):
    """
    Configure LMCache environment variables for SSD offloading.
    Args:
        num_prompts: Number of prompts to process
        num_tokens: Number of tokens per prompt
        bypass_method: Method to bypass page cache ("odirect", "mmap_dontneed", "posix_fadvise", "drop_caches")
    """
    # For 100k tokens: actual size is ~12.2GB, but LMCache tracking has overhead
    # Setting to 20GB to ensure no allocation failures
    disk_size = 20.0  # Fixed 20GB allocation
    
    extra_config = {}
    if bypass_method == "odirect":
        extra_config["use_odirect"] = True
    elif bypass_method == "mmap_dontneed":
        extra_config["use_mmap"] = True
        extra_config["mmap_advise"] = "MADV_DONTNEED"
    elif bypass_method == "posix_fadvise":
        extra_config["use_fadvise"] = True
        extra_config["fadvise_flag"] = "POSIX_FADV_DONTNEED"
    elif bypass_method == "drop_caches":
        # This method requires manual cache dropping between runs
        pass
    
    env_vars = {
        "LMCACHE_USE_EXPERIMENTAL": "True",      # Enable experimental features
        "LMCACHE_CHUNK_SIZE": "256",             # Set tokens per chunk
        "LMCACHE_LOCAL_DISK": "/home/jding/local_disk_storage",  # SSD storage directory
        "LMCACHE_LOCAL_CPU": "False",            # Disable CPU backend when using SSD
        "LMCACHE_MAX_LOCAL_DISK_SIZE": str(disk_size),  # Dynamic disk size based on tokens
        "LMCACHE_MAX_LOCAL_CPU_SIZE": "20.0",      # 5GB CPU limit (even though CPU is disabled)
    }
    
    if extra_config:
        env_vars["LMCACHE_EXTRA_CONFIG"] = json.dumps(extra_config)
    
    for key, value in env_vars.items():
        os.environ[key] = value

def calculate_gpu_utilization(target_memory_gb=24):
    """
    Calculate GPU memory utilization to use exactly target_memory_gb of GPU memory.
    Args:
        target_memory_gb: Target GPU memory usage in gigabytes
    Returns:
        float: GPU memory utilization ratio (0.0 to 1.0)
    Raises:
        RuntimeError: If GPU memory is less than target_memory_gb
    """
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)  # Convert to GB
    if total_memory < target_memory_gb:
        raise RuntimeError(f"GPU memory ({total_memory:.1f}GB) is less than required memory ({target_memory_gb}GB)")

    return target_memory_gb / total_memory

def create_test_prompts(num_prompts=10, num_tokens=1000):
    """
    Create test prompts with index prefix and dummy body.
    Args:
        num_prompts: Number of prompts to generate
        num_tokens: Approximate number of tokens per prompt (using 'Hi ' as token unit)
    Returns:
        list: List of prompts with format '[index] Hi Hi Hi...'
    """
    prompts = []
    dummy_text = "Hi " * num_tokens

    for i in range(num_prompts):
        prompt = f"[Prompt {i}] {dummy_text} how are you?"
        prompts.append(prompt)

    return prompts

def initialize_llm(model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_len=16384, enable_lmcache=True):
    """
    Initialize the LLM with appropriate configurations.
    Args:
        model_name: Name of the model to load
        max_len: Maximum sequence length
    Returns:
        LLM: Configured LLM instance
    """
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    ) if enable_lmcache else None

    return LLM(
        model=model_name,
        kv_transfer_config=ktc,
        max_model_len=max_len,
        enable_prefix_caching=False,
        gpu_memory_utilization=calculate_gpu_utilization()
    )

def generate_and_print_output(llm, prompts, sampling_params):
    """
    Generate text and print the results.
    Args:
        llm: LLM instance
        prompts: List of input prompts
        sampling_params: Configured sampling parameters
    Returns:
        float: Time taken for generation in seconds
    """
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()

    for output in outputs:
        generated_text = output.outputs[0].text
        print(f"Generated text: {generated_text!r}")

    return end_time - start_time

def drop_page_caches():
    """Drop Linux page caches. Requires sudo privileges."""
    import subprocess
    try:
        # Sync first to ensure all buffered data is written
        subprocess.run(["sync"], check=True)
        # Drop page caches (1 = pagecache, 2 = dentries/inodes, 3 = both)
        subprocess.run(["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"], check=True)
        print("Successfully dropped page caches")
    except subprocess.CalledProcessError as e:
        print(f"Warning: Failed to drop page caches: {e}")
        print("Consider running with sudo or using alternative bypass methods")

def main():
    """Main execution function."""
    # Parse command line arguments
    args = parse_arguments()

    # Setup environment based on cache configuration
    if args.enable_lmcache_ssd:
        setup_lmcache_ssd_environment(args.num_prompts, args.num_tokens, args.bypass_method)
        enable_cache = True
    elif args.enable_lmcache:
        setup_lmcache_environment(args.num_prompts, args.num_tokens)
        enable_cache = True
    else:
        enable_cache = False

    # Create prompts and sampling parameters
    prompts = create_test_prompts(num_prompts=args.num_prompts, num_tokens=args.num_tokens)
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    # Initialize model
    llm = initialize_llm(enable_lmcache=enable_cache)

    # First run
    print("\nFirst run:")
    first_run_time = generate_and_print_output(llm, prompts, sampling_params)
    print(f"First run time: {first_run_time:.2f} seconds")

    # Drop caches between runs if requested
    if args.drop_caches_between_runs:
        print("\nDropping page caches between runs...")
        drop_page_caches()

    # Second run
    print("\nSecond run:")
    second_run_time = generate_and_print_output(llm, prompts, sampling_params)
    print(f"Second run time: {second_run_time:.2f} seconds")

    # Print speedup
    if first_run_time > 0:
        speedup = first_run_time / second_run_time
        print(f"\nSpeedup (first run / second run): {speedup:.2f}x")

    # Cleanup if LMCache was enabled
    if enable_cache:
        LMCacheEngineBuilder.destroy(ENGINE_NAME)

if __name__ == "__main__":
    main()