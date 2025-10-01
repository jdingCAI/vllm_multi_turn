#!/usr/bin/env python3
"""
Benchmark script that loads model fresh for each configuration.
Compares CPU/host memory loading time vs disk loading time across multiple configurations.
Each configuration gets its own model instance and appropriately sized cache.
"""

import os
import logging

# Set environment variable to allow info messages
os.environ['LMCACHE_LOG_LEVEL'] = 'ERROR'
# Allow setting max_model_len beyond model's max_position_embeddings
os.environ['VLLM_ALLOW_LONG_MAX_MODEL_LEN'] = '1'

# Configure logging BEFORE any other imports
# Custom handler that filters specific warning messages but allows info
class LMCacheFilterHandler(logging.Handler):
    def __init__(self):
        super().__init__()
        self.setLevel(logging.DEBUG)

    def emit(self, record):
        # Skip O_DIRECT alignment warnings
        msg = record.getMessage()
        if "O_DIRECT requires" in msg:
            return
        if "alignment" in msg.lower() and "warning" in msg.lower():
            return
        if "not aligned" in msg.lower():
            return

        # For info messages and non-alignment warnings, use the default console handler
        if record.levelno <= logging.INFO or "alignment" not in msg.lower():
            # Get the console handler
            console_handler = logging.StreamHandler()
            console_handler.setFormatter(logging.Formatter('%(name)s - %(levelname)s - %(message)s'))
            console_handler.handle(record)

# Set up LMCache loggers with custom handler
lmcache_handler = LMCacheFilterHandler()

# Configure main LMCache loggers
for logger_name in ["lmcache", "lmcache.v1", "lmcache.v1.storage_backend",
                    "lmcache.v1.storage_backend.local_disk_backend"]:
    logger = logging.getLogger(logger_name)
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Clear existing handlers
    logger.addHandler(lmcache_handler)
    logger.propagate = False  # Don't propagate to root logger

# Also configure any dynamically created LMCache loggers
for name in logging.root.manager.loggerDict:
    if "lmcache" in name:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        logger.handlers = []
        logger.addHandler(lmcache_handler)
        logger.propagate = False

# Configure the warning filter to ignore LMCache warnings
import warnings
warnings.filterwarnings("ignore", module="lmcache")
warnings.filterwarnings("ignore", message=".*O_DIRECT.*")
warnings.filterwarnings("ignore", message=".*alignment.*")

import torch
import argparse
import time
import json
import subprocess
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from vllm import LLM, SamplingParams
from vllm.config import KVTransferConfig

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Optimized benchmark compute vs disk loading")
    parser.add_argument("--output-dir", type=str, default="benchmark_results",
                      help="Directory to save results (default: benchmark_results)")
    parser.add_argument("--model", type=str, default="meta-llama/Meta-Llama-3.1-8B-Instruct",
                      help="Model to use for benchmarking")
    parser.add_argument("--max-model-len", type=int, default=131072,
                      help="Maximum model length")
    parser.add_argument("--runs-per-config", type=int, default=1,
                      help="Number of runs per configuration for averaging")
    parser.add_argument("--full", action="store_true",
                      help="Run full test suite with all configurations (default: quick mode)")
    parser.add_argument("--mode", type=str, choices=["cpu", "disk", "both"], default="both",
                      help="Benchmark mode: cpu (CPU/host memory only), disk (disk only), or both (default: both)")
    return parser.parse_args()

def drop_page_cache():
    """Drop all page caches."""
    try:
        # sync first
        subprocess.run(["sync"], check=True)

        # Try custom script first
        if os.path.exists("/usr/local/bin/drop-caches"):
            result = subprocess.run(["sudo", "/usr/local/bin/drop-caches"],
                                  capture_output=True, text=True)
            if result.returncode == 0:
                print("  ✓ Dropped page cache")
                return True

        # Fallback to sudo
        result = subprocess.run(
            ["sudo", "sh", "-c", "echo 3 > /proc/sys/vm/drop_caches"],
            capture_output=True, text=True)

        if result.returncode == 0:
            print("  ✓ Dropped page cache (sudo)")
            return True

        return False
    except Exception as e:
        print(f"  ⚠ Could not drop cache: {e}")
        return False

def setup_lmcache_environment(enable=True, size_gb=30, mode="disk"):
    """Configure LMCache environment variables.

    Args:
        enable: Whether to enable LMCache
        size_gb: Size in GB for cache
        mode: "disk" for SSD offloading or "cpu" for CPU/host memory offloading
    """
    if enable:
        if mode == "disk":
            # Ensure cache directory exists and is clean
            cache_dir = "/home/jding/local_disk_storage"
            os.makedirs(cache_dir, exist_ok=True)

            env_vars = {
                "LMCACHE_USE_EXPERIMENTAL": "True",
                "LMCACHE_CHUNK_SIZE": "256",
                "LMCACHE_LOCAL_DISK": cache_dir,
                "LMCACHE_LOCAL_CPU": "False",
                "LMCACHE_MAX_LOCAL_DISK_SIZE": str(size_gb),
                "LMCACHE_MAX_LOCAL_CPU_SIZE": "30.0",
                "LMCACHE_LOG_LEVEL": "INFO",  # Allow info messages
                "LMCACHE_ERROR_HANDLING": "True",  # Enable error handling for cache misses
            }
            for key, value in env_vars.items():
                os.environ[key] = value
            print(f"  ✓ LMCache enabled with {size_gb}GB disk storage and error handling")
        elif mode == "cpu":
            # Configure for CPU/host memory offloading
            env_vars = {
                "LMCACHE_USE_EXPERIMENTAL": "True",  # Enable experimental features for CPU
                "LMCACHE_CHUNK_SIZE": "256",
                "LMCACHE_LOCAL_CPU": "True",  # Enable CPU backend
                "LMCACHE_MAX_LOCAL_CPU_SIZE": str(size_gb),  # CPU memory limit (GB)
                "LMCACHE_LOG_LEVEL": "INFO",  # Suppress warnings (info handled by filter)
                "LMCACHE_ERROR_HANDLING": "True",  # Enable error handling
            }
            # Clear disk-related vars when using CPU
            os.environ.pop("LMCACHE_LOCAL_DISK", None)
            os.environ.pop("LMCACHE_MAX_LOCAL_DISK_SIZE", None)

            for key, value in env_vars.items():
                os.environ[key] = value
            print(f"  ✓ LMCache enabled with {size_gb}GB CPU/host memory offloading")
    else:
        # Clear LMCache variables
        keys_to_clear = [
            "LMCACHE_USE_EXPERIMENTAL", "LMCACHE_CHUNK_SIZE",
            "LMCACHE_LOCAL_DISK", "LMCACHE_LOCAL_CPU",
            "LMCACHE_MAX_LOCAL_DISK_SIZE", "LMCACHE_MAX_LOCAL_CPU_SIZE"
            # "LMCACHE_ERROR_HANDLING"
        ]
        for key in keys_to_clear:
            os.environ.pop(key, None)
        print("  ✓ LMCache disabled")

def calculate_gpu_utilization(target_memory_gb=75):
    """Calculate GPU memory utilization."""
    if not torch.cuda.is_available():
        raise RuntimeError("No GPU available")

    total_memory = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    return max(target_memory_gb / total_memory, 0.95)  # Cap at 95%

def create_test_prompts(num_prompts, num_tokens, config_label="default"):
    """Create test prompts with unique content for each configuration.

    Uses config-specific and prompt-specific content to ensure uniqueness
    across different configurations and avoid cache conflicts.
    """
    prompts = []
    dummy_text = "Hi " * num_tokens

    for i in range(num_prompts):
        prompt = f"[Prompt {i}] {dummy_text} how are you?"
        prompts.append(prompt)

    # prompts = []

    # for i in range(num_prompts):
    #     # Create unique base word for each prompt and config
    #     base_word = f"test_{config_label}_prompt{i}_token "
    #     word_count = len(base_word.split())
    #     repeat_count = num_tokens // word_count
    #     remainder = num_tokens % word_count

    #     # Build the full text
    #     base_text = base_word * repeat_count
    #     if remainder > 0:
    #         base_text += " ".join(base_word.split()[:remainder])

    #     prompts.append(f"{base_text} Please respond.")

    return prompts

def run_inference(llm, prompts, sampling_params):
    """Run inference and measure time."""
    #Optional: tokenize to see actual token counts
    for i, prompt in enumerate(prompts):
        tokens = llm.get_tokenizer().encode(prompt)
        print(f"Prompt {i}: {len(tokens)} tokens")
    start_time = time.time()
    outputs = llm.generate(prompts, sampling_params)
    end_time = time.time()
    return end_time - start_time, outputs

def clear_kv_cache(llm):
    """Clear KV cache from the model without reloading weights."""
    # Clear GPU cache
    torch.cuda.empty_cache()

    # Reset internal caches if possible
    # Note: vLLM doesn't expose a direct API for this, but clearing GPU cache helps
    if hasattr(llm, 'llm_engine') and hasattr(llm.llm_engine, 'model_executor'):
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()

def benchmark_all_compute_configs(llm, configs, runs=3):
    """Benchmark compute time for all configurations with a single model instance."""
    results = {}
    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    print("\n📊 Running compute benchmarks (no cache)...")

    for num_prompts, num_tokens, label in configs:
        print(f"\n  Testing {label} ({num_prompts}x{num_tokens})...")

        # Calculate required memory for this config
        kv_size_gb = calculate_message_size_gb(num_prompts, num_tokens)
        print(f"    KV cache size: {kv_size_gb:.2f}GB")

        prompts = create_test_prompts(num_prompts, num_tokens, config_label=label)
        times = []

        # Warm-up run
        print("    Warm-up...", end="", flush=True)
        clear_kv_cache(llm)
        _, _ = run_inference(llm, prompts[:1], sampling_params)
        print(" done")

        for run in range(runs):
            print(f"    Run {run+1}/{runs}...", end="", flush=True)

            # Clear caches between runs
            clear_kv_cache(llm)
            time.sleep(1)

            # Run inference
            run_time, _ = run_inference(llm, prompts, sampling_params)
            times.append(run_time)
            print(f" {run_time:.2f}s")

        avg_time = np.mean(times)
        std_time = np.std(times)
        print(f"    ➜ Compute time: {avg_time:.2f} ± {std_time:.2f}s")

        results[label] = (avg_time, std_time)

    return results

def benchmark_single_disk_config(num_prompts, num_tokens, label, runs=3, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_model_len=131072):
    """Benchmark disk loading for a single configuration with fresh model load."""

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)
    cache_dir = "/home/jding/local_disk_storage"

    print(f"\n  Testing {label} ({num_prompts}x{num_tokens})...")

    # Calculate required cache size using shared function
    kv_size_gb = calculate_message_size_gb(num_prompts, num_tokens)
    cache_size = max(30, int(kv_size_gb * 1.2))  # 20% buffer
    print(f"    KV cache size: {kv_size_gb:.2f}GB, Cache allocation: {cache_size}GB")

    # Clean cache files before this test
    if os.path.exists(cache_dir):
        print(f"    Cleaning cache directory: {cache_dir}")
        subprocess.run(f"rm -rf {cache_dir}/*.pt", shell=True, capture_output=True)
        # Ensure directory exists
        os.makedirs(cache_dir, exist_ok=True)

    # Setup environment for this specific configuration
    setup_lmcache_environment(enable=True, size_gb=cache_size, mode="disk")

    # Load model fresh for this configuration
    print(f"    Loading model for {label}...")
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm = LLM(
        model=model_name,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
        gpu_memory_utilization=calculate_gpu_utilization()
    )

    # Use same prompt creation function with config label
    prompts = create_test_prompts(num_prompts, num_tokens, config_label=label)

    # First run - populate cache (this is effectively compute time with cache write)
    print("    Populating cache (compute + write)...", end="", flush=True)
    populate_time, _ = run_inference(llm, prompts, sampling_params)
    print(f" {populate_time:.2f}s")

    # Check cache files created
    cache_files = [f for f in os.listdir(cache_dir) if f.endswith('.pt')]
    print(f"    Cache files created: {len(cache_files)}")

    # Benchmark disk loading
    times = []
    for run in range(runs):
        print(f"    Run {run+1}/{runs} (disk load)...", end="", flush=True)

        # Clear page cache
        drop_page_cache()
        time.sleep(1)

        # Clear GPU cache to force reload
        clear_kv_cache(llm)

        # Run inference (should load from disk)
        run_time, _ = run_inference(llm, prompts, sampling_params)
        times.append(run_time)
        print(f" {run_time:.2f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"    ➜ Disk load time: {avg_time:.2f} ± {std_time:.2f}s")
    print(f"    ➜ Populate time (compute+write): {populate_time:.2f}s")

    # Cleanup model
    del llm
    torch.cuda.empty_cache()

    # Thorough cache cleanup
    print("    Cleaning up cache files...")
    if os.path.exists(cache_dir):
        # Remove all .pt files
        subprocess.run(f"find {cache_dir} -name '*.pt' -type f -delete", shell=True, capture_output=True)
        # Also try to clean any temp files
        subprocess.run(f"find {cache_dir} -name '*.tmp' -type f -delete", shell=True, capture_output=True)
        time.sleep(1)  # Give filesystem time to sync

    time.sleep(2)

    # Return results for this configuration
    return (avg_time, std_time, populate_time)


def benchmark_single_cpu_config(num_prompts, num_tokens, label, runs=3, model_name="meta-llama/Meta-Llama-3.1-8B-Instruct", max_model_len=131072):
    """Benchmark CPU/host memory loading for a single configuration with fresh model load."""

    sampling_params = SamplingParams(temperature=0, top_p=0.95, max_tokens=1)

    print(f"\n  Testing {label} ({num_prompts}x{num_tokens})...")

    # Calculate required cache size using shared function
    kv_size_gb = calculate_message_size_gb(num_prompts, num_tokens)
    cache_size = max(30, int(kv_size_gb * 1.1))  # 10% buffer
    print(f"    KV cache size: {kv_size_gb:.2f}GB, Cache allocation: {cache_size}GB")

    # Setup environment for this specific configuration
    setup_lmcache_environment(enable=True, size_gb=cache_size, mode="cpu")

    # Load model fresh for this configuration
    print(f"    Loading model for {label}...")
    ktc = KVTransferConfig(
        kv_connector="LMCacheConnectorV1",
        kv_role="kv_both",
    )

    llm = LLM(
        model=model_name,
        kv_transfer_config=ktc,
        max_model_len=max_model_len,
        enable_prefix_caching=False,
        gpu_memory_utilization=calculate_gpu_utilization()
    )

    # Use same prompt creation function with config label
    prompts = create_test_prompts(num_prompts, num_tokens, config_label=label)

    # First run - populate cache (this is effectively compute time with cache write to CPU memory)
    print("    Populating CPU cache (compute + CPU write)...", end="", flush=True)
    populate_time, _ = run_inference(llm, prompts, sampling_params)
    print(f" {populate_time:.2f}s")

    # Benchmark CPU loading
    times = []
    for run in range(runs):
        print(f"    Run {run+1}/{runs} (CPU load)...", end="", flush=True)

        # Don't clear cache - let LMCache handle loading from CPU
        # The cache should be automatically managed by LMCache

        # Run inference (should load from CPU memory)
        run_time, _ = run_inference(llm, prompts, sampling_params)
        times.append(run_time)
        print(f" {run_time:.2f}s")

    avg_time = np.mean(times)
    std_time = np.std(times)
    print(f"    ➜ CPU load time: {avg_time:.2f} ± {std_time:.2f}s")
    print(f"    ➜ Populate time (compute+CPU write): {populate_time:.2f}s")

    # Cleanup model
    del llm
    torch.cuda.empty_cache()
    time.sleep(2)

    # Return results for this configuration
    return (avg_time, std_time, populate_time)

def get_test_configurations(quick=False):
    """Get test configurations.

    Returns a list of tuples: (num_prompts, num_tokens_per_prompt, label)
    Each configuration will be tested for both compute and disk loading.
    """
    if quick:
        # Quick test with fewer configs
        return [
            (1, 10000, "1x10K")
        ]
    else:
        # Full test suite with various message sizes
        return [
            (1, 1000, "1x1K"),
            (1, 5000, "1x5K"),
            (1, 10000, "1x10K"),
            (1, 20000, "1x20K"),
            (1, 50000, "1x50K"),
            (1, 100000, "1x100K"),
            (2, 1000, "2x1K"),
            (2, 5000, "2x5K"),
            (2, 10000, "2x10K"),
            (2, 20000, "2x20K"),
            (2, 50000, "2x50K"),
            (2, 100000, "2x100K"),
            (8, 1000, "8x1K"),
            (8, 5000, "8x5K"),
            (8, 10000, "8x10K"),
            (8, 20000, "8x20K"),
            (8, 50000, "8x50K"),
            (8, 100000, "8x100K"),
            (16, 1000, "16x1K"),
            (16, 5000, "16x5K"),
            (16, 10000, "16x10K"),
            (16, 20000, "16x20K"),
            (16, 50000, "16x50K"),
            (16, 100000, "16x100K"),
            (32, 1000, "32x1K"),
            (32, 5000, "32x5K"),
            (32, 10000, "32x10K"),
            (32, 20000, "32x20K"),
            (32, 50000, "32x50K"),
            (32, 100000, "32x100K")
        ]

def calculate_message_size_gb(num_prompts, num_tokens):
    """Calculate KV cache size in GB."""
    # For Llama-3.1-8B: 32 layers, 128 head_dim, 32 heads
    size_bytes = num_prompts * num_tokens * 2 * 2 * 8 * 128 * 32
    return size_bytes / (1024**3)

def plot_results(results, output_dir):
    """Create visualization of benchmark results."""
    if not results:
        print("No results to plot")
        return

    # Extract data
    configs = [r['label'] for r in results]
    message_sizes = [r['message_size_gb'] for r in results]
    populate_times = [r['populate_time'] for r in results]
    cpu_times = [r['cpu_avg'] for r in results]
    cpu_stds = [r['cpu_std'] for r in results]
    disk_times = [r['disk_avg'] for r in results]
    disk_stds = [r['disk_std'] for r in results]
    cpu_vs_disk_speedups = [r['cpu_vs_disk_speedup'] for r in results]

    # Create figure
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Plot 1: Performance comparison
    x = np.arange(len(configs))
    width = 0.25

    bars1 = ax1.bar(x - width, populate_times, width,
                   label='Populate (compute)', capsize=5, color='#2E86AB')
    bars2 = ax1.bar(x, cpu_times, width, yerr=cpu_stds,
                   label='CPU Loading', capsize=5, color='#FFA500')
    bars3 = ax1.bar(x + width, disk_times, width, yerr=disk_stds,
                   label='Disk Loading', capsize=5, color='#A23B72')

    ax1.set_xlabel('Configuration', fontsize=12)
    ax1.set_ylabel('Time (seconds)', fontsize=12)
    ax1.set_title('Populate vs CPU vs Disk Loading Performance', fontsize=14, fontweight='bold')
    ax1.set_xticks(x)
    ax1.set_xticklabels(configs, rotation=45, ha='right')
    ax1.legend()
    ax1.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar in bars1:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)
    for bar in bars3:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.1f}', ha='center', va='bottom', fontsize=8)

    # Plot 2: Speedup comparison
    colors = ['#E74C3C' if s < 1 else '#2ECC71' for s in cpu_vs_disk_speedups]
    bars = ax2.bar(x, cpu_vs_disk_speedups, color=colors, alpha=0.7)

    ax2.set_xlabel('Configuration', fontsize=12)
    ax2.set_ylabel('Speedup Factor', fontsize=12)
    ax2.set_title('CPU Loading Time / Disk Loading Time', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(configs, rotation=45, ha='right')
    ax2.axhline(y=1, color='black', linestyle='--', alpha=0.5, label='Break-even')
    ax2.grid(True, alpha=0.3, axis='y')
    ax2.legend()

    # Add speedup labels
    for bar, speedup in zip(bars, cpu_vs_disk_speedups):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{speedup:.2f}x', ha='center',
                va='bottom' if height > 0 else 'top', fontsize=9)

    # Add message sizes as subtitle
    size_text = "Message sizes (GB): " + ", ".join([f"{s:.2f}" for s in message_sizes[:4]])
    fig.text(0.5, 0.02, size_text, ha='center', fontsize=10, alpha=0.7)

    plt.tight_layout()

    # Save figure
    output_file = os.path.join(output_dir, 'compute_vs_disk_optimized.png')
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    print(f"\n📈 Plot saved to: {output_file}")

    # Show plot if in interactive environment
    try:
        plt.show()
    except:
        pass

def main():
    """Main execution function."""
    args = parse_arguments()

    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)

    print("="*60)
    print("🚀 OPTIMIZED COMPUTE vs DISK LOADING BENCHMARK")
    print("="*60)
    print(f"Model: {args.model}")
    print(f"Max length: {args.max_model_len}")
    print(f"Runs per config: {args.runs_per_config}")
    print(f"Test Mode: {'Full' if args.full else 'Quick'}")
    print(f"Benchmark Mode: {args.mode.upper()}")
    print("="*60)

    # Get configurations
    configs = get_test_configurations(quick=not args.full)

    print(f"\n🔧 Benchmarking with fresh model load for each configuration")
    print(f"Configurations to test: {len(configs)}")

    # Initialize results dictionaries
    cpu_results = {}
    disk_results = {}

    # Phase 1: Benchmark CPU/host memory loading with fresh models (if enabled)
    if args.mode in ["cpu", "both"]:
        print("\n" + "="*60)
        print("PHASE 1: CPU/HOST MEMORY LOADING BENCHMARKS")
        print("="*60)

        for num_prompts, num_tokens, label in configs:
            try:
                cpu_results[label] = benchmark_single_cpu_config(
                    num_prompts, num_tokens, label,
                    runs=args.runs_per_config,
                    model_name=args.model,
                    max_model_len=args.max_model_len
                )
                print(f"  ✓ Completed CPU benchmark for {label}")
            except Exception as e:
                print(f"  ✗ Failed CPU benchmark for {label}: {e}")
                cpu_results[label] = (0, 0, 0)  # Default values for failed runs
    else:
        print("\n⚠️ Skipping CPU benchmarks (mode: disk only)")

    # Phase 2: Benchmark disk loading with fresh models (if enabled)
    if args.mode in ["disk", "both"]:
        print("\n" + "="*60)
        print("PHASE 2: DISK LOADING BENCHMARKS")
        print("="*60)

        for num_prompts, num_tokens, label in configs:
            try:
                disk_results[label] = benchmark_single_disk_config(
                    num_prompts, num_tokens, label,
                    runs=args.runs_per_config,
                    model_name=args.model,
                    max_model_len=args.max_model_len
                )
                print(f"  ✓ Completed disk benchmark for {label}")
            except Exception as e:
                print(f"  ✗ Failed disk benchmark for {label}: {e}")
                disk_results[label] = (0, 0, 0)  # Default values for failed runs
    else:
        print("\n⚠️ Skipping disk benchmarks (mode: cpu only)")

    # Combine and analyze results
    print("\n" + "="*60)
    print("📊 FINAL RESULTS")
    print("="*60)

    results = []
    for num_prompts, num_tokens, label in configs:
        # Handle different modes
        if args.mode == "cpu" and label in cpu_results:
            # CPU only mode
            cpu_avg, cpu_std, cpu_populate_time = cpu_results[label]
            result = {
                'num_prompts': num_prompts,
                'num_tokens': num_tokens,
                'label': label,
                'message_size_gb': calculate_message_size_gb(num_prompts, num_tokens),
                'populate_time': cpu_populate_time,
                'cpu_avg': cpu_avg,
                'cpu_std': cpu_std,
                'disk_avg': 0,
                'disk_std': 0,
                'cpu_vs_disk_speedup': 0,
                'disk_vs_populate_speedup': 0,
                'cpu_vs_populate_speedup': cpu_populate_time / cpu_avg if cpu_avg > 0 else 0
            }
            results.append(result)
        elif args.mode == "disk" and label in disk_results:
            # Disk only mode
            disk_avg, disk_std, disk_populate_time = disk_results[label]
            result = {
                'num_prompts': num_prompts,
                'num_tokens': num_tokens,
                'label': label,
                'message_size_gb': calculate_message_size_gb(num_prompts, num_tokens),
                'populate_time': disk_populate_time,
                'cpu_avg': 0,
                'cpu_std': 0,
                'disk_avg': disk_avg,
                'disk_std': disk_std,
                'cpu_vs_disk_speedup': 0,
                'disk_vs_populate_speedup': disk_populate_time / disk_avg if disk_avg > 0 else 0,
                'cpu_vs_populate_speedup': 0
            }
            results.append(result)
        elif args.mode == "both" and label in cpu_results and label in disk_results:
            # Both mode
            # Get CPU results
            cpu_avg, cpu_std, cpu_populate_time = cpu_results[label]
            # Get disk results
            disk_avg, disk_std, disk_populate_time = disk_results[label]

            # Use the populate time from disk benchmark (they should be similar)
            populate_time = disk_populate_time

            # Calculate speedups
            cpu_vs_disk_speedup = cpu_avg / disk_avg if disk_avg > 0 else 0
            message_size_gb = calculate_message_size_gb(num_prompts, num_tokens)

            # Calculate all speedups
            disk_vs_populate_speedup = populate_time / disk_avg if disk_avg > 0 else 0
            cpu_vs_populate_speedup = populate_time / cpu_avg if cpu_avg > 0 else 0

            result = {
                'num_prompts': num_prompts,
                'num_tokens': num_tokens,
                'label': label,
                'message_size_gb': message_size_gb,
                'populate_time': populate_time,
                'cpu_avg': cpu_avg,
                'cpu_std': cpu_std,
                'disk_avg': disk_avg,
                'disk_std': disk_std,
                'cpu_vs_disk_speedup': cpu_vs_disk_speedup,
                'disk_vs_populate_speedup': disk_vs_populate_speedup,
                'cpu_vs_populate_speedup': cpu_vs_populate_speedup
            }
            results.append(result)

    if results:
        # Adjust header based on mode
        if args.mode == "cpu":
            print("\n{:<12} {:<10} {:<15} {:<15} {:<18}".format(
                "Config", "Size(GB)", "Populate(s)", "CPU Load(s)", "CPU Speedup"))
            print("-"*75)
        elif args.mode == "disk":
            print("\n{:<12} {:<10} {:<15} {:<15} {:<18}".format(
                "Config", "Size(GB)", "Populate(s)", "Disk Load(s)", "Disk Speedup"))
            print("-"*75)
        else:  # both
            print("\n{:<12} {:<10} {:<15} {:<15} {:<15} {:<18} {:<18}".format(
                "Config", "Size(GB)", "Populate(s)", "CPU Load(s)", "Disk Load(s)",
                "Disk Speedup", "CPU Speedup"))
            print("-"*108)

        for r in results:
            # Calculate speedups for each configuration
            disk_speedup = r['populate_time'] / r['disk_avg'] if r['disk_avg'] > 0 else 0
            cpu_speedup = r['populate_time'] / r['cpu_avg'] if r['cpu_avg'] > 0 else 0

            if args.mode == "cpu":
                print("{:<12} {:<10.3f} {:<15.2f} {:<15.2f} {:<18.2f}x".format(
                    r['label'], r['message_size_gb'],
                    r['populate_time'], r['cpu_avg'], cpu_speedup))
            elif args.mode == "disk":
                print("{:<12} {:<10.3f} {:<15.2f} {:<15.2f} {:<18.2f}x".format(
                    r['label'], r['message_size_gb'],
                    r['populate_time'], r['disk_avg'], disk_speedup))
            else:  # both
                print("{:<12} {:<10.3f} {:<15.2f} {:<15.2f} {:<15.2f} {:<18.2f}x {:<18.2f}x".format(
                    r['label'], r['message_size_gb'],
                    r['populate_time'], r['cpu_avg'], r['disk_avg'],
                    disk_speedup, cpu_speedup))

        # Calculate average metrics based on mode
        avg_populate_time = np.mean([r['populate_time'] for r in results])

        if args.mode == "cpu":
            avg_cpu_time = np.mean([r['cpu_avg'] for r in results if r['cpu_avg'] > 0])
            print(f"\n📊 Average times:")
            print(f"   Populate (compute): {avg_populate_time:.2f}s")
            print(f"   CPU load: {avg_cpu_time:.2f}s")

            if avg_cpu_time > 0:
                cpu_speedup = avg_populate_time / avg_cpu_time
                print(f"\n📈 Average speedup vs compute (populate):")
                print(f"   ⚡ CPU loading is {cpu_speedup:.2f}x faster than compute")

        elif args.mode == "disk":
            avg_disk_time = np.mean([r['disk_avg'] for r in results if r['disk_avg'] > 0])
            print(f"\n📊 Average times:")
            print(f"   Populate (compute): {avg_populate_time:.2f}s")
            print(f"   Disk load: {avg_disk_time:.2f}s")

            if avg_disk_time > 0:
                disk_speedup = avg_populate_time / avg_disk_time
                print(f"\n📈 Average speedup vs compute (populate):")
                print(f"   ⚡ Disk loading is {disk_speedup:.2f}x faster than compute")

        else:  # both
            avg_cpu_time = np.mean([r['cpu_avg'] for r in results if r['cpu_avg'] > 0])
            avg_disk_time = np.mean([r['disk_avg'] for r in results if r['disk_avg'] > 0])

            print(f"\n📊 Average times:")
            print(f"   Populate (compute): {avg_populate_time:.2f}s")
            print(f"   CPU load: {avg_cpu_time:.2f}s")
            print(f"   Disk load: {avg_disk_time:.2f}s")

            if avg_disk_time > 0 and avg_cpu_time > 0:
                disk_speedup = avg_populate_time / avg_disk_time
                cpu_speedup = avg_populate_time / avg_cpu_time
                cpu_vs_disk = avg_cpu_time / avg_disk_time

                print(f"\n📈 Average speedups vs compute (populate):")
                print(f"   ⚡ Disk loading is {disk_speedup:.2f}x faster than compute")
                print(f"   ⚡ CPU loading is {cpu_speedup:.2f}x faster than compute")

                print(f"\n📊 Relative performance:")
                if cpu_vs_disk > 1:
                    print(f"   💾 Disk is {cpu_vs_disk:.2f}x faster than CPU loading")
                else:
                    print(f"   🧠 CPU is {1/cpu_vs_disk:.2f}x faster than disk loading")

        # Save results with mode-specific filename
        filename = f'results_optimized_{args.mode}.json' if args.mode != "both" else 'results_optimized.json'
        output_file = os.path.join(args.output_dir, filename)
        with open(output_file, 'w') as f:
            json.dump(results, f, indent=2)

        # Plot results (only for 'both' mode - modify plot function if needed for single modes)
        if args.mode == "both":
            plot_results(results, args.output_dir)

        print(f"\n✅ Results saved to: {output_file}")

    print("\n🎉 Optimized benchmark complete!")

    # Update final message based on mode
    if args.mode == "both":
        print("📊 Model weights were loaded TWICE (once for CPU, once for disk benchmarks)")
    elif args.mode == "cpu":
        print("📊 Model weights were loaded ONCE for CPU benchmarking")
    elif args.mode == "disk":
        print("📊 Model weights were loaded ONCE for disk benchmarking")

if __name__ == "__main__":
    main()