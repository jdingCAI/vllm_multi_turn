# Multi-Turn KV Cache Offloading Guide

This comprehensive guide covers KV cache offloading strategies for vLLM multi-turn conversations, including CPU and SSD offloading with LMCache integration.

## Table of Contents

1. [Quick Start](#quick-start)
2. [CPU Offloading](#cpu-offloading)
3. [SSD Offloading & Page Cache](#ssd-offloading--page-cache)
4. [Benchmark Tools](#benchmark-tools)
5. [Performance Analysis](#performance-analysis)
6. [Troubleshooting](#troubleshooting)

---

## Quick Start

### Install Dependencies
```bash
pip install lmcache psutil
```

### Run CPU Offloading Benchmark
```bash
# Compare CPU offloading vs baseline
python benchmarks/run_benchmark.py --config benchmarks/benchmark_config.yaml
```

### Run Individual Tools
```bash
# CPU offloading demo
python benchmarks/cpu-offloading.py --num-prompts 10 --num-tokens 10000 --enable-lmcache

# Online inference with offloading
python online_cpu_offloading.py

# Compare compute vs disk loading
python benchmark_compute_vs_disk_optimized.py
```

---

## CPU Offloading

### Overview
CPU offloading moves KV cache from GPU memory to CPU memory while keeping model weights on GPU. This enables:
- Longer context windows on limited GPU memory
- More concurrent requests
- Multi-turn conversations with extensive history

### Implementation Methods

#### Method 1: Native vLLM (Recommended)
```bash
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --cpu-offload-gb 20 \
    --gpu-memory-utilization 0.3 \
    --enable-prefix-caching
```

#### Method 2: LMCache Integration
```bash
# Set environment variables
export LMCACHE_LOCAL_CPU="True"
export LMCACHE_MAX_LOCAL_CPU_SIZE="10.0"  # GB
export LMCACHE_CHUNK_SIZE="256"
export LMCACHE_USE_EXPERIMENTAL="True"

# Start server
python -m vllm.entrypoints.openai.api_server \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --kv-connector LMCacheConnectorV1 \
    --kv-role kv_both \
    --gpu-memory-utilization 0.25 \
    --enable-prefix-caching
```

### Configuration Guidelines

**For production (latency-sensitive):**
- GPU-only with `--gpu-memory-utilization 0.9`
- No CPU offloading

**For development/testing:**
- `--cpu-offload-gb 20` with `--gpu-memory-utilization 0.3`
- Balanced resource usage

**For batch processing:**
- `--cpu-offload-gb 40` with `--gpu-memory-utilization 0.25`
- Maximize throughput over latency

**For mixed workloads:**
- Start with `--cpu-offload-gb 10` and adjust based on monitoring

### Monitoring Metrics

```bash
# Check metrics endpoint
curl http://localhost:8000/metrics | grep -E "cpu_cache|gpu_cache|swapped"
```

Key metrics:
- `vllm:cpu_cache_usage_perc` - CPU cache utilization
- `vllm:gpu_cache_usage_perc` - GPU cache utilization
- `vllm:num_requests_swapped` - Requests swapped to CPU
- `vllm:gpu_prefix_cache_hit_rate` - Cache hit rate

### Expected Behavior

**With CPU Offloading:**
- ✅ CPU memory usage increases during conversations
- ✅ GPU memory stays within configured limit
- ✅ ~12-50% latency increase
- ✅ 4-5x more KV cache capacity
- ✅ `cpu_cache_usage_perc` > 0

**Without CPU Offloading (Baseline):**
- ✅ Minimal CPU memory usage
- ✅ GPU memory uses up to 90%
- ✅ Lower latency, better throughput
- ✅ Limited to GPU memory capacity

### Performance Characteristics

Based on testing with Llama-3.1-8B on A100 80GB:

| Metric | LMCache (KV on CPU) | Baseline (KV on GPU) | Difference |
|--------|---------------------|----------------------|------------|
| **Average Latency** | 1.53s | 1.36s | +12.1% |
| **First Turn Latency** | 1.91s | 1.42s | +34.5% |
| **Later Turns Latency** | 1.33s | 1.33s | ~0% |
| **Input Throughput** | 141.6 tok/s | 144.5 tok/s | -2% |
| **Output Throughput** | 67.9 tok/s | 73.9 tok/s | -8% |
| **KV Capacity** | 125,696 tokens | 28,640 tokens | **4.4x** |

---

## SSD Offloading & Page Cache

### Problem: Page Cache Interference

When measuring actual SSD I/O performance, the Linux page cache can interfere with benchmarks by serving cached data from RAM instead of disk.

### Why O_DIRECT Fails

O_DIRECT (bypasses page cache) requires:
1. **Buffer memory alignment**: 4096 bytes (page boundary)
2. **File offset alignment**: Block size aligned
3. **Transfer size alignment**: Multiple of block size

PyTorch tensors are only 64-byte aligned, causing `EINVAL (22)` errors.

### Solutions for Bypassing Page Cache

#### Method 1: Drop Page Cache (Requires sudo)

```bash
# Clear page cache between runs
sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'

# Use with setup script
sudo ./setup_drop_caches.sh
```

This drops:
- `1` = page cache
- `2` = dentries and inodes
- `3` = both

#### Method 2: posix_fadvise

Tell kernel to not cache specific files:

```python
import ctypes
import ctypes.util
import os

libc = ctypes.CDLL(ctypes.util.find_library('c'))
POSIX_FADV_DONTNEED = 4

def drop_file_cache(filepath):
    fd = os.open(filepath, os.O_RDONLY)
    size = os.path.getsize(filepath)
    libc.posix_fadvise(fd, 0, size, POSIX_FADV_DONTNEED)
    os.close(fd)
```

#### Method 3: Monitor Actual Disk I/O

```bash
# Terminal 1: Monitor I/O
iostat -x 1

# Terminal 2: Run benchmark
python benchmarks/cpu-offloading.py --enable-lmcache-ssd \
    --num-prompts 10 --num-tokens 10000 --bypass-method drop_caches
```

### Expected Performance

**With page cache bypass:**
- First run: Writes to disk (slow)
- Second run: Reads from disk (~500-2000 MB/s for NVMe)

**Without page cache bypass:**
- First run: Writes to disk and page cache
- Second run: Reads from RAM (~10-50 GB/s)

### Verification

1. **Check disk I/O stats**: Second run should show disk reads
2. **Monitor with iostat**: `iostat -x 1` shows disk activity
3. **Use iotop**: `sudo iotop -o` shows processes doing I/O
4. **Check speedup**: Should be lower when bypassing cache

---

## Benchmark Tools

### 1. benchmark_serving_multi_turn.py

Main benchmark script for multi-turn conversations.

```bash
# Via run_benchmark.py orchestrator
python benchmarks/run_benchmark.py \
    --config benchmarks/benchmark_config.yaml \
    --output-dir results/

# Direct usage
python benchmarks/benchmark_serving_multi_turn.py \
    --model meta-llama/Meta-Llama-3.1-8B-Instruct \
    --dataset benchmarks/conversations.json \
    --request-rate 1.0
```

**Features:**
- CPU/GPU memory monitoring
- LMCache metrics collection
- Multi-turn conversation support
- Configurable via YAML
- Structured results output

### 2. cpu-offloading.py

Demonstrates CPU offloading with LMCache.

```bash
python benchmarks/cpu-offloading.py \
    --num-prompts 10 \
    --num-tokens 10000 \
    --enable-lmcache
```

**Options:**
- `--enable-lmcache`: Enable CPU offloading
- `--enable-lmcache-ssd`: Enable SSD offloading
- `--bypass-method`: Page cache bypass method
- `--drop-caches-between-runs`: Clear cache between runs

### 3. online_cpu_offloading.py

Online inference with real-time metrics.

```bash
python online_cpu_offloading.py
```

**Features:**
- Automatic server lifecycle management
- Real-time LMCache hit rate calculation
- Prometheus metrics monitoring
- Progress tracking with rich formatting

### 4. benchmark_compute_vs_disk_optimized.py

Compares CPU/host memory loading vs disk loading.

```bash
python benchmark_compute_vs_disk_optimized.py \
    --num-prompts 10 \
    --max-active-conversations 5
```

**Features:**
- Fresh model load for each config
- Multiple GPU memory configurations
- Visualization of results
- Detailed timing breakdown

### 5. run_benchmark.py

Orchestrator for complex benchmark workflows.

```bash
python benchmarks/run_benchmark.py \
    --config benchmarks/benchmark_config.yaml \
    --output-dir results/run_$(date +%Y%m%d_%H%M%S)
```

**Features:**
- YAML-based configuration
- Server lifecycle management
- Structured output (JSON + logs)
- Dataset generation from text files
- Multiple offload modes (cpu/ssd/none)

---

## Performance Analysis

### Key Metrics to Monitor

1. **Memory Usage Pattern**
   - CPU memory growth over time
   - GPU memory utilization
   - Peak vs average consumption

2. **Performance Trade-offs**
   - Latency increase percentage
   - Throughput reduction
   - TTFT and TPOT differences

3. **Offloading Effectiveness**
   - KV cache distribution (CPU vs GPU)
   - Swap frequency and volume
   - Cache hit rates

### Analysis Tools

```bash
# Parse LMCache hit rates
python parse_lmcache_cpu_hits.py /tmp/vllm_server.log

# Check token statistics
python benchmarks/analyze_conv_tokens.py conversations.json
python benchmarks/check_chat_tokens.py
python benchmarks/check_prompt_tokens.py
```

### Recommended Visualizations

1. Time-series plot of CPU/GPU memory usage
2. Box plot of latency distributions
3. Bar chart comparing average metrics
4. Scatter plot of request size vs offloading amount

---

## Troubleshooting

### LMCache Not Found
```bash
pip install lmcache
```

### Server Won't Start
- Check GPU availability: `nvidia-smi`
- Verify model access: May need HuggingFace token
- Check port 8000 is free: `lsof -i :8000`

### No CPU Offloading Observed
- Verify environment variables are set
- Check server logs for "LMCache" initialization
- Ensure GPU memory limit forces offloading
- Monitor `/metrics` endpoint for `cpu_cache_usage_perc`

### Page Cache Not Bypassing
- Use `iostat -x 1` to verify disk I/O
- Check `sudo` permissions for drop_caches
- Verify file paths in bypass method
- Try multiple bypass methods simultaneously

### Poor Performance
- First-time compilation takes 60+ seconds (normal)
- Consider `--enforce-eager` to skip compilation
- Check PCIe bandwidth: typically 15-30 GB/s
- Adjust chunk size and memory limits
- Monitor for frequent swapping

### Metrics Not Available
- Ensure `--metrics-url` provided to benchmark
- Check metrics endpoint enabled on server
- Verify Prometheus format with `curl http://localhost:8000/metrics`

---

## Configuration Files

### lmcache_config.yaml
CPU offloading configuration:
```yaml
local_cpu:
  enabled: true
  max_size_gb: 10.0
chunk_size: 256
experimental: true
```

### lmcacheSSD_config.yaml
SSD offloading configuration:
```yaml
local_disk:
  enabled: true
  path: /path/to/ssd/cache
  max_size_gb: 100.0
chunk_size: 256
bypass_method: drop_caches
```

### benchmark_config.yaml
Benchmark orchestration:
```yaml
model: "meta-llama/Llama-3.1-8B-Instruct"
workload:
  num_conversations: 24
  num_turns: 10
  common_prefix_num_tokens: 512
server:
  offload_mode: cpu  # cpu, ssd, none
  cpu_memory_size: 10.0
  gpu_memory_utilization: 0.25
```

---

## References

- [vLLM GitHub Issue #23087](https://github.com/vllm-project/vllm/issues/23087) - CPU offloading discussion
- [LMCache Documentation](https://docs.lmcache.ai/) - Official LMCache docs
- [vLLM KV Cache Management](https://docs.vllm.ai/en/latest/)
- [CPU Offloading Example](examples/others/lmcache/cpu_offload_lmcache.py) - vLLM example

---

## When to Use CPU/SSD Offloading

**Use CPU Offloading When:**
- Long multi-turn conversations (>10 turns)
- Memory-constrained GPU (<40GB)
- Multiple concurrent users
- Development/testing with limited resources
- Batch processing (latency less critical)

**Use SSD Offloading When:**
- Extremely long contexts (>100K tokens)
- Very limited GPU memory (<16GB)
- Cold storage for inactive sessions
- Cost optimization (cheaper than GPU RAM)

**Avoid Offloading When:**
- Latency-critical applications
- Sufficient GPU memory available
- Short conversations (<5 turns)
- High-throughput requirements
- Real-time inference needs
