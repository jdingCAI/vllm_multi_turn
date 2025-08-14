Regarding KV cache hit and host memory offloading

1. Use Prometheus Metrics

  vLLM exposes several cache-related metrics via the /metrics endpoint:

  curl http://localhost:8000/metrics | grep cache

  Key metrics include:
  - vllm:gpu_cache_usage_perc - GPU cache utilization percentage
  - vllm:cpu_cache_usage_perc - CPU cache utilization percentage
  - vllm:gpu_prefix_cache_hit_rate - Cache hit rate
  - vllm:num_requests_swapped - Number of requests currently swapped to CPU

2. Use vLLM's Built-in Profiling

  For more detailed analysis, you can use vLLM's profiling capabilities:

  # Enable CUDA profiling
  export VLLM_TORCH_PROFILER_DIR=/path/to/profile/output

  # Or use nsys for system-wide profiling
  nsys profile -o vllm_profile python -m vllm serve model_name


# Summary:
  To see KV cache offloading time in vLLM:

  1. Use existing metrics: The benchmark already shows KV cache hit rates in the output
  2. Monitor swap counts: Check vllm:num_requests_swapped metric via /metrics endpoint
  3. Add custom timing: Use the measure_kv_cache_timing.py script I created to wrap cache operations with timing
  4. System monitoring: Use the monitor_kv_cache_system.py script to track cache usage in real-time
  5. Enable debug logging: Set VLLM_LOGGING_LEVEL=DEBUG for more detailed logs

  The actual swap timing depends on:
  - PCIe bandwidth (typically 15-30 GB/s)
  - Block size (depends on model configuration)
  - Number of blocks being swapped
  - System memory bandwidth

  For production monitoring, I recommend using Prometheus to collect the /metrics endpoint data and visualize it in Grafana to track KV cache performance over time.

