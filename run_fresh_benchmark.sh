#!/bin/bash
# Script to run benchmark with fresh KV cache

# Kill any existing vLLM server
echo "Stopping existing vLLM server..."
pkill -f "vllm serve" || true
sleep 2

# Start vLLM server
echo "Starting fresh vLLM server..."
export MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
vllm serve $MODEL_NAME --disable-log-requests --enable-prefix-caching &
SERVER_PID=$!

# Wait for server to be ready
echo "Waiting for server to start..."
for i in {1..30}; do
    if curl -s http://localhost:8000/health > /dev/null 2>&1; then
        echo "Server is ready!"
        break
    fi
    sleep 1
done

# Run the benchmark
echo "Running benchmark..."
cd benchmarks
python benchmark_serving_multi_turn.py "$@"

# Kill the server after benchmark
echo "Stopping vLLM server..."
kill $SERVER_PID
wait $SERVER_PID 2>/dev/null

echo "Benchmark complete with fresh KV cache!"