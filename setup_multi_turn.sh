#!/bin/bash

# vLLM Multi-Turn Benchmark Setup Script
# This script sets up vLLM with the multi-turn benchmark branch and prepares the environment

set -e  # Exit on any error

echo "=== vLLM Multi-Turn Benchmark Setup ==="

# # Step 1: Clone the specific branch
# echo "Cloning vLLM multi-turn benchmark branch..."
# if [ -d "vllm" ]; then
#     echo "vllm directory already exists. Removing it..."
#     rm -rf vllm
# fi

# git clone -b multi_turn_benchmark https://github.com/pliops/vllm.git
# cd vllm

# Step 2: Install vLLM in editable mode with precompiled binaries
echo "Installing vLLM in editable mode with precompiled binaries..."
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Step 3: Set model name environment variable
echo "Setting MODEL_NAME environment variable..."
export MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct
echo "MODEL_NAME set to: $MODEL_NAME"

# Step 4: Prepare benchmark files
echo "Preparing benchmark files..."
cd benchmarks

# Download benchmark text file
echo "Downloading benchmark text file..."
if [ ! -f "pg1184.txt" ]; then
    wget https://www.gutenberg.org/ebooks/1184.txt.utf-8
    mv 1184.txt.utf-8 pg1184.txt
    echo "Downloaded and renamed benchmark text file to pg1184.txt"
else
    echo "pg1184.txt already exists, skipping download"
fi

# Step 5: Fix the benchmark script by commenting out the path check
echo "Fixing benchmark_serving_multi_turn.py..."
if [ -f "benchmark_serving_multi_turn.py" ]; then
    # Create backup if it doesn't exist
    if [ ! -f "benchmark_serving_multi_turn.py.backup" ]; then
        cp benchmark_serving_multi_turn.py benchmark_serving_multi_turn.py.backup
        echo "Created backup of benchmark_serving_multi_turn.py"
    fi
    
    # Comment out the problematic lines
    sed -i 's/^    if not os\.path\.exists(args\.model):$/    # if not os.path.exists(args.model):/' benchmark_serving_multi_turn.py
    sed -i 's/^        raise OSError(f"Path does not exist: {args\.model}")$/        # raise OSError(f"Path does not exist: {args.model}")/' benchmark_serving_multi_turn.py
    
    echo "Fixed benchmark script - commented out path existence check"
else
    echo "Warning: benchmark_serving_multi_turn.py not found"
fi

echo ""
echo "=== Setup Complete! ==="
echo ""
echo "To start the vLLM server, run:"
echo "  cd vllm"
echo "  export MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "  vllm serve \$MODEL_NAME --disable-log-requests"
echo ""
echo "To run benchmarks (in another terminal), run:"
echo "  cd vllm/benchmarks"
echo "  export MODEL_NAME=meta-llama/Meta-Llama-3.1-8B-Instruct"
echo "  python benchmark_serving_multi_turn.py --model \$MODEL_NAME --input-file generate_multi_turn.json --num-clients 2 --max-active-conversations 6"
echo ""
echo "Or run the benchmark without model parameter:"
echo "  python benchmark_serving_multi_turn.py --input-file generate_multi_turn.json --num-clients 2 --max-active-conversations 6"
echo ""
echo "Note: Make sure to export MODEL_NAME in each new terminal session"