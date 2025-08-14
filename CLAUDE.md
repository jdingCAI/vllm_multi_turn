# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

vLLM is a high-throughput and memory-efficient inference and serving engine for Large Language Models (LLMs). This repository is currently on the `multi_turn_benchmark` branch, focused on multi-turn conversation benchmarking capabilities.

## Development Commands

### Build and Installation
```bash
# Development installation (editable)
pip install --editable .

# Fast installation with precompiled binaries
VLLM_USE_PRECOMPILED=1 pip install --editable .

# Install development dependencies
pip install -r requirements/dev.txt
pip install -r requirements/test.txt
pip install -r requirements/lint.txt
```

### Code Quality and Linting
```bash
# Setup pre-commit hooks (replaces legacy format.sh)
pre-commit install

# Run all pre-commit checks manually
pre-commit run --all-files

# Type checking
./tools/mypy.sh

# Legacy format script (deprecated)
./format.sh
```

### Testing
```bash
# Run all tests
pytest tests/

# Run specific test categories
pytest tests/basic_correctness/
pytest tests/model_executor/
pytest tests/distributed/

# Run benchmarks
python benchmarks/benchmark_serving_multi_turn.py
python benchmarks/benchmark_throughput.py
python benchmarks/benchmark_latency.py

# Multi-turn specific scripts
./setup_multi_turn.sh
./run_fresh_benchmark.sh
```

### Build System
- Uses setuptools with CMake for C++/CUDA extensions
- Requires PyTorch 2.7.0, CMake ≥3.26.1, Ninja
- Supports CUDA, ROCm, CPU, and TPU targets
- Target device controlled by `VLLM_TARGET_DEVICE` environment variable

## Architecture Overview

### Core Components
- **`vllm/engine/`** - Main LLM engine (async/sync implementations)
- **`vllm/core/`** - Scheduler, block manager, and core execution logic
- **`vllm/model_executor/`** - Model loading, execution, and weight management
- **`vllm/worker/`** - Worker implementations for different hardware backends
- **`vllm/attention/`** - PagedAttention and attention mechanism implementations
- **`vllm/distributed/`** - Multi-GPU and distributed inference support

### Hardware Support
- **`vllm/platforms/`** - Hardware abstraction layer (CUDA, ROCm, TPU, CPU)
- **`csrc/`** - C++/CUDA kernels for optimized operations
- **`vllm/v1/`** - Next-generation engine implementation

### API and Serving
- **`vllm/entrypoints/`** - OpenAI-compatible API server and CLI interfaces
- **`vllm/multimodal/`** - Multi-modal support (images, audio, video)
- **Entry point**: `vllm` command defined in pyproject.toml

### Key Data Flow
1. Requests enter through entrypoints (API server or CLI)
2. Engine schedules and batches requests using core scheduler
3. Model executor loads and runs models on workers
4. Attention mechanisms handle memory-efficient computation
5. Results streamed back through serving infrastructure

## Development Patterns

### Model Addition
- Models in `vllm/model_executor/models/`
- Follow model registration system in `vllm/model_executor/models/__init__.py`
- Add tests in `tests/models/`
- Support multi-modal inputs through `vllm/multimodal/`

### Hardware Backend Development
- Platform-specific code in `vllm/platforms/`
- CUDA kernels in `csrc/`
- Worker implementations in `vllm/worker/`
- Use hardware detection from `vllm/envs.py`

### Testing Strategy
- Unit tests for individual components
- Integration tests for end-to-end workflows
- Model-specific correctness tests
- Performance benchmarks in `benchmarks/`
- Hardware-specific test markers in pytest

## Code Quality Standards

### Pre-commit Hooks
- **yapf** for Python formatting
- **ruff** for linting (line length: 80 chars)
- **isort** for import sorting
- **clang-format** for C++/CUDA code
- **mypy** for static type checking
- **typos** for spell checking
- **DCO sign-off** for all commits

### Type Checking
- MyPy configuration in pyproject.toml
- Gradual typing adoption - some directories excluded
- Python 3.9-3.12 compatibility required

### Contributing Requirements
- Issues required for large changes (>500 lines)
- PR titles prefixed with category: `[Bugfix]`, `[Model]`, `[Kernel]`, etc.
- Signed-off-by required for all commits (DCO)
- Tests must pass and new functionality needs test coverage

## Current Branch Focus

The `multi_turn_benchmark` branch includes:
- Multi-turn conversation benchmarking tools
- KV cache metrics testing (`test_kv_metrics.py`)
- Enhanced benchmark serving scripts (`benchmark_serving_multi_turn.py`)
- Fresh benchmark execution scripts (`run_fresh_benchmark.sh`)