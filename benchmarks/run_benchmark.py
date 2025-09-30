#!/usr/bin/env python3
"""
Centralized benchmark runner for vLLM multi-turn conversation with LMCache offloading.
Handles server lifecycle, benchmark execution, and structured results output.
"""

import argparse
import json
import yaml
import subprocess
import time
import os
import signal
import sys
import re
import logging
import shutil
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from dataclasses import dataclass, asdict

# Import bench_dataset functions
sys.path.insert(0, str(Path(__file__).parent))
from bench_dataset import (
    generate_conversations,
    parse_input_json_file,
    conversations_dict_to_list
)
from transformers import AutoTokenizer


# Initial logging configuration (will be updated after output_dir is known)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


@dataclass
class BenchmarkResult:
    """Structured benchmark result"""
    cpu_memory_size: float
    timestamp: str
    server_start_time: float
    benchmark_duration: float
    
    # Offload mode: 'cpu', 'ssd', 'none', or 'hbm'
    offload_mode: str = 'cpu'
    disk_memory_size: Optional[float] = None
    
    # LMCache specific metrics
    lmcache_total_tokens: Optional[int] = None
    lmcache_hit_tokens: Optional[int] = None
    lmcache_hit_rate: Optional[float] = None
    
    # Performance metrics
    requests_per_second: Optional[float] = None
    runtime: Optional[float] = None
    
    # Latency statistics (in ms)
    ttft_ms: Dict[str, float] = None
    tpot_ms: Dict[str, float] = None
    latency_ms: Dict[str, float] = None
    
    # Token counts with statistics
    input_num_tokens: Dict[str, float] = None
    output_num_tokens: Dict[str, float] = None
    
    # Errors
    errors: List[str] = None
    
    def __post_init__(self):
        if self.ttft_ms is None:
            self.ttft_ms = {}
        if self.tpot_ms is None:
            self.tpot_ms = {}
        if self.latency_ms is None:
            self.latency_ms = {}
        if self.input_num_tokens is None:
            self.input_num_tokens = {}
        if self.output_num_tokens is None:
            self.output_num_tokens = {}
        if self.errors is None:
            self.errors = []


class VLLMServerManager:
    """Manages vLLM server lifecycle"""
    
    def __init__(self, model_name: str, timeout: int = 180):
        self.model_name = model_name
        self.timeout = timeout  # timeout in seconds
        self.process = None
        self.log_file = None
        
    def start(self, lmcache_config_path: Path, log_path: Path, cpu_size: float = None, offload_mode: str = 'cpu', use_experimental: bool = False) -> bool:
        """Start vLLM server with appropriate configuration based on cpu_size
        
        Args:
            lmcache_config_path: Path to LMCache config (used when cpu_size > 0)
            log_path: Path to server log file
            cpu_size: CPU memory size. Special values:
                     0: No caching (full recomputation)
                     -1: HBM/GPU-only with prefix caching
                     >0: Use LMCache with offloading
            offload_mode: 'cpu' for CPU offloading, 'ssd' for SSD offloading
            use_experimental: Whether to use experimental features (needed for SSD)
        """
        self.stop()  # Ensure clean state
        
        cmd = ['vllm', 'serve', self.model_name]
        env = os.environ.copy()
        
        if cpu_size == 0:
            # Case 1: No caching, full recomputation
            cmd.extend([
                '--no-enable-prefix-caching',
                '--disable-log-requests'
            ])
            # Don't set LMCACHE_CONFIG_FILE
            
        elif cpu_size == -1:
            # Case 2: HBM/GPU-only with prefix caching
            cmd.extend([
                '--enable-prefix-caching',
                '--disable-log-requests'
            ])
            # Don't set LMCACHE_CONFIG_FILE
            
        else:
            # Case 3: Use LMCache with offloading (cpu_size > 0)
            cmd.extend([
                '--kv-transfer-config',
                json.dumps({"kv_connector": "LMCacheConnectorV1", "kv_role": "kv_both"}),
                '--no-enable-prefix-caching',
                '--disable-log-requests'
            ])
            env['LMCACHE_CONFIG_FILE'] = str(lmcache_config_path)
            if use_experimental:
                env['LMCACHE_USE_EXPERIMENTAL'] = 'True'
        
        try:
            self.log_file = open(log_path, 'w')
            self.process = subprocess.Popen(
                cmd,
                stdout=self.log_file,
                stderr=subprocess.STDOUT,
                env=env,
                preexec_fn=os.setsid  # Create new process group
            )
            
            logger.info(f"Started vLLM server (PID: {self.process.pid})")
            
            # Wait for server to be ready
            if not self._wait_for_ready():
                logger.error("Server failed to start within timeout")
                self.stop()
                return False
                
            return True
            
        except Exception as e:
            logger.error(f"Failed to start server: {e}")
            self.stop()
            return False
    
    def _wait_for_ready(self) -> bool:
        """Wait for server to be ready"""
        import requests
        
        start_time = time.time()
        # vLLM typically uses /v1/models endpoint
        urls_to_try = [
            "http://localhost:8000/v1/models",  # OpenAI compatible endpoint
            "http://localhost:8000/health",      # Health check endpoint
            "http://localhost:8000/metrics"      # Metrics endpoint
        ]
        
        while time.time() - start_time < self.timeout:
            for url in urls_to_try:
                try:
                    response = requests.get(url, timeout=1)
                    if response.status_code == 200:
                        logger.info(f"Server is ready (responded at {url})")
                        return True
                except requests.exceptions.RequestException:
                    pass
            
            # Also check if process is still alive
            if self.process and self.process.poll() is not None:
                logger.error(f"Server process died with return code: {self.process.returncode}")
                # Try to read some log output for debugging
                try:
                    self.log_file.flush()
                    with open(self.log_file.name, 'r') as f:
                        last_lines = f.readlines()[-20:]  # Last 20 lines
                        logger.error("Server log tail:\n" + "".join(last_lines))
                except:
                    pass
                return False
            
            time.sleep(2)
        
        logger.error(f"Server failed to respond within {self.timeout} seconds")
        return False
    
    def stop(self):
        """Stop vLLM server"""
        if self.process:
            try:
                # Kill process group
                os.killpg(os.getpgid(self.process.pid), signal.SIGTERM)
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                os.killpg(os.getpgid(self.process.pid), signal.SIGKILL)
                self.process.wait()
            except ProcessLookupError:
                pass  # Process already dead
            
            self.process = None
            logger.info("Server stopped")
        
        if self.log_file:
            self.log_file.close()
            self.log_file = None


class BenchmarkRunner:
    """Orchestrates benchmark execution"""

    def __init__(self, args):
        self.args = args
        self.sudo_password = None  # Will be set if needed
        
        # Parse benchmark params first
        if args.benchmark_params:
            self.benchmark_params = json.loads(args.benchmark_params)
        else:
            self.benchmark_params = {
                'num_clients': 2,
                'warmup_step': True,
                'no_early_stop': True
            }
        
        # Load workload config
        self.workload_config = self._load_json(args.workload_config)
        
        # Output directory is already the run-specific directory from notebook
        # Don't create another subdirectory
        self.output_dir = Path(args.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Extract run ID from output directory path if it follows the pattern
        output_dir_name = self.output_dir.name
        if output_dir_name.startswith('run_'):
            self.run_id = output_dir_name
        else:
            # Fallback: generate a simple run ID if needed
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            offload_suffix = 'ssd' if getattr(args, 'offload_mode', 'cpu') == 'ssd' else 'cpu'
            self.run_id = f"run_{timestamp}_{offload_suffix}"
        
        # Load LMCache templates
        self.lmcache_template = self._load_yaml(args.lmcache_template)
        
        # Load SSD template if provided
        if hasattr(args, 'lmcache_ssd_template') and args.lmcache_ssd_template:
            self.lmcache_ssd_template = self._load_yaml(args.lmcache_ssd_template)
        else:
            self.lmcache_ssd_template = None
        
        self.server_manager = VLLMServerManager(args.model, timeout=600)
        self.results = []
        
        # Reconfigure logging to include file handler in output directory
        log_file = self.output_dir / 'benchmark_runner.log'
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Add file handler to the current logger (not root to avoid duplication)
        # Remove any existing file handlers from this logger
        for handler in logger.handlers[:]:
            if isinstance(handler, logging.FileHandler):
                logger.removeHandler(handler)
        logger.addHandler(file_handler)
        
        logger.info(f"Run ID: {self.run_id}")
        logger.info(f"Output directory: {self.output_dir}")
        logger.info(f"Log file: {log_file}")

        # Check if sudo is needed and available
        if args.clear_cache_between_turns:
            if not self._check_sudo_available():
                logger.warning("Cache clearing between turns requested but sudo is not available or passwordless")
                logger.warning("Either run with sudo, setup passwordless sudo, or disable cache clearing")
    
    def _load_json(self, path: Path) -> dict:
        with open(path, 'r') as f:
            return json.load(f)
    
    def _load_yaml(self, path: Path) -> dict:
        with open(path, 'r') as f:
            return yaml.safe_load(f)
    
    def _save_yaml(self, data: dict, path: Path):
        with open(path, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)

    def _check_sudo_available(self) -> bool:
        """Check if sudo is available for cache clearing"""
        try:
            # Check if we can run sudo without password
            result = subprocess.run(
                ['sudo', '-n', 'true'],
                capture_output=True,
                timeout=1
            )
            if result.returncode == 0:
                logger.info("Passwordless sudo available for cache clearing")
                return True

            # Check if we're already running as root
            if os.geteuid() == 0:
                logger.info("Running as root, cache clearing available")
                return True

            return False
        except (subprocess.SubprocessError, FileNotFoundError):
            return False

    
    def _create_lmcache_config(self, cpu_size: float, offload_mode: str = 'cpu', disk_size: float = None) -> Path:
        """Create LMCache config for specific CPU/disk size
        
        Args:
            cpu_size: CPU memory size in GB
            offload_mode: 'cpu' or 'ssd'
            disk_size: Disk size in GB (for SSD mode)
        """
        if offload_mode == 'ssd' and self.lmcache_ssd_template:
            config = self.lmcache_ssd_template.copy()
            config['max_local_disk_size'] = disk_size if disk_size is not None else cpu_size
            # Set CPU size to max of disk_size or default
            config['max_local_cpu_size'] = max(disk_size if disk_size is not None else cpu_size, 
                                                config.get('max_local_cpu_size', 5))
            config_path = self.output_dir / f'lmcache_ssd_{disk_size if disk_size is not None else cpu_size}GB.yaml'
        else:
            config = self.lmcache_template.copy()
            config['max_local_cpu_size'] = cpu_size
            config_path = self.output_dir / f'lmcache_{cpu_size}GB.yaml'
        
        self._save_yaml(config, config_path)
        logger.info(f"Created LMCache config ({offload_mode}): {config_path}")
        return config_path
    
    def _generate_workload(self) -> Path:
        """Generate workload dataset"""
        output_path = self.output_dir / 'workload_dataset.json'
        
        # Check if already exists
        if output_path.exists() and not self.args.regenerate_workload:
            logger.info(f"Using existing workload: {output_path}")
            return output_path
        
        logger.info("Generating workload dataset...")
        logger.info(f"Using config with filetype: {self.workload_config.get('filetype', 'unknown')}")
        
        try:
            # Parse the config using bench_dataset's parser
            gen_conv_args = parse_input_json_file(self.workload_config)
            
            # Disable warning from tokenizers when using multiprocessing
            os.environ["TOKENIZERS_PARALLELISM"] = "true"
            
            # Initialize tokenizer
            tokenizer = AutoTokenizer.from_pretrained(self.args.model)
            
            # Generate synthetic conversations with seed for consistency
            conversations = generate_conversations(gen_conv_args, tokenizer, seed=42)
            
            # Convert to ShareGPT format (list of dicts)
            sharegpt_data = conversations_dict_to_list(conversations)
            
            # Save to file
            with open(output_path, 'w') as f:
                json.dump(sharegpt_data, f, indent=2)
            
            logger.info(f"Generated {len(sharegpt_data)} conversations")
            logger.info(f"Saved workload to: {output_path}")
            
        except Exception as e:
            logger.error(f"Failed to generate workload: {e}")
            raise RuntimeError(f"Workload generation failed: {e}")
        
        return output_path
    
    def _run_single_benchmark(self, workload_path: Path, cpu_size: float, offload_mode: str = 'cpu', disk_size: float = None) -> BenchmarkResult:
        """Run benchmark for single configuration
        
        Args:
            workload_path: Path to workload file
            cpu_size: CPU memory size in GB
            offload_mode: 'cpu', 'ssd', 'none', or 'hbm'
            disk_size: Disk size in GB (for SSD mode)
        """
        timestamp = datetime.now().isoformat()
        
        # Determine actual offload mode based on cpu_size
        if cpu_size == 0:
            actual_offload_mode = 'none'
        elif cpu_size == -1:
            actual_offload_mode = 'hbm'
        else:
            actual_offload_mode = offload_mode
        
        result = BenchmarkResult(
            cpu_memory_size=cpu_size,
            timestamp=timestamp,
            server_start_time=0,
            benchmark_duration=0,
            offload_mode=actual_offload_mode,
            disk_memory_size=disk_size
        )
        
        # Create LMCache config only for offloading cases (cpu_size > 0)
        lmcache_config = None
        use_experimental = False
        if cpu_size > 0:
            lmcache_config = self._create_lmcache_config(cpu_size, offload_mode, disk_size)
            use_experimental = (offload_mode == 'ssd')
        
        # Start server with appropriate configuration
        # Use special log file naming for different cases
        if cpu_size == 0:
            server_log = self.output_dir / 'server_no_cache.log'
        elif cpu_size == -1:
            server_log = self.output_dir / 'server_hbm.log'
        elif offload_mode == 'ssd':
            server_log = self.output_dir / f'server_ssd_{disk_size if disk_size else cpu_size}GB.log'
        else:
            server_log = self.output_dir / f'server_{cpu_size}GB.log'
            
        server_start = time.time()
        
        if not self.server_manager.start(lmcache_config, server_log, cpu_size, offload_mode, use_experimental):
            result.errors.append("Failed to start server")
            return result
        
        result.server_start_time = time.time() - server_start
        
        try:
            # Run benchmark
            # Use special output file naming for different cases
            if cpu_size == 0:
                benchmark_output = self.output_dir / 'benchmark_no_cache.txt'
            elif cpu_size == -1:
                benchmark_output = self.output_dir / 'benchmark_hbm.txt'
            elif offload_mode == 'ssd':
                benchmark_output = self.output_dir / f'benchmark_ssd_{disk_size if disk_size else cpu_size}GB.txt'
            else:
                benchmark_output = self.output_dir / f'benchmark_{cpu_size}GB.txt'
            cmd = [
                'python', 'benchmark_serving_multi_turn.py',
                '--model', self.args.model,
                '--input-file', str(workload_path),
                '--num-clients', str(self.benchmark_params.get('num_clients', 2)),
                '--max-active-conversations', str(self.benchmark_params.get('max_active_conversations', self.workload_config.get('num_conversations', 24))),
                '--lmcache-log-file', str(server_log)  # Pass the correct log file path
            ]
            
            if self.benchmark_params.get('no_early_stop'):
                cmd.append('--no-early-stop')
            if self.benchmark_params.get('warmup_step'):
                cmd.append('--warmup-step')

            # Pass cache clearing flag if needed for between-turns clearing
            if self.args.clear_cache_between_turns and offload_mode == 'ssd':
                cmd.append('--clear-cache-between-turns')
            
            logger.info(f"Running benchmark for {cpu_size}GB...")
            benchmark_start = time.time()
            
            with open(benchmark_output, 'w') as f:
                proc = subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, text=True)
            
            result.benchmark_duration = time.time() - benchmark_start
            
            if proc.returncode != 0:
                result.errors.append(f"Benchmark exited with code {proc.returncode}")
            
            # Parse results
            self._parse_benchmark_output(benchmark_output, result)
            
            # Parse server log for cache metrics
            self._parse_server_log(server_log, result)
            
        except Exception as e:
            logger.error(f"Benchmark failed: {e}")
            result.errors.append(str(e))
        
        finally:
            self.server_manager.stop()
        
        return result
    
    def _parse_benchmark_output(self, output_path: Path, result: BenchmarkResult):
        """Parse benchmark output file"""
        try:
            with open(output_path, 'r') as f:
                content = f.read()
            
            # Parse performance metrics
            patterns = {
                'requests_per_second': r'requests_per_sec\s*=\s*([0-9.]+)',
                'runtime': r'runtime_sec\s*=\s*([0-9.]+)',
                'input_tokens_total': r'Total input tokens:\s*(\d+)',
                'output_tokens_total': r'Total output tokens:\s*(\d+)',
            }
            
            for field, pattern in patterns.items():
                match = re.search(pattern, content, re.IGNORECASE)
                if match:
                    setattr(result, field, float(match.group(1)))
            
            # Parse LMCache metrics from benchmark output
            lmcache_tokens_match = re.search(r'lmcache_cpu_tokens\s*=\s*([0-9,]+)', content)
            if lmcache_tokens_match:
                result.lmcache_total_tokens = int(lmcache_tokens_match.group(1).replace(',', ''))
            
            lmcache_hits_match = re.search(r'lmcache_cpu_hits\s*=\s*([0-9,]+)', content)
            if lmcache_hits_match:
                result.lmcache_hit_tokens = int(lmcache_hits_match.group(1).replace(',', ''))
            
            lmcache_rate_match = re.search(r'lmcache_cpu_hit_rate\s*=\s*([0-9.]+)', content)
            if lmcache_rate_match:
                result.lmcache_hit_rate = float(lmcache_rate_match.group(1))
            
            # Parse latency metrics with statistics
            self._parse_latency_metrics(content, result)
            
        except Exception as e:
            logger.error(f"Failed to parse benchmark output: {e}")
            result.errors.append(f"Parse error: {e}")
    
    def _parse_latency_metrics(self, content: str, result: BenchmarkResult):
        """Parse latency metrics with percentiles"""
        # The table format is: metric_name count mean std min 25% 50% 75% 90% 99% max
        # We need to extract: mean, min, 25%, 50%, 75%, 90%, 99%, max
        metrics_to_parse = ['ttft_ms', 'tpot_ms', 'latency_ms', 'input_num_tokens', 'output_num_tokens']
        
        for metric in metrics_to_parse:
            # Pattern matches: metric_name count mean std min 25% 50% 75% 90% 99% max
            pattern = rf'^{metric}\s+[\d.]+\s+([\d.]+)\s+[\d.]+\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)'
            match = re.search(pattern, content, re.MULTILINE)
            
            if match:
                metric_dict = {
                    'mean': float(match.group(1)),  # mean
                    'min': float(match.group(2)),   # min
                    '25%': float(match.group(3)),   # 25%
                    '50%': float(match.group(4)),   # 50%
                    '75%': float(match.group(5)),   # 75%
                    '90%': float(match.group(6)),   # 90%
                    '99%': float(match.group(7)),   # 99%
                    'max': float(match.group(8))    # max
                }
                setattr(result, metric, metric_dict)
    
    def _parse_server_log(self, log_path: Path, result: BenchmarkResult):
        """Parse server log for any additional metrics if needed"""
        # The LMCache metrics are already parsed by benchmark_serving_multi_turn.py
        # and included in its output, so we don't need to parse them here
        pass
    
    def run(self):
        """Run benchmarks for all CPU sizes"""
        logger.info("Starting benchmark suite")
        offload_mode = getattr(self.args, 'offload_mode', 'cpu')
        logger.info(f"Offload mode: {offload_mode}")
        logger.info(f"CPU sizes to test: {self.args.cpu_sizes}")
        
        # Track overall success
        has_failures = False
        critical_failure = False
        
        try:
            # Generate workload
            workload_path = self._generate_workload()
        except Exception as e:
            logger.error(f"Failed to generate workload: {e}")
            raise
        
        # Run benchmarks
        for cpu_size in self.args.cpu_sizes:
            logger.info(f"\n{'='*60}")
            if offload_mode == 'ssd' and cpu_size > 0:
                logger.info(f"Testing SSD disk size: {cpu_size}GB")
            else:
                logger.info(f"Testing CPU size: {cpu_size}GB")
            logger.info(f"{'='*60}")
            
            # For SSD mode, cpu_size represents disk size
            if offload_mode == 'ssd' and cpu_size > 0:
                result = self._run_single_benchmark(workload_path, cpu_size, offload_mode='ssd', disk_size=cpu_size)
            else:
                result = self._run_single_benchmark(workload_path, cpu_size, offload_mode='cpu')
            self.results.append(result)
            
            # Check for critical failures
            if result.errors:
                has_failures = True
                # Check if this is a critical error (server failed to start)
                if any('Failed to start server' in err for err in result.errors):
                    critical_failure = True
                    logger.error(f"Critical failure for {cpu_size}GB: Server failed to start")
            
            # Log summary with special case naming
            if cpu_size == 0:
                logger.info(f"Summary for No Cache (full recomputation):")
            elif cpu_size == -1:
                logger.info(f"Summary for HBM (GPU prefix caching):")
            elif offload_mode == 'ssd' and cpu_size > 0:
                logger.info(f"Summary for SSD {cpu_size}GB:")
            else:
                logger.info(f"Summary for CPU {cpu_size}GB:")
                
            if result.lmcache_hit_rate is not None:
                logger.info(f"  LMCache hit rate: {result.lmcache_hit_rate:.2f}%")
                logger.info(f"  LMCache tokens: {result.lmcache_total_tokens}, hits: {result.lmcache_hit_tokens}")
            else:
                logger.info("  LMCache hit rate: N/A")
            logger.info(f"  TTFT mean: {result.ttft_ms.get('mean', 'N/A')} ms")
            logger.info(f"  Throughput: {result.requests_per_second} req/s" if result.requests_per_second else "  Throughput: N/A")
            if result.errors:
                logger.warning(f"  Errors: {result.errors}")
            
            # Clean up SSD cache directory between runs if in SSD mode
            if offload_mode == 'ssd' and self.lmcache_ssd_template:
                disk_path = self.lmcache_ssd_template.get('local_disk')
                if disk_path and os.path.exists(disk_path):
                    logger.info(f"Cleaning up SSD cache directory: {disk_path}")
                    try:
                        import shutil
                        # Remove all contents but keep the directory
                        for item in os.listdir(disk_path):
                            item_path = os.path.join(disk_path, item)
                            if os.path.isdir(item_path):
                                shutil.rmtree(item_path)
                            else:
                                os.remove(item_path)
                        logger.info(f"Successfully cleaned up {disk_path}")
                    except Exception as e:
                        logger.warning(f"Failed to clean up SSD cache: {e}")

            # Small delay between runs
            time.sleep(5)
        
        # Save results even if there were failures
        self._save_results()
        
        # Report final status and exit accordingly
        if critical_failure:
            logger.error("Benchmark suite completed with CRITICAL FAILURES")
            raise RuntimeError("One or more benchmarks failed critically (server startup failed)")
        elif has_failures:
            logger.warning("Benchmark suite completed with some failures")
            # Still save results but indicate partial success
        else:
            logger.info("Benchmark suite completed successfully")
    
    def _save_results(self):
        """Save results to JSON"""
        output_file = self.output_dir / 'benchmark_summary.json'
        
        # Convert to serializable format
        results_dict = {
            'metadata': {
                'timestamp': datetime.now().isoformat(),
                'model': self.args.model,
                'workload_config': str(self.args.workload_config),
                'lmcache_template': str(self.args.lmcache_template),
                'lmcache_ssd_template': str(getattr(self.args, 'lmcache_ssd_template', None)),
                'cpu_sizes': self.args.cpu_sizes,
                'benchmark_params': self.benchmark_params,
                'offload_mode': getattr(self.args, 'offload_mode', 'cpu'),
                'run_id': self.run_id
            },
            'results': [asdict(r) for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to: {output_file}")
        
        # Also save as CSV for easy analysis
        self._save_csv()
    
    def _save_csv(self):
        """Save results as CSV"""
        import csv
        
        csv_file = self.output_dir / 'benchmark_summary.csv'
        
        if not self.results:
            return
        
        # Flatten the results
        rows = []
        for r in self.results:
            row = {
                'cpu_memory_size': r.cpu_memory_size,
                'config_type': 'HBM' if r.cpu_memory_size == -1 else ('No Cache' if r.cpu_memory_size == 0 else (f'SSD Offload' if r.offload_mode == 'ssd' else 'CPU Offload')),
                'offload_mode': r.offload_mode,
                'disk_memory_size': r.disk_memory_size if r.disk_memory_size else None,
                'timestamp': r.timestamp,
                'lmcache_hit_rate': r.lmcache_hit_rate if r.lmcache_hit_rate is not None else 0,
                'requests_per_second': r.requests_per_second,
                'runtime': r.runtime,
                'server_start_time': r.server_start_time,
                'benchmark_duration': r.benchmark_duration
            }
            
            # Add latency metrics
            for metric in ['ttft_ms', 'tpot_ms', 'latency_ms']:
                metric_dict = getattr(r, metric, {})
                for stat in ['mean', 'min', '50%', '90%', '99%', 'max']:
                    row[f'{metric}_{stat}'] = metric_dict.get(stat)
            
            rows.append(row)
        
        # Write CSV
        with open(csv_file, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
        
        logger.info(f"CSV saved to: {csv_file}")


def main():
    parser = argparse.ArgumentParser(
        description='Run vLLM multi-turn benchmarks with LMCache offloading'
    )
    
    parser.add_argument(
        '--workload-config', 
        type=Path, 
        required=True,
        help='Path to workload JSON configuration'
    )
    
    parser.add_argument(
        '--lmcache-template', 
        type=Path, 
        required=True,
        help='Path to LMCache template YAML'
    )
    
    parser.add_argument(
        '--cpu-sizes', 
        type=float, 
        nargs='+', 
        default=[20, 10, 5, 1, 0.1],
        help='List of CPU memory sizes to test (in GB)'
    )
    
    parser.add_argument(
        '--output-dir', 
        type=Path, 
        default=Path('benchmark_results'),
        help='Directory for results (default: benchmark_results)'
    )
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='meta-llama/Llama-3.1-8B-Instruct',
        help='Model name (default: meta-llama/Llama-3.1-8B-Instruct)'
    )
    
    parser.add_argument(
        '--benchmark-params', 
        type=str,
        help='JSON string for benchmark settings'
    )
    
    parser.add_argument(
        '--regenerate-workload',
        action='store_true',
        help='Force regeneration of workload dataset'
    )
    
    parser.add_argument(
        '--run-id',
        type=str,
        help='Optional run ID for organizing results (auto-generated if not provided)'
    )
    
    parser.add_argument(
        '--offload-mode',
        type=str,
        choices=['cpu', 'ssd'],
        default='cpu',
        help='Offload mode: cpu or ssd (default: cpu)'
    )
    
    parser.add_argument(
        '--lmcache-ssd-template',
        type=Path,
        help='Path to LMCache SSD template YAML (required for SSD mode)'
    )

    parser.add_argument(
        '--clear-cache-between-turns',
        action='store_true',
        help='Clear OS page cache between conversation turns (requires sudo, for fine-grained SSD benchmarking)'
    )
    
    args = parser.parse_args()
    
    # Run benchmarks
    runner = BenchmarkRunner(args)
    
    try:
        runner.run()
        sys.exit(0)  # Explicit success
    except KeyboardInterrupt:
        logger.info("\nBenchmark interrupted by user")
        runner.server_manager.stop()
        sys.exit(130)  # Standard exit code for SIGINT
    except RuntimeError as e:
        logger.error(f"Benchmark failed: {e}")
        runner.server_manager.stop()
        sys.exit(2)  # Exit code 2 for runtime/critical errors
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        runner.server_manager.stop()
        sys.exit(1)  # Exit code 1 for general errors


if __name__ == '__main__':
    main()