#!/usr/bin/env python3
"""
Parse LMCache logs to extract CPU KV cache hit rate metrics.

This script reads LMCache logs and calculates:
- Total tokens processed
- Total tokens hit in CPU cache
- CPU KV cache hit rate
- Per-request hit statistics
"""

import re
import sys
import json
from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path


@dataclass
class RequestStats:
    """Statistics for a single request."""
    request_id: str
    total_tokens: int
    hit_tokens: int
    tokens_to_load: int
    hit_rate: float


def parse_lmcache_logs(log_file: str) -> tuple[List[RequestStats], dict]:
    """
    Parse LMCache log file to extract KV cache hit statistics.
    
    Returns:
        Tuple of (list of RequestStats, summary statistics dict)
    """
    requests = []
    
    # Patterns to match in logs
    # Example: "Reqid: 2, Total tokens 338, LMCache hit tokens: 256, need to load: -32"
    hit_pattern = re.compile(
        r'Reqid:\s*(\S+),\s*Total tokens\s*(\d+),\s*LMCache hit tokens:\s*(\d+),\s*need to load:\s*(-?\d+)'
    )
    
    # Pattern for stored tokens (offloading to CPU)
    # Example: "Stored 256 out of total 256 tokens. size: 0.0312 gb, cost 2.0444 ms"
    store_pattern = re.compile(
        r'Stored\s*(\d+)\s*out of total\s*(\d+)\s*tokens.*size:\s*([\d.]+)\s*gb.*cost\s*([\d.]+)\s*ms'
    )
    
    total_tokens_processed = 0
    total_tokens_hit = 0
    total_tokens_stored = 0
    total_store_operations = 0
    
    with open(log_file, 'r') as f:
        for line in f:
            # Check for hit information
            match = hit_pattern.search(line)
            if match:
                request_id = match.group(1)
                total_tokens = int(match.group(2))
                hit_tokens = int(match.group(3))
                tokens_to_load = int(match.group(4))
                
                hit_rate = (hit_tokens / total_tokens * 100) if total_tokens > 0 else 0
                
                request = RequestStats(
                    request_id=request_id,
                    total_tokens=total_tokens,
                    hit_tokens=hit_tokens,
                    tokens_to_load=tokens_to_load,
                    hit_rate=hit_rate
                )
                requests.append(request)
                
                total_tokens_processed += total_tokens
                total_tokens_hit += hit_tokens
            
            # Check for store operations (CPU offloading)
            match = store_pattern.search(line)
            if match:
                stored_tokens = int(match.group(1))
                total_tokens_stored += stored_tokens
                total_store_operations += 1
    
    # Calculate summary statistics
    overall_hit_rate = (total_tokens_hit / total_tokens_processed * 100) if total_tokens_processed > 0 else 0
    
    summary = {
        'total_requests': len(requests),
        'total_tokens_processed': total_tokens_processed,
        'total_tokens_hit': total_tokens_hit,
        'total_tokens_stored_to_cpu': total_tokens_stored,
        'total_store_operations': total_store_operations,
        'overall_hit_rate': overall_hit_rate,
        'average_hit_rate': sum(r.hit_rate for r in requests) / len(requests) if requests else 0,
        'min_hit_rate': min((r.hit_rate for r in requests), default=0),
        'max_hit_rate': max((r.hit_rate for r in requests), default=0)
    }
    
    return requests, summary


def print_statistics(requests: List[RequestStats], summary: dict):
    """Print formatted statistics."""
    print("\n" + "="*80)
    print("LMCache CPU KV Cache Hit Rate Analysis")
    print("="*80)
    
    print("\n📊 Summary Statistics:")
    print("-"*40)
    print(f"Total Requests:           {summary['total_requests']}")
    print(f"Total Tokens Processed:   {summary['total_tokens_processed']:,}")
    print(f"Total Tokens Hit (CPU):   {summary['total_tokens_hit']:,}")
    print(f"Total Tokens Stored:      {summary['total_tokens_stored_to_cpu']:,}")
    print(f"Store Operations:         {summary['total_store_operations']}")
    print(f"\n🎯 CPU KV Cache Hit Rate: {summary['overall_hit_rate']:.2f}%")
    print(f"Average Hit Rate:         {summary['average_hit_rate']:.2f}%")
    print(f"Min Hit Rate:             {summary['min_hit_rate']:.2f}%")
    print(f"Max Hit Rate:             {summary['max_hit_rate']:.2f}%")
    
    if requests:
        print("\n📝 Per-Request Details:")
        print("-"*40)
        print(f"{'Request ID':<15} {'Total Tokens':<12} {'Hit Tokens':<12} {'Hit Rate':<10}")
        print("-"*40)
        
        for req in requests[:20]:  # Show first 20 requests
            print(f"{req.request_id:<15} {req.total_tokens:<12} {req.hit_tokens:<12} {req.hit_rate:>6.2f}%")
        
        if len(requests) > 20:
            print(f"... and {len(requests) - 20} more requests")
    
    print("\n" + "="*80)


def export_to_json(requests: List[RequestStats], summary: dict, output_file: str):
    """Export statistics to JSON file."""
    data = {
        'summary': summary,
        'requests': [
            {
                'request_id': r.request_id,
                'total_tokens': r.total_tokens,
                'hit_tokens': r.hit_tokens,
                'tokens_to_load': r.tokens_to_load,
                'hit_rate': r.hit_rate
            }
            for r in requests
        ]
    }
    
    with open(output_file, 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"\n📁 Results exported to: {output_file}")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python parse_lmcache_cpu_hits.py <log_file> [--export <output.json>]")
        print("\nExample:")
        print("  python parse_lmcache_cpu_hits.py server.log")
        print("  python parse_lmcache_cpu_hits.py server.log --export results.json")
        sys.exit(1)
    
    log_file = sys.argv[1]
    
    if not Path(log_file).exists():
        print(f"Error: Log file '{log_file}' not found")
        sys.exit(1)
    
    # Parse the logs
    requests, summary = parse_lmcache_logs(log_file)
    
    # Print statistics
    print_statistics(requests, summary)
    
    # Export to JSON if requested
    if len(sys.argv) >= 4 and sys.argv[2] == '--export':
        export_to_json(requests, summary, sys.argv[3])
    
    # Also compare with vLLM's reported metrics if available
    print("\n💡 Tips:")
    print("- Compare these values with vLLM's benchmark output (if available)")
    print("- Monitor CPU memory usage to ensure offloading is working")
    print("- Higher hit rates indicate better cache utilization")
    print("- Use --export flag to save results for further analysis")


if __name__ == "__main__":
    main()