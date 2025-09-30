#!/usr/bin/env python3
"""
Analyze token counts for conversation turns in the workload dataset.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer
from typing import List, Dict

def count_tokens_per_turn(conversation: Dict, tokenizer: AutoTokenizer) -> List[Dict]:
    """Count tokens for each turn in a conversation."""
    
    results = []
    turn_num = 0
    
    # Process messages in pairs (user + assistant = 1 turn)
    for i in range(0, len(conversation['messages']), 2):
        if i+1 >= len(conversation['messages']):
            break
            
        turn_num += 1
        user_msg = conversation['messages'][i]
        assistant_msg = conversation['messages'][i+1]
        
        # Tokenize messages
        user_tokens = tokenizer.encode(user_msg['content'], add_special_tokens=False)
        assistant_tokens = tokenizer.encode(assistant_msg['content'], add_special_tokens=False)
        
        # Store results
        turn_data = {
            'turn': turn_num,
            'user_tokens': len(user_tokens),
            'assistant_tokens': len(assistant_tokens),
            'total_tokens': len(user_tokens) + len(assistant_tokens),
            'user_preview': user_msg['content'][:100] + '...' if len(user_msg['content']) > 100 else user_msg['content'],
            'assistant_preview': assistant_msg['content'][:100] + '...' if len(assistant_msg['content']) > 100 else assistant_msg['content']
        }
        results.append(turn_data)
    
    return results

def main():
    # Load the workload dataset
    workload_path = Path('benchmark_results/workload_dataset.json')
    
    if not workload_path.exists():
        print(f"Error: {workload_path} not found")
        return
    
    with open(workload_path, 'r') as f:
        data = json.load(f)
    
    # Find CONV_ID_0
    conv_id_0 = None
    for conv in data:
        if conv['id'] == 'CONV_ID_0':
            conv_id_0 = conv
            break
    
    if not conv_id_0:
        print("Error: CONV_ID_0 not found in dataset")
        return
    
    # Initialize tokenizer (using the same model as benchmarks)
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    print(f"Loading tokenizer for {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Analyze token counts
    print(f"\n{'='*80}")
    print(f"Token Analysis for CONV_ID_0")
    print(f"{'='*80}")
    print(f"Total messages: {len(conv_id_0['messages'])}")
    print(f"Total turns: {len(conv_id_0['messages']) // 2}")
    
    # Count tokens for each turn
    turn_results = count_tokens_per_turn(conv_id_0, tokenizer)
    
    # Display results
    print(f"\n{'='*80}")
    print(f"{'Turn':<6} {'User Tokens':<15} {'Assistant Tokens':<20} {'Total':<10}")
    print(f"{'-'*80}")
    
    total_user_tokens = 0
    total_assistant_tokens = 0
    
    for turn in turn_results:
        print(f"{turn['turn']:<6} {turn['user_tokens']:<15} {turn['assistant_tokens']:<20} {turn['total_tokens']:<10}")
        total_user_tokens += turn['user_tokens']
        total_assistant_tokens += turn['assistant_tokens']
    
    print(f"{'-'*80}")
    print(f"{'Total':<6} {total_user_tokens:<15} {total_assistant_tokens:<20} {total_user_tokens + total_assistant_tokens:<10}")
    
    # Calculate averages
    avg_user = total_user_tokens / len(turn_results)
    avg_assistant = total_assistant_tokens / len(turn_results)
    avg_total = (total_user_tokens + total_assistant_tokens) / len(turn_results)
    
    print(f"{'Avg':<6} {avg_user:<15.1f} {avg_assistant:<20.1f} {avg_total:<10.1f}")
    
    # Show detailed breakdown with message previews
    print(f"\n{'='*80}")
    print("Detailed Turn Breakdown:")
    print(f"{'='*80}")
    
    for turn in turn_results:
        print(f"\nTurn {turn['turn']}:")
        print(f"  User ({turn['user_tokens']} tokens): {turn['user_preview']}")
        print(f"  Assistant ({turn['assistant_tokens']} tokens): {turn['assistant_preview']}")
    
    # Calculate cumulative tokens for KV cache analysis
    print(f"\n{'='*80}")
    print("Cumulative Token Count (for KV cache sizing):")
    print(f"{'='*80}")
    
    cumulative = 0
    print(f"{'Turn':<6} {'Turn Tokens':<15} {'Cumulative':<15} {'KV Cache Size (approx)':<25}")
    print(f"{'-'*80}")
    
    for turn in turn_results:
        cumulative += turn['total_tokens']
        # Approximate KV cache size (rough estimate: 2 * hidden_size * num_layers * seq_len * 2 bytes)
        # For Llama-3.1-8B: hidden_size=4096, num_layers=32
        kv_cache_mb = (cumulative * 4096 * 32 * 2 * 2) / (1024 * 1024)
        print(f"{turn['turn']:<6} {turn['total_tokens']:<15} {cumulative:<15} {kv_cache_mb:<25.1f} MB")
    
    # Summary statistics
    print(f"\n{'='*80}")
    print("Summary Statistics:")
    print(f"{'='*80}")
    print(f"Total turns: {len(turn_results)}")
    print(f"Total tokens: {total_user_tokens + total_assistant_tokens}")
    print(f"Average tokens per turn: {avg_total:.1f}")
    print(f"Average user tokens: {avg_user:.1f}")
    print(f"Average assistant tokens: {avg_assistant:.1f}")
    print(f"Max turn tokens: {max(t['total_tokens'] for t in turn_results)}")
    print(f"Min turn tokens: {min(t['total_tokens'] for t in turn_results)}")

if __name__ == '__main__':
    main()