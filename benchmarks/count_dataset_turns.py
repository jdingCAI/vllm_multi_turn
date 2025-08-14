#!/usr/bin/env python3
"""Count the exact number of turns in the generated dataset."""

import sys
import os
sys.path.append(os.path.dirname(__file__))

from bench_dataset import parse_input_json_file, generate_conversations
from transformers import AutoTokenizer

def count_dataset_turns(input_file: str, seed: int = 0):
    # Load configuration
    with open(input_file, 'r') as f:
        import json
        config = json.load(f)
    
    # Parse configuration
    gen_conv_args = parse_input_json_file(config)
    
    # Load tokenizer (needed for conversation generation)
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3.1-8B-Instruct")
    
    # Generate conversations
    conversations = generate_conversations(gen_conv_args, tokenizer, seed)
    
    # Count turns
    total_turns = 0
    total_conversations = len(conversations)
    
    print(f"Dataset Analysis:")
    print(f"Number of conversations: {total_conversations}")
    print(f"\nTurns per conversation:")
    
    turns_per_conv = []
    for conv_id, messages in conversations.items():
        num_turns = len(messages)
        turns_per_conv.append(num_turns)
        total_turns += num_turns
        print(f"  {conv_id}: {num_turns} turns")
    
    print(f"\nSummary:")
    print(f"Total turns/requests: {total_turns}")
    print(f"Average turns per conversation: {total_turns/total_conversations:.1f}")
    print(f"Min turns: {min(turns_per_conv)}")
    print(f"Max turns: {max(turns_per_conv)}")
    
    return total_turns

if __name__ == "__main__":
    seed = 42 if len(sys.argv) < 3 else int(sys.argv[2])
    input_file = sys.argv[1] if len(sys.argv) > 1 else "generate_multi_turn.json"
    total = count_dataset_turns(input_file, seed)
    print(f"\n🎯 Total requests/turns in dataset: {total}")