#!/usr/bin/env python3
"""
Check how tokens are counted with chat template applied.
"""

import json
from pathlib import Path
from transformers import AutoTokenizer

def analyze_chat_template_tokens():
    # Load the workload dataset
    workload_path = Path('benchmark_results/workload_dataset.json')
    
    with open(workload_path, 'r') as f:
        data = json.load(f)
    
    # Get first conversation
    conv_id_0 = None
    for conv in data:
        if conv['id'] == 'CONV_ID_0':
            conv_id_0 = conv
            break
    
    if not conv_id_0:
        print("CONV_ID_0 not found")
        return
    
    # Initialize tokenizer
    model_name = 'meta-llama/Llama-3.1-8B-Instruct'
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    print(f"Analyzing token counts for CONV_ID_0")
    print("=" * 80)
    
    # Get first turn (first user message only)
    first_user_message = conv_id_0['messages'][0]
    
    # Method 1: Raw content tokens (what we calculated before)
    raw_tokens = tokenizer.encode(first_user_message['content'], add_special_tokens=False)
    print(f"Raw content tokens: {len(raw_tokens)}")
    
    # Method 2: With special tokens
    with_special = tokenizer.encode(first_user_message['content'], add_special_tokens=True)
    print(f"With special tokens: {len(with_special)}")
    
    # Method 3: Using chat template (what vLLM actually uses)
    # This is what the model sees when processing the chat request
    messages = [first_user_message]  # Just the first user message
    
    # Apply chat template
    chat_input = tokenizer.apply_chat_template(
        messages, 
        tokenize=False,  # Get the string first
        add_generation_prompt=True  # Add the assistant prompt
    )
    print(f"\nChat template applied:")
    print("-" * 40)
    print(chat_input[:500] + "..." if len(chat_input) > 500 else chat_input)
    print("-" * 40)
    
    # Tokenize the chat template
    chat_tokens = tokenizer.encode(chat_input, add_special_tokens=False)
    print(f"\nChat template tokens: {len(chat_tokens)}")
    
    # Show the breakdown
    print(f"\nToken count breakdown:")
    print(f"  Raw content: {len(raw_tokens)} tokens")
    print(f"  + Special tokens: {len(with_special) - len(raw_tokens)} tokens")
    print(f"  = With special tokens: {len(with_special)} tokens")
    print(f"  Chat template total: {len(chat_tokens)} tokens")
    print(f"  Overhead from chat template: {len(chat_tokens) - len(raw_tokens)} tokens")
    
    # Check what the chat template adds
    print(f"\nChat template components:")
    
    # Get the system prompt if any
    system_prompt = tokenizer.apply_chat_template(
        [], 
        tokenize=False,
        add_generation_prompt=True
    )
    system_tokens = tokenizer.encode(system_prompt, add_special_tokens=False)
    print(f"  Empty chat (system/format only): {len(system_tokens)} tokens")
    
    # Calculate actual content contribution
    content_contribution = len(chat_tokens) - len(system_tokens)
    print(f"  Content contribution: {content_contribution} tokens")
    print(f"  Format/template overhead: {len(chat_tokens) - len(raw_tokens)} tokens")
    
    print(f"\n{'='*80}")
    print(f"Summary:")
    print(f"  Your calculation (raw content): 775 tokens")
    print(f"  LMCache reports: 809 tokens")
    print(f"  Chat template result: {len(chat_tokens)} tokens")
    print(f"  Difference explained: {809 - 775} = 34 extra tokens from chat formatting")

if __name__ == '__main__':
    analyze_chat_template_tokens()