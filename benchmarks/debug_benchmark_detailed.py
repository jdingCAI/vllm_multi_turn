#!/usr/bin/env python3

import json

# First, let's analyze the test_small.json conversations
with open('test_small.json', 'r') as f:
    config = json.load(f)

print("=== test_small.json configuration ===")
print(f"Number of conversations: {config['num_conversations']}")
print(f"Turns per conversation: {config['prompt_input']['num_turns']}")

# Load the generated conversations
try:
    with open('generate_multi_turn_generated.json', 'r') as f:
        conversations = json.load(f)
    
    print("\n=== Generated conversations ===")
    print(f"Total conversations: {len(conversations)}")
    
    # Analyze each conversation
    for conv_id, messages in conversations.items():
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        assistant_messages = [msg for msg in messages if msg['role'] == 'assistant']
        
        print(f"\n{conv_id}:")
        print(f"  Total messages: {len(messages)}")
        print(f"  User messages: {len(user_messages)}")
        print(f"  Assistant messages: {len(assistant_messages)}")
        print(f"  Expected turns to process: {len(user_messages)}")
        
except FileNotFoundError:
    print("\ngenerate_multi_turn_generated.json not found")

# Now let's trace through what should happen with max_active_conversations=4
print("\n=== Simulation with max_active_conversations=4 ===")
print("When max_active_conversations=4 and we have 4 conversations:")
print("- All 4 conversations should be loaded into active_convs")
print("- Each conversation should process all its user messages")
print("- Total requests should be sum of all user messages")

# Check test_small generated conversations
print("\n=== Generating test_small conversations ===")
import sys
sys.path.append('/home/jding/vllm_multi_turn/benchmarks')
from transformers import AutoTokenizer
from bench_dataset import parse_input_json_file, generate_conversations

tokenizer = AutoTokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token

gen_conv_args = parse_input_json_file(config)
test_conversations = generate_conversations(gen_conv_args, tokenizer, seed=0)

print(f"\nGenerated {len(test_conversations)} conversations:")
total_user_messages = 0
for conv_id, messages in test_conversations.items():
    user_messages = [msg for msg in messages if msg['role'] == 'user']
    print(f"  {conv_id}: {len(user_messages)} user messages")
    total_user_messages += len(user_messages)

print(f"\nTotal user messages (expected requests): {total_user_messages}")
print(f"All {len(test_conversations)} conversations should complete")