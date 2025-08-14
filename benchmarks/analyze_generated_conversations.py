#!/usr/bin/env python3

import sys
import json
from transformers import AutoTokenizer
from bench_dataset import GenConvArgs, generate_conversations, parse_input_json_file

def analyze_generated_conversations(config_file, seed=0):
    """Generate and analyze conversations from the config file."""
    
    # Load the configuration
    with open(config_file, 'r') as f:
        config = json.load(f)
    
    # Create GenConvArgs from the config using the proper parser
    gen_conv_args = parse_input_json_file(config)
    
    # Initialize tokenizer (using a simple one for testing)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    
    # Generate conversations
    print(f"Generating conversations with seed {seed}...")
    conversations = generate_conversations(gen_conv_args, tokenizer, seed)
    
    print(f"=== Analysis of Generated Conversations ===")
    print(f"Number of conversations: {len(conversations)}")
    
    # Analyze each conversation
    total_user_messages = 0
    total_assistant_messages = 0
    conversation_details = []
    
    for conv_id, messages in conversations.items():
        user_count = sum(1 for msg in messages if msg['role'] == 'user')
        assistant_count = sum(1 for msg in messages if msg['role'] == 'assistant')
        
        conversation_details.append({
            'conv_id': conv_id,
            'total_messages': len(messages),
            'user_messages': user_count,
            'assistant_messages': assistant_count
        })
        
        total_user_messages += user_count
        total_assistant_messages += assistant_count
    
    print(f"Total user messages (requests): {total_user_messages}")
    print(f"Total assistant messages: {total_assistant_messages}")
    print(f"Average user messages per conversation: {total_user_messages / len(conversations):.2f}")
    
    # Show details for each conversation
    print("\nConversation details:")
    for details in conversation_details:
        print(f"  {details['conv_id']}: {details['user_messages']} user, {details['assistant_messages']} assistant ({details['total_messages']} total)")
    
    # Save the generated conversations for inspection
    output_file = config_file.replace('.json', '_generated.json')
    with open(output_file, 'w') as f:
        json.dump(conversations, f, indent=2)
    print(f"\nGenerated conversations saved to: {output_file}")
    
    return conversations, total_user_messages

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python analyze_generated_conversations.py <config_file> [seed]")
        sys.exit(1)
    
    config_file = sys.argv[1]
    seed = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    
    conversations, total_requests = analyze_generated_conversations(config_file, seed)
    
    print(f"\n=== Summary ===")
    print(f"With seed {seed}, the benchmark should process {total_requests} user requests")
    print(f"Both max-active-conversations=24 and max-active-conversations=12 should produce the same count: {total_requests}")