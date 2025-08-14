#!/usr/bin/env python3

import json
import sys
from collections import defaultdict

def analyze_conversations(filename):
    """Analyze the conversation structure to understand total requests."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"=== Analysis of {filename} ===")
    print(f"Number of conversations: {len(data)}")
    
    total_requests = 0
    conversation_details = []
    
    for conv_id, messages in data.items():
        # Count user messages (requests to the server)
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        conversation_details.append({
            'conv_id': conv_id,
            'total_messages': len(messages),
            'user_messages': len(user_messages),
            'assistant_messages': len(messages) - len(user_messages)
        })
        total_requests += len(user_messages)
    
    print(f"Total user requests across all conversations: {total_requests}")
    print(f"Average user requests per conversation: {total_requests / len(data):.2f}")
    
    # Show first few conversations for debugging
    print("\nFirst 5 conversations:")
    for i, details in enumerate(conversation_details[:5]):
        print(f"  Conv {details['conv_id']}: {details['user_messages']} user messages, {details['assistant_messages']} assistant messages")
    
    return total_requests, conversation_details

def simulate_benchmark_processing(filename, max_active_conversations, max_turns=None):
    """Simulate how the benchmark processes conversations."""
    with open(filename, 'r') as f:
        data = json.load(f)
    
    print(f"\n=== Simulating benchmark with max_active_conversations={max_active_conversations} ===")
    
    # Count what should be processed
    processed_requests = 0
    conversations_completed = 0
    
    for conv_id, messages in data.items():
        # Count user messages that would be sent as requests
        user_messages = [msg for msg in messages if msg['role'] == 'user']
        turns_to_process = len(user_messages)
        
        if max_turns is not None:
            turns_to_process = min(turns_to_process, max_turns)
        
        processed_requests += turns_to_process
        if turns_to_process > 0:
            conversations_completed += 1
    
    print(f"Expected processed requests: {processed_requests}")
    print(f"Expected completed conversations: {conversations_completed}")
    
    return processed_requests, conversations_completed

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python debug_conversation_processing.py <json_file>")
        sys.exit(1)
    
    filename = sys.argv[1]
    
    # Analyze the conversation structure
    total_requests, details = analyze_conversations(filename)
    
    # Simulate different benchmark configurations
    simulate_benchmark_processing(filename, 24)
    simulate_benchmark_processing(filename, 12)
    
    print(f"\nExpected result: Both configurations should process {total_requests} requests")