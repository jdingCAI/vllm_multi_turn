#!/usr/bin/env python3

import json
import sys
from collections import defaultdict

def analyze_benchmark_logic():
    """Debug the benchmark logic to understand why counts differ."""
    
    # Load the generated conversations
    with open('generate_multi_turn_generated.json', 'r') as f:
        conversations = json.load(f)
    
    print("=== Debugging Benchmark Logic ===")
    print(f"Total conversations: {len(conversations)}")
    
    # Simulate different max_active_conversations scenarios
    for max_active in [24, 12]:
        print(f"\n--- Simulation with max_active_conversations={max_active} ---")
        
        # Each conversation should be processed fully
        total_requests = 0
        conversations_completed = 0
        
        for conv_id, messages in conversations.items():
            # Count user messages (these become requests to the server)
            user_messages = [msg for msg in messages if msg['role'] == 'user']
            conversation_requests = len(user_messages)
            
            print(f"  {conv_id}: {conversation_requests} user requests")
            
            total_requests += conversation_requests
            conversations_completed += 1
        
        print(f"  Expected total requests: {total_requests}")
        print(f"  Expected completed conversations: {conversations_completed}")
        
        # This should be the same for both max_active_conversations values
        assert total_requests == 185, f"Expected 185 requests, got {total_requests}"

if __name__ == "__main__":
    analyze_benchmark_logic()