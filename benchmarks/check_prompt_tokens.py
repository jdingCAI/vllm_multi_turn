#!/usr/bin/env python3
"""
Check how many tokens the prompt prefix adds.
"""

from transformers import AutoTokenizer

# Initialize tokenizer
model_name = 'meta-llama/Llama-3.1-8B-Instruct'
tokenizer = AutoTokenizer.from_pretrained(model_name)

# The prompt used in bench_dataset.py
prompt = "Please analyze the following text: "

# Count tokens
tokens = tokenizer.encode(prompt, add_special_tokens=False)

print(f"Prompt: '{prompt}'")
print(f"Number of tokens: {len(tokens)}")
print(f"Token IDs: {tokens}")
print(f"Decoded tokens: {[tokenizer.decode([t]) for t in tokens]}")

print(f"\nBreakdown:")
print(f"  Config: 512 (common prefix) + 256 (user prefix) = 768 tokens")
print(f"  + Prompt: {len(tokens)} tokens")
print(f"  = Total: {768 + len(tokens)} tokens")
print(f"  Actual: 775 tokens")
print(f"  Match: {768 + len(tokens) == 775}")