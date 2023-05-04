from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import random

def detect_watermark(input_str, prompt_len, seed, gamma):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    inputs = tokenizer(input_str, return_tensors="pt")
    tokens = inputs['input_ids'][0]

    for i in range(prompt_len - 1, len(tokens)):
        random.seed(seed + tokens[i])
    
        for i in range(tokens[i + 1] + 1):
            is_red = random.random() > gamma
        if is_red:
            return False

    return True

def main():
    detect_watermark("Hello, my name is hairy fish and I like water.")

if __name__ == "__main__":
    main()