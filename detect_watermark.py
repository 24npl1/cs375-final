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
    hard = "Recently, scientists have discovered that obesity by itself may not be a contributing cause of heart problem. Infants' obesity generating lip"
    soft = "Recently, scientists have discovered that smoking in childhood doesn't plummet after birth by any means, but it has a profound impact on"
    none = "Recently, scientists have discovered that the brain's ability to process information is impaired when it is exposed to a high-energy source"
    detect_watermark("Hello, my name is hairy fish and I like water.")

if __name__ == "__main__":
    main()