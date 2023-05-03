from transformers import GPT2Tokenizer, GPT2LMHeadModel
import ipdb
import numpy as np
import random
import torch

def hash_tokens(tokens):
    hash_val = 0
    for token in tokens:
        hash_val = (hash_val + token) * 31 # Use a prime number for better distribution
    return int(hash_val)

def hard_redlist(tensor):
    sequence = tensor[0][0][:-1]
    scores = tensor[1][0]

    hash = hash_tokens(sequence)
    random.seed(hash)

    for i in range(len(scores[0])):
        if random.random() < 0.5:
            scores[0][i] = -float('inf')

    new_tok = torch.argmax(scores)
    tensor[0][0][-1] = new_tok
    return tensor[0][0]

def hard_watermark(input_str, output_len = 20):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    for _ in range(output_len):
        inputs = tokenizer(input_str, return_tensors="pt")
        generation_output = model.generate(**inputs, max_new_tokens = 1, return_dict_in_generate=True, output_scores=True, early_stopping = True)
        redlist = hard_redlist(generation_output)
        input_str = tokenizer.decode(redlist)

    return input_str

def main():
    input = "My name is Elias and as a collge-aged male "
    print(hard_watermark(input))

if __name__ == "__main__":
    main()
