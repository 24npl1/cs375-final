from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import random
import torch

def hash_tokens(tokens):
    hash_val = 0
    for token in tokens:
        hash_val = ((hash_val + token) * 31) # Use a prime number for better distribution
    return int(hash_val)

def hard_redlist(tensor):
    sequence = tensor[0][0][:-1]
    scores = tensor[1][0][0]

    hash = hash_tokens(sequence)
    random.seed(hash)

    for i in range(len(scores)):
        if random.random() < 0.5:
            scores[i] = -float('inf')

    new_tok = torch.argmax(scores)
    tensor[0][0][-1] = new_tok
    return tensor[0][0]

def hard_watermark(input_str, output_len = 20):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    for _ in range(output_len):
        inputs = tokenizer(input_str, return_tensors="pt")
        generation_output = model.generate(**inputs, max_new_tokens = 1, return_dict_in_generate=True, output_scores=True, early_stopping = True)
        sequence = hard_redlist(generation_output)
        input_str = tokenizer.decode(sequence)

    return input_str

def soft_redlist(tensor, sigma, gamma):
    sequence = tensor[0][0][:-1]
    scores = tensor[1][0][0]

    hash = hash_tokens(sequence)
    random.seed(hash)
    print(hash)

    green_red = {}
    for i in range(len(scores)):
        if random.random() < gamma:
            green_red[(scores[i], i)] = "G"
        else:
            green_red[(scores[i], i)] = "R"

    green_list = [(l + sigma, i) for (l, i), color in green_red.items() if color == 'G']
    red_list = [(r, i) for (r, i), color in green_red.items() if color == 'R']

    greens, green_idx = zip(*green_list)
    reds, red_idx = zip(*red_list)

    greens = np.array(greens)
    reds = np.array(reds)

    green_shift = greens - np.min(greens)
    red_shift = reds - np.min(reds)

    exp_g = sum(np.exp(green_shift))
    exp_r = sum(np.exp(red_shift))

    prob_g = [np.exp(g) / (exp_r + exp_g + 1e-5) for g in greens]
    prob_r = [np.exp(r) / (exp_r + exp_g + 1e-5) for r in reds]

    probs = np.concatenate([prob_g, prob_r])
    word_probs = np.nan_to_num(probs, nan = 0.0)
    words = np.concatenate([green_idx, red_idx])

    norms = word_probs / np.sum(word_probs) 
    word_norms = np.nan_to_num(norms, nan = 0.0)
    print(sum(word_norms))

    next_tok = np.random.choice(words, p = word_norms)
    tensor[0][0][-1] = next_tok

    return tensor[0][0]

def soft_watermark(input_str, output_len = 20, gamma = 0.5, sigma = 0.7):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    for _ in range(output_len):
        inputs = tokenizer(input_str, return_tensors="pt")
        generation_output = model.generate(**inputs, max_new_tokens = 1, return_dict_in_generate=True, output_scores=True, early_stopping = True)
        sequence = soft_redlist(generation_output, gamma, sigma)
        input_str = tokenizer.decode(sequence)
        print(input_str)

    return input_str

def main():
    input = "My name is Elias and as a collge-aged male "
    print(soft_watermark(input))

if __name__ == "__main__":
    main()
