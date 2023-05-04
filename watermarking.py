from transformers import GPT2Tokenizer, GPT2LMHeadModel
import numpy as np
import random

def redlist(tensor, method, gamma, delta, seed):
    scores = tensor[1][0][0]
    last_tok = tensor[0][0][-1]
    last_tok = last_tok.item()
    random.seed(last_tok + seed)

    for i in range(len(scores)):
        if method == "hard":
            if gamma > random.random():
                scores[i] = -float('inf')
        else:
            if gamma < random.random():
                scores[i] = scores[i] + delta

    sum_score = sum(np.exp(np.array(scores)))

    word_probs = [np.exp(score.item()) / (sum_score + 1e-10) for score in scores]
    word_probs = np.nan_to_num(word_probs, nan = 0.0)

    words = [i for i in range(len(word_probs))]
    prob_factor = 1 / sum(word_probs)
    
    norms = [prob_factor * p for p in word_probs]
    tensor[0][0][-1] = np.random.choice(words, p = norms)

    return tensor[0][0]

def watermark(input_str, output_len = 20, method = "hard", gamma = 0.5, delta = 0.7, seed = 1729):
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")
    
    for _ in range(output_len):
        inputs = tokenizer(input_str, return_tensors="pt")
        generation_output = model.generate(**inputs, max_new_tokens = 1, return_dict_in_generate=True, output_scores=True)
        sequence = redlist(generation_output, method, gamma, delta, seed)
        input_str = tokenizer.decode(sequence)

    return input_str

def main():
    input = "My favorite president is Barack"
    print(watermark(input, method = "hard"))

if __name__ == "__main__":
    main()
