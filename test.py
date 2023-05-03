from transformers import GPT2Tokenizer, GPT2LMHeadModel





def main():
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    model = GPT2LMHeadModel.from_pretrained("gpt2")

    inputs = tokenizer("Hello, my dog is cute and ", return_tensors="pt")
    generation_output = model.generate(**inputs, max_new_tokens = 1, return_dict_in_generate=True, output_scores=True)
    print(generation_output[1][0].size())

if __name__ == "__main__":
    main()
