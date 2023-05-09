
import csv
import random
from detect_watermark import *
from watermarking import *
import pandas as pd
import matplotlib.pyplot as plt


def get_random_csv_entries(filename, col):
    # Read the CSV file and extract the column of interest
    with open(filename, 'r') as csvfile:
        reader = csv.reader(csvfile)
        column = [row[col] for row in reader]

    # Select 100 random entries from the column
    random_entries = random.sample(column, k=100)

    return random_entries

def histogram(csv_file_path, Type):
    # Read csv file into a pandas dataframe
    df = pd.read_csv(csv_file_path)

    # Filter the dataframe to keep only rows where the column value is "Hard Watermark"
    filtered_df = df[df.type == Type]
    probs = sorted(filtered_df["prob"])
    print(probs)

    # Create a histogram of the column with value "Hard Watermark"
    plt.hist(probs)
    plt.ylabel("Count")
    plt.xlabel("Probabilty that text is generated with watermark")
    plt.title(f"Retrieval Probabilites of {Type} text")
    plt.show()

def generate_csv():
    l = get_random_csv_entries("AGNEWS.csv", 2)
    res = []
    for example in l:
        temp = {}
        temp["text"] = example
        temp["type"] = "Non-Generated"
        temp["prob"] = fancy_detect_watermark(example, 5, 1729, gamma = 0.5)
        res.append(temp)
    
    l2 = get_random_csv_entries("AGNEWS.csv", 2)
    for e2 in l2:
        temp = {}
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        inputs = tokenizer(e2, return_tensors="pt")
        prompt_toks = inputs['input_ids'][0][:5]
        prompt = tokenizer.decode(prompt_toks)
        try:
            e = watermark(prompt)
            temp["text"] = e
            temp["type"] = "Hard Watermark"
            temp["prob"] = fancy_detect_watermark(e, 5, 1729, gamma = 0.5)
            res.append(temp)
        except:
            continue

    l3 = get_random_csv_entries("AGNEWS.csv", 2)
    for e3 in l3:
        temp = {}
        tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
        inputs = tokenizer(e3, return_tensors="pt")
        prompt_toks = inputs['input_ids'][0][:5]
        prompt = tokenizer.decode(prompt_toks)
        try:
            e = watermark(prompt, method = "soft")
            temp["text"] = e
            temp["type"] = "Soft Watermark"
            temp["prob"] = fancy_detect_watermark(e, 5, 1729, gamma = 0.5)
            res.append(temp)
        except:
            continue

    with open('results_1.csv', 'w', newline='') as csvfile:
        fieldnames = ["text", "type", "prob"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        # Write the header row
        writer.writeheader()

        # Write the data row
        for row in res:
            writer.writerow(row)

def main():
    histogram("results_1.csv", "Hard Watermark")


if __name__ == "__main__":
    main()