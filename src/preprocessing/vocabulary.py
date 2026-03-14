import re
from collections import Counter

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3
}

def smart_tokenize(text):
    # Regex pattern breakdown:
    # [A-Z]+_-?\d+      : Matches consolidated tokens (e.g., SIN_1, COS_-2, EXP_5)
    # [a-z]+            : Matches variables (e.g., 'x')
    # [\d]              : Matches single digits
    # [\+\-\*\/\^\(\),] : Matches mathematical operators
    pattern = r'[A-Z]+_-?\d+|[a-z]+|[\d]|[\+\-\*\/\^\(\),]'
    return re.findall(pattern, str(text))

def build_vocabulary(df):
    counter = Counter()

    for col in ['function', 'taylor']:
        for text in df[col]:
            tokens = smart_tokenize(text)
            counter.update(tokens)

    vocab = dict(SPECIAL_TOKENS)
    index = len(SPECIAL_TOKENS)

    for token in sorted(counter.keys()):
        if token not in vocab:
            vocab[token] = index
            index += 1

    return vocab

def encode_tokens(tokens, vocab):
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

if __name__ == "__main__":
    import pandas as pd
    import json
    import os

    DATA_PATH = "data/processed/dataset_ready_100k.csv"
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        vocab = build_vocabulary(df)
        
        print(f"Vocabulary built. Size: {len(vocab)}")
        
        os.makedirs("models", exist_ok=True)
        with open("models/vocab.json", "w") as f:
            json.dump(vocab, f, indent=4)
        
        sample_text = df['taylor'].iloc[0]
        tokens = smart_tokenize(sample_text)
        encoded = encode_tokens(tokens, vocab)
        
        print(f"\nExample Text: {sample_text}")
        print(f"Tokens: {tokens}")
        print(f"Encoded: {encoded}")
    else:
        print(f"File not found: {DATA_PATH}. Please run the preparation script first.")