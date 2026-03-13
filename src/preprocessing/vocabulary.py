import re
from collections import Counter

SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3
}

def smart_tokenize(text):
    """
    Tokenizes math strings while keeping SIN_1, COS_5, etc., as single tokens.
    """
    # Regex breakdown:
    # 1. [A-Z]+_-?\d+  -> Matches SIN_1, COS_-2, EXP_5
    # 2. [a-z]+        -> Matches variables like 'x' or 'exp' (if not consolidated)
    # 3. [\d]          -> Matches single digits
    # 4. [\+\-\*\/\^\(\),] -> Matches operators including our new ^
    pattern = r'[A-Z]+_-?\d+|[a-z]+|[\d]|[\+\-\*\/\^\(\),]'
    return re.findall(pattern, str(text))

def build_vocabulary(df):
    """
    Build token → integer mapping from the entire dataframe.
    """
    counter = Counter()

    # Tokenize both columns to get all possible tokens
    for col in ['function', 'taylor']:
        for text in df[col]:
            tokens = smart_tokenize(text)
            counter.update(tokens)

    vocab = dict(SPECIAL_TOKENS)
    index = len(SPECIAL_TOKENS)

    # Sort tokens so the vocab is consistent every time you run it
    for token in sorted(counter.keys()):
        if token not in vocab:
            vocab[token] = index
            index += 1

    return vocab

def encode_tokens(tokens, vocab):
    """
    Convert tokens into integer ids with UNK handling.
    """
    return [vocab.get(token, vocab["<UNK>"]) for token in tokens]

if __name__ == "__main__":
    import pandas as pd
    import json
    import os

    # 1. Path to your NEW ready dataset
    DATA_PATH = "data/processed/dataset_ready_100k.csv"
    
    if os.path.exists(DATA_PATH):
        df = pd.read_csv(DATA_PATH)
        vocab = build_vocabulary(df)
        
        print(f"✅ Vocabulary Built! Size: {len(vocab)}")
        
        # 2. Save it for the training script to use
        os.makedirs("models", exist_ok=True)
        with open("models/vocab.json", "w") as f:
            json.dump(vocab, f, indent=4)
        
        # 3. Test on one of your new complex strings
        sample_text = df['taylor'].iloc[0]
        tokens = smart_tokenize(sample_text)
        encoded = encode_tokens(tokens, vocab)
        
        print(f"\nExample Text: {sample_text}")
        print(f"Tokens: {tokens}")
        print(f"Encoded: {encoded}")
    else:
        print(f"❌ File not found: {DATA_PATH}. Run your preparation script first!")