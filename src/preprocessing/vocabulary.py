from collections import Counter


SPECIAL_TOKENS = {
    "<PAD>": 0,
    "<SOS>": 1,
    "<EOS>": 2,
    "<UNK>": 3
}


def build_vocabulary(tokenized_data):
    """
    Build token → integer mapping.
    """

    counter = Counter()

    for tokens in tokenized_data:
        counter.update(tokens)

    vocab = dict(SPECIAL_TOKENS)

    index = len(SPECIAL_TOKENS)

    for token in counter:

        if token not in vocab:
            vocab[token] = index
            index += 1

    return vocab


def encode_tokens(tokens, vocab):
    """
    Convert tokens into integer ids.
    """

    encoded = []

    for token in tokens:

        if token in vocab:
            encoded.append(vocab[token])
        else:
            encoded.append(vocab["<UNK>"])

    return encoded


if __name__ == "__main__":

    sample_tokens = [
        ['sin', '(', 'x', ')'],
        ['x', '*', 'sin', '(', 'x', ')'],
        ['exp', '(', 'x', ')', '+', '3'],
        ['x', '**', '2', '-', 'x']
    ]

    vocab = build_vocabulary(sample_tokens)

    print("Vocabulary:")
    print(vocab)

    print("\nEncoded example:")

    encoded = encode_tokens(sample_tokens[0], vocab)

    print(sample_tokens[0])
    print(encoded)