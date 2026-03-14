import torch

from src.preprocessing.vocabulary import encode_tokens

def add_special_tokens(tokens):
    return ["<SOS>"] + tokens + ["<EOS>"]

def pad_sequence(sequence, max_length, pad_value=0):
    if len(sequence) > max_length:
        sequence = sequence[:max_length]

    padded = sequence + [pad_value] * (max_length - len(sequence))
    return padded

def encode_dataset(tokenized_data, vocab, max_length):
    encoded_sequences = []

    for tokens in tokenized_data:
        tokens = add_special_tokens(tokens)
        encoded = encode_tokens(tokens, vocab)
        padded = pad_sequence(encoded, max_length)
        encoded_sequences.append(padded)

    return torch.tensor(encoded_sequences)

if __name__ == "__main__":
    sample_tokens = [
        ['sin', '(', 'x', ')'],
        ['x', '*', 'sin', '(', 'x', ')']
    ]

    vocab = {
        '<PAD>': 0, '<SOS>': 1, '<EOS>': 2, '<UNK>': 3,
        'sin': 4, '(': 5, 'x': 6, ')': 7, '*': 8
    }

    max_length = 10
    encoded = encode_dataset(sample_tokens, vocab, max_length)
    print(encoded)