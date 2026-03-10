import torch
import torch.nn as nn
import pandas as pd

from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.tokenizer import tokenize_expression
from src.preprocessing.vocabulary import build_vocabulary
from src.preprocessing.vocabulary import encode_tokens
from src.preprocessing.encoder import encode_dataset

from src.models.lstm_seq2seq import Encoder, Decoder, Seq2Seq

from src.utils.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_dataset():

    df = pd.read_csv("data/raw/dataset.csv")

    functions = df["function"].tolist()
    expansions = df["taylor"].tolist()

    return functions, expansions

def tokenize_data(functions, expansions):

    tokenized_inputs = []
    tokenized_outputs = []

    for f, t in zip(functions, expansions):

        tokenized_inputs.append(tokenize_expression(f))
        tokenized_outputs.append(tokenize_expression(t))

    return tokenized_inputs, tokenized_outputs

def build_vocab(tokenized_inputs, tokenized_outputs):

    combined = tokenized_inputs + tokenized_outputs

    vocab = build_vocabulary(combined)

    return vocab

def encode_data(tokenized_inputs, tokenized_outputs, vocab):

    encoded_inputs = encode_dataset(
        tokenized_inputs,
        vocab,
        MAX_SEQUENCE_LENGTH
    )

    encoded_outputs = encode_dataset(
        tokenized_outputs,
        vocab,
        MAX_SEQUENCE_LENGTH
    )

    return encoded_inputs, encoded_outputs

def create_dataloader(inputs, outputs):

    dataset = TensorDataset(inputs, outputs)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    return loader

def initialize_model(vocab_size):

    encoder = Encoder(vocab_size)

    decoder = Decoder(vocab_size)

    model = Seq2Seq(encoder, decoder, device).to(device)

    return model

def initialize_training(model):

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    return criterion, optimizer

