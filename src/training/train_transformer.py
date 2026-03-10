import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.tokenizer import tokenize_expression
from src.preprocessing.vocabulary import build_vocabulary
from src.preprocessing.encoder import encode_dataset

from src.training.train_lstm import load_dataset, tokenize_data

from src.models.transformer_seq2seq import TransformerSeq2Seq

from src.utils.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH
)
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, TensorDataset

from src.preprocessing.tokenizer import tokenize_expression
from src.preprocessing.vocabulary import build_vocabulary
from src.preprocessing.encoder import encode_dataset

from src.training.train_lstm import load_dataset, tokenize_data

from src.models.transformer_seq2seq import TransformerSeq2Seq

from src.utils.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MAX_SEQUENCE_LENGTH
)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("Using device:", device)

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

    model = TransformerSeq2Seq(vocab_size).to(device)

    return model

def initialize_training(model):

    criterion = nn.CrossEntropyLoss(ignore_index=0)

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    return criterion, optimizer

def train_model(model, dataloader, criterion, optimizer):

    model.train()

    total_loss = 0

    for src, trg in dataloader:

        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg[:, :-1])

        vocab_size = output.shape[-1]

        output = output.reshape(-1, vocab_size)

        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

import os

def save_model(model):

    os.makedirs("models", exist_ok=True)

    torch.save(model.state_dict(), "models/transformer_model.pth")

    print("Model saved to models/transformer_model.pth")

def main():

    functions, expansions = load_dataset()

    tokenized_inputs, tokenized_outputs = tokenize_data(functions, expansions)

    vocab = build_vocabulary(tokenized_inputs + tokenized_outputs)

    encoded_inputs, encoded_outputs = encode_data(
        tokenized_inputs,
        tokenized_outputs,
        vocab
    )

    dataloader = create_dataloader(encoded_inputs, encoded_outputs)

    model = initialize_model(len(vocab))

    criterion, optimizer = initialize_training(model)

    for epoch in range(EPOCHS):

        loss = train_model(model, dataloader, criterion, optimizer)

        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")

    save_model(model)

if __name__ == "__main__":
    main()

