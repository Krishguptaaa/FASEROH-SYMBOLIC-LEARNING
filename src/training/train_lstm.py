import torch
import torch.nn as nn
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
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

def train_model(model, dataloader, criterion, optimizer):

    model.train()

    total_loss = 0
    print("Using device:", device)

    for src, trg in dataloader:

        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()

        output = model(src, trg)

        vocab_size = output.shape[-1]

        output = output.reshape(-1, vocab_size)

        trg = trg.reshape(-1)

        loss = criterion(output, trg)

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(dataloader)

def main():
    # 1. Load data
    functions, expansions = load_dataset()

    # 2. Build Universal Vocab from 100% of the data
    all_in, all_out = tokenize_data(functions, expansions)
    vocab = build_vocabulary(all_in + all_out)
    
    # 3. Save Vocab to disk
    os.makedirs("models", exist_ok=True)
    with open("models/vocab.json", "w") as f:
        json.dump(vocab, f)
    
    print(f"Vocabulary saved. Size: {len(vocab)}")

    # 4. Split data for Training (80/20)
    train_f, test_f, train_t, test_t = train_test_split(
        functions, expansions, test_size=0.2, random_state=42
    )

    # 5. Tokenize and Encode ONLY the training split
    tokenized_inputs, tokenized_outputs = tokenize_data(train_f, train_t)
    encoded_inputs, encoded_outputs = encode_data(
        tokenized_inputs, tokenized_outputs, vocab
    )

    # 6. Standard Training Pipeline
    dataloader = create_dataloader(encoded_inputs, encoded_outputs)
    model = initialize_model(len(vocab))
    criterion, optimizer = initialize_training(model)

    for epoch in range(EPOCHS):
        loss = train_model(model, dataloader, criterion, optimizer)
        print(f"Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f}")
    
    save_model(model)



def save_model(model):
    """
    Saves the trained model state dictionary to the models directory.
    """
    os.makedirs("models", exist_ok=True)
    torch.save(model.state_dict(), "models/lstm_model.pth")
    print("Model saved to models/lstm_model.pth")

if __name__ == "__main__":
    main()
