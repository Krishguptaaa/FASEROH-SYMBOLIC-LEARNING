import torch
import torch.nn as nn
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Updated to use smart_tokenize to protect math tokens like SIN_1
from src.preprocessing.vocabulary import smart_tokenize
from src.preprocessing.encoder import encode_dataset

from src.models.lstm_seq2seq import Encoder, Decoder, Seq2Seq

from src.utils.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE_LSTM,
    MAX_SEQUENCE_LENGTH
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- OPTUNA WINNERS ---
EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 2
LR = 0.0001463962829299425

def load_dataset():
    df = pd.read_csv("data/processed/dataset_ready_100k.csv")
    functions = df["function"].tolist()
    expansions = df["taylor"].tolist()
    return functions, expansions

def tokenize_data(functions, expansions):
    tokenized_inputs = []
    tokenized_outputs = []
    for f, t in zip(functions, expansions):
        tokenized_inputs.append(smart_tokenize(f))
        tokenized_outputs.append(smart_tokenize(t))
    return tokenized_inputs, tokenized_outputs

def load_vocab():
    # Changed from build_vocab to load_vocab to prevent token ID shuffling
    with open("models/vocab.json", "r") as f:
        vocab = json.load(f)
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
    
    # Ensure tensors are returned for the DataLoader
    if not isinstance(encoded_inputs, torch.Tensor):
        encoded_inputs = torch.tensor(encoded_inputs, dtype=torch.long)
        encoded_outputs = torch.tensor(encoded_outputs, dtype=torch.long)
        
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
    # Added the Optuna dimensions required by your Encoder/Decoder classes
    encoder = Encoder(vocab_size, EMB_DIM, HID_DIM, N_LAYERS)
    decoder = Decoder(vocab_size, EMB_DIM, HID_DIM, N_LAYERS)
    model = Seq2Seq(encoder, decoder, device).to(device)
    return model

def initialize_training(model):
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LR
    )
    return criterion, optimizer

def train_model(model, dataloader, criterion, optimizer, tf_ratio, epoch, total_epochs):
    model.train()
    total_loss = 0
    
    # Added tqdm with ncols=80 to fix the visual waterfall glitch
    pbar = tqdm(dataloader, desc=f"Epoch {epoch}/{total_epochs}", ncols=80)

    for src, trg in pbar:
        src = src.to(device)
        trg = trg.to(device)

        optimizer.zero_grad()
        output = model(src, trg, teacher_forcing_ratio=tf_ratio)

        vocab_size = output.shape[-1]
        
        # Sliced [:, 1:] to align targets correctly
        output = output[:, 1:].reshape(-1, vocab_size)
        trg = trg[:, 1:].reshape(-1)

        loss = criterion(output, trg)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        pbar.set_postfix(loss=f"{loss.item():.4f}")

    return total_loss / len(dataloader)

def main():
    # 1. Load data
    functions, expansions = load_dataset()

    # 2. Split data for Training (80/20)
    train_f, test_f, train_t, test_t = train_test_split(
        functions, expansions, test_size=0.2, random_state=42
    )

    # 3. Load Universal Vocab from disk
    vocab = load_vocab()
    print(f"Loaded Vocabulary. Size: {len(vocab)}")

    # 4. Tokenize and Encode ONLY the training split
    tokenized_inputs, tokenized_outputs = tokenize_data(train_f, train_t)
    encoded_inputs, encoded_outputs = encode_data(
        tokenized_inputs, tokenized_outputs, vocab
    )

    # 5. Standard Training Pipeline
    dataloader = create_dataloader(encoded_inputs, encoded_outputs)
    model = initialize_model(len(vocab))
    criterion, optimizer = initialize_training(model)

    current_tf_ratio = 0.58
    print("Using device:", device)
    
    for epoch in range(EPOCHS):
        loss = train_model(model, dataloader, criterion, optimizer, current_tf_ratio, epoch+1, EPOCHS)
        print(f"✅ Epoch {epoch+1}/{EPOCHS} | Loss: {loss:.4f} | TF: {current_tf_ratio:.2f}")
        
        # Decay teacher forcing slightly each epoch
        current_tf_ratio = max(0.4, current_tf_ratio - 0.01)
    
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