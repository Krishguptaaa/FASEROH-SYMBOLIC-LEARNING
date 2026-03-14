import torch
import torch.nn as nn
import pandas as pd
import os
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

from src.preprocessing.vocabulary import smart_tokenize
from src.preprocessing.encoder import encode_dataset
from src.models.transformer_seq2seq import TransformerSeq2Seq
from src.utils.config import MAX_SEQUENCE_LENGTH

BATCH_SIZE = 128 
EPOCHS = 30      
LR = 0.0001
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_data_and_vocab():
    print("Loading dataset and vocab...")
    df = pd.read_csv("data/processed/dataset_ready_100k.csv")
    functions = df["function"].tolist()
    expansions = df["taylor"].tolist()
    
    with open("models/vocab.json", "r") as f:
        vocab = json.load(f)
        
    return functions, expansions, vocab

def prepare_dataloaders(functions, expansions, vocab):
    train_f, test_f, train_t, test_t = train_test_split(
        functions, expansions, test_size=0.2, random_state=42
    )

    print("Tokenizing and encoding training data...")
    tokenized_inputs = [smart_tokenize(f) for f in train_f]
    tokenized_outputs = [smart_tokenize(t) for t in train_t]

    encoded_inputs = encode_dataset(tokenized_inputs, vocab, MAX_SEQUENCE_LENGTH)
    encoded_outputs = encode_dataset(tokenized_outputs, vocab, MAX_SEQUENCE_LENGTH)

    if not isinstance(encoded_inputs, torch.Tensor):
        encoded_inputs = torch.tensor(encoded_inputs, dtype=torch.long)
        encoded_outputs = torch.tensor(encoded_outputs, dtype=torch.long)

    dataset = TensorDataset(encoded_inputs, encoded_outputs)
    return DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

def train():
    functions, expansions, vocab = load_data_and_vocab()
    train_loader = prepare_dataloaders(functions, expansions, vocab)
    
    vocab_size = len(vocab)
    pad_idx = vocab.get("<PAD>", 0)

    model = TransformerSeq2Seq(
        vocab_size=vocab_size,
        d_model=256,
        nhead=8,
        num_encoder_layers=3,
        num_decoder_layers=3,
        dim_feedforward=1024,
        pad_idx=pad_idx
    ).to(DEVICE)

    criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    print(f"\nStarting Transformer training on {DEVICE}...")
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", ncols=80)
        
        for src, trg in pbar:
            src, trg = src.to(DEVICE), trg.to(DEVICE)
            
            tgt_input = trg[:, :-1]
            tgt_expected = trg[:, 1:]
            
            optimizer.zero_grad()
            
            output = model(src, tgt_input)
            
            output = output.reshape(-1, output.shape[-1])
            tgt_expected = tgt_expected.reshape(-1)
            
            loss = criterion(output, tgt_expected)
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            total_loss += loss.item()
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        avg_loss = total_loss / len(train_loader)
        print(f"Epoch {epoch+1} Summary | Avg Loss: {avg_loss:.4f}")

        os.makedirs("models", exist_ok=True)
        torch.save(model.state_dict(), "models/transformer_model.pth")
        print(f"Checkpoint saved for Epoch {epoch+1}")
        
    print("Training complete.")

if __name__ == "__main__":
    train()
