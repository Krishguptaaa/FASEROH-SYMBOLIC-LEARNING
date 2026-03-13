import optuna
import torch
import torch.nn as nn
import pandas as pd
import json
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from tqdm import tqdm

# Core imports
from src.models.lstm_seq2seq import Encoder, Decoder, Seq2Seq
from src.preprocessing.vocabulary import smart_tokenize # Use our new regex!
from src.utils.config import MAX_SEQUENCE_LENGTH

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# --- 1. GLOBAL DATA LOADING (Do this ONCE, not every trial) ---
print("Loading dataset and vocabulary...")
df = pd.read_csv("data/processed/dataset_ready_100k.csv").sample(15000, random_state=42) # Bumped to 15k for better signal

with open("models/vocab.json", "r") as f:
    global_vocab = json.load(f)

vocab_size = len(global_vocab)

# Tokenize using our new smart_tokenize that understands SIN_1 and ^
t_in = [smart_tokenize(text) for text in df['function']]
t_out = [smart_tokenize(text) for text in df['taylor']]

# Helper function to encode (since we aren't using the old vocabulary.py builder here)
def encode_sequence(tokens, vocab_dict, max_len):
    encoded = [vocab_dict.get(t, vocab_dict["<UNK>"]) for t in tokens]
    # Pad or truncate
    if len(encoded) < max_len:
        encoded += [vocab_dict["<PAD>"]] * (max_len - len(encoded))
    return torch.tensor(encoded[:max_len], dtype=torch.long)

# Encode all data globally
X_all = torch.stack([encode_sequence(seq, global_vocab, MAX_SEQUENCE_LENGTH) for seq in t_in])
Y_all = torch.stack([encode_sequence(seq, global_vocab, MAX_SEQUENCE_LENGTH) for seq in t_out])

# Split once
X_train, X_val, Y_train, Y_val = train_test_split(X_all, Y_all, test_size=0.15, random_state=42)

def objective(trial):
    # --- Search Space ---
    emb_dim = trial.suggest_categorical("emb_dim", [128, 256]) 
    hid_dim = trial.suggest_categorical("hid_dim", [512, 1024]) 
    n_layers = trial.suggest_int("n_layers", 2, 4)
    lr = trial.suggest_float("lr", 1e-4, 8e-4, log=True)
    tf_ratio = trial.suggest_float("tf_ratio", 0.5, 0.8) 
    
    # Force Batch Size 128 for safe VRAM limits during Optuna
    batch_size = 128 

    train_loader = DataLoader(TensorDataset(X_train, Y_train), batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(TensorDataset(X_val, Y_val), batch_size=batch_size)

    # --- Model Setup ---
    enc = Encoder(vocab_size, emb_dim, hid_dim, n_layers).to(device)
    dec = Decoder(vocab_size, emb_dim, hid_dim, n_layers).to(device)
    model = Seq2Seq(enc, dec, device).to(device)
    
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=global_vocab["<PAD>"])

    for epoch in range(4):
        model.train()
        
        # Wrap the loader in TQDM so you see the batches moving!
        progress_bar = tqdm(train_loader, desc=f"Trial {trial.number} | Epoch {epoch+1}/3", leave=False)
        
        for src, trg in progress_bar:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg, teacher_forcing_ratio=tf_ratio)
            
            output_dim = output.shape[-1]
            loss = criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()
            
            # Show the live loss on the right side of the progress bar
            progress_bar.set_postfix(loss=f"{loss.item():.4f}")

        # --- Validation Phase ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, trg in val_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg, teacher_forcing_ratio=0) 
                val_loss += criterion(output[:, 1:].reshape(-1, output_dim), trg[:, 1:].reshape(-1)).item()
        
        avg_loss = val_loss / len(val_loader)
        trial.report(avg_loss, epoch)
        
        # Pruning check (Stops bad trials early to save time)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return avg_loss

if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", pruner=optuna.pruners.MedianPruner())

    print("Optimization Engine Started for 20 Trials...")
    study.optimize(objective, n_trials=20) 
    
    print("\n" + "="*30)
    print("OPTIMAL PARAMETERS FOUND.")
    print(f"Trials Completed: {len([t for t in study.trials if t.state.name == 'COMPLETE'])}")
    print(f"Trials Pruned: {len([t for t in study.trials if t.state.name == 'PRUNED'])}")
    print("Best Params:", study.best_params)
    print("="*30)