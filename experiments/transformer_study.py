import optuna
import torch
import torch.nn as nn
import os
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset

from src.training.train_lstm import load_dataset, tokenize_data
from src.preprocessing.vocabulary import build_vocabulary
from src.preprocessing.encoder import encode_dataset
from src.utils.config import MAX_SEQUENCE_LENGTH, BATCH_SIZE
from src.models.transformer_seq2seq import TransformerSeq2Seq

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def objective(trial):
    if device.type != 'cuda':
        raise RuntimeError("GPU not detected!")

    d_model = trial.suggest_categorical("d_model", [128, 256, 512])
    nhead = trial.suggest_categorical("nhead", [4, 8])
    num_layers = trial.suggest_int("num_layers", 4, 6)
    lr = trial.suggest_float("lr", 1e-4, 8e-4, log=True)
    
    functions, expansions = load_dataset()
    all_in, all_out = tokenize_data(functions, expansions)
    vocab = build_vocabulary(all_in + all_out)

    train_f, val_f, train_t, val_t = train_test_split(
        functions, expansions, test_size=0.2, random_state=42
    )

    t_in, t_out = tokenize_data(train_f, train_t)
    enc_train_in = encode_dataset(t_in, vocab, MAX_SEQUENCE_LENGTH)
    enc_train_out = encode_dataset(t_out, vocab, MAX_SEQUENCE_LENGTH)
    train_loader = DataLoader(TensorDataset(enc_train_in, enc_train_out), batch_size=BATCH_SIZE, shuffle=True)

    v_in, v_out = tokenize_data(val_f, val_t)
    enc_val_in = encode_dataset(v_in, vocab, MAX_SEQUENCE_LENGTH)
    enc_val_out = encode_dataset(v_out, vocab, MAX_SEQUENCE_LENGTH)
    val_loader = DataLoader(TensorDataset(enc_val_in, enc_val_out), batch_size=BATCH_SIZE)

    model = TransformerSeq2Seq(
        vocab_size=len(vocab),
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_layers,
        num_decoder_layers=num_layers
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=0)

    for epoch in range(5):
        model.train()
        for src, trg in train_loader:
            src, trg = src.to(device), trg.to(device)
            optimizer.zero_grad()
            output = model(src, trg[:, :-1]) 
            loss = criterion(output.reshape(-1, len(vocab)), trg[:, 1:].reshape(-1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0
        with torch.no_grad():
            for src, trg in val_loader:
                src, trg = src.to(device), trg.to(device)
                output = model(src, trg[:, :-1])
                loss = criterion(output.reshape(-1, len(vocab)), trg[:, 1:].reshape(-1))
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        
        trial.report(avg_val_loss, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    torch.save(model.state_dict(), f"models/temp_trial_{trial.number}.pth")
    return avg_val_loss

if __name__ == "__main__":
    print(f"Starting expanded study on: {torch.cuda.get_device_name(0)}")
    
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=50)

    print("\nOptimization complete")
    print("Best trial params:", study.best_params)
    
    best_model_path = f"models/temp_trial_{study.best_trial.number}.pth"
    if os.path.exists(best_model_path):
        os.replace(best_model_path, "models/transformer_optuna_best.pth")
        print("Best model weights saved to models/transformer_optuna_best.pth")
    
    for f in os.listdir("models"):
        if f.startswith("temp_trial_"):
            os.remove(os.path.join("models", f))