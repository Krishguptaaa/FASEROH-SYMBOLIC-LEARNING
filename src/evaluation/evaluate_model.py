import sys
import json
import difflib
import warnings
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from sympy import sympify, lambdify, symbols
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

from src.evaluation.predict import predict_sequence, decode_tokens, load_model

warnings.filterwarnings("ignore", category=RuntimeWarning)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_r2(prediction_str, target_str):
    x = symbols('x')
    try:
        pred_expr = sympify(prediction_str)
        target_expr = sympify(target_str)
        
        pred_func = lambdify(x, pred_expr, modules=['numpy', 'sympy'])
        target_func = lambdify(x, target_expr, modules=['numpy', 'sympy'])
        
        x_vals = np.linspace(-5, 5, 100)
        
        y_pred = pred_func(x_vals)
        y_target = target_func(x_vals)
        
        if isinstance(y_pred, (int, float, np.integer, np.floating)):
            y_pred = np.full_like(x_vals, y_pred, dtype=float)
        if isinstance(y_target, (int, float, np.integer, np.floating)):
            y_target = np.full_like(x_vals, y_target, dtype=float)
            
        y_pred = np.asarray(y_pred, dtype=float)
        y_target = np.asarray(y_target, dtype=float)
        
        mask = np.isfinite(y_pred) & np.isfinite(y_target)
        
        if np.sum(mask) < 2: 
            return -np.inf
            
        return r2_score(y_target[mask], y_pred[mask])
        
    except Exception:
        return -np.inf

def compute_metrics(predictions, targets):
    exact_match_count = 0
    symbolic_match_count = 0
    total_similarity = 0.0
    valid_r2_scores = []

    print("\nCalculating math metrics via SymPy...")
    for p, t in tqdm(zip(predictions, targets), total=len(predictions), ncols=80):
        
        similarity = difflib.SequenceMatcher(None, p.strip(), t.strip()).ratio()
        total_similarity += similarity

        if p.strip() == t.strip():
            exact_match_count += 1
            symbolic_match_count += 1
            valid_r2_scores.append(1.0) 
            continue
            
        try:
            # Using .expand() instead of .simplify() as SymPy can freeze on unconstrained transformer hallucinations
            if sympify(p).expand() == sympify(t).expand():
                symbolic_match_count += 1
                valid_r2_scores.append(1.0)
                continue
        except Exception:
            pass 
            
        r2 = compute_r2(p, t)
        if r2 != -np.inf:
            valid_r2_scores.append(r2)
        else:
            valid_r2_scores.append(0.0)

    total = len(predictions)
    avg_similarity = (total_similarity / total) * 100
    
    capped_r2s = [max(0.0, r) for r in valid_r2_scores]
    mean_r2 = np.mean(capped_r2s) if capped_r2s else 0.0
    
    return (exact_match_count / total) * 100, (symbolic_match_count / total) * 100, avg_similarity, mean_r2

def evaluate():
    df = pd.read_csv("data/processed/dataset_ready_100k.csv")
    _, test_df = train_test_split(df, test_size=0.1, random_state=42)
    
    test_f = test_df['function'].tolist()
    test_t = test_df['taylor'].tolist()

    with open("models/vocab.json", "r") as f:
        vocab = json.load(f)

    model_type = sys.argv[1].lower() if len(sys.argv) > 1 else "lstm"

    if model_type == "lstm":
        model = load_model(vocab, device)
        print(f"Evaluating LSTM model on {device}")
    elif model_type == "transformer":
        from src.models.transformer_seq2seq import TransformerSeq2Seq
        model = TransformerSeq2Seq(len(vocab)).to(device)
        model.load_state_dict(torch.load("models/transformer_model.pth", weights_only=True))
        model.eval()
        print(f"Evaluating Transformer model on {device}")
    
    predictions = []
    targets = []

    print(f"Running inference on {len(test_f)} test cases...")

    for f, t in tqdm(zip(test_f, test_t), total=len(test_f), ncols=80):
        if model_type == "lstm":
            predicted_ids = predict_sequence(model, f, vocab, device)
        else:
            from src.evaluation.predict import predict_sequence_transformer
            predicted_ids = predict_sequence_transformer(model, f, vocab, device)

        prediction = decode_tokens(predicted_ids, vocab)
        predictions.append(prediction)
        targets.append(t)

    exact_acc, symbolic_acc, similarity_score, r2 = compute_metrics(predictions, targets)
    
    print("\n" + "="*45)
    print(f"EVALUATION RESULTS FOR {model_type.upper()}")
    print("="*45)
    print(f"String Exact Match:   {exact_acc:.2f}%")
    print(f"Symbolic Equivalence: {symbolic_acc:.2f}%")
    print(f"Sequence Similarity:  {similarity_score:.2f}%")
    print(f"Mean R^2 Score:       {r2:.4f}")
    print("="*45)

if __name__ == "__main__":
    evaluate()