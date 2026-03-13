import torch
import sys
import json
import pandas as pd
from tqdm import tqdm
from sympy import sympify, simplify
from sklearn.model_selection import train_test_split

from src.evaluation.predict import predict_sequence, decode_tokens, load_model

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def compute_metrics(predictions, targets):
    exact_match_count = 0
    symbolic_match_count = 0

    for p, t in zip(predictions, targets):
        # 1. Standard String Exact Match (Fast)
        if p.strip() == t.strip():
            exact_match_count += 1
            symbolic_match_count += 1
            continue
            
        # 2. Symbolic Equivalence (Slow but robust)
        try:
            if simplify(sympify(p) - sympify(t)) == 0:
                symbolic_match_count += 1
        except:
            pass # Sympy couldn't parse the model's output

    total = len(predictions)
    return (exact_match_count / total) * 100, (symbolic_match_count / total) * 100

def evaluate():
    # Load identical test set split from training to prevent data leakage
    df = pd.read_csv("data/processed/dataset_ready_100k.csv")
    _, test_df = train_test_split(df, test_size=0.1, random_state=42)
    
    test_f = test_df['function'].tolist()
    test_t = test_df['taylor'].tolist()

    with open("models/vocab.json", "r") as f:
        vocab = json.load(f)

    model_type = sys.argv[1].lower() if len(sys.argv) > 1 else "lstm"

    if model_type == "lstm":
        model = load_model(vocab, device)
        print(f"✅ Evaluating LSTM model on {device}")
    elif model_type == "transformer":
        from src.models.transformer_seq2seq import TransformerSeq2Seq
        model = TransformerSeq2Seq(len(vocab)).to(device)
        model.load_state_dict(torch.load("models/transformer_model.pth", weights_only=True))
        model.eval()
        print(f"✅ Evaluating Transformer model on {device}")
    
    predictions = []
    targets = []
    
    # Testing on 500 samples
    samples_to_test = 500
    print(f"Running inference on {samples_to_test} test cases...")

    for f, t in tqdm(zip(test_f[:samples_to_test], test_t[:samples_to_test]), total=samples_to_test, ncols=80):
        if model_type == "lstm":
            predicted_ids = predict_sequence(model, f, vocab, device)
        else:
            from src.evaluation.predict import predict_sequence_transformer
            predicted_ids = predict_sequence_transformer(model, f, vocab, device)

        prediction = decode_tokens(predicted_ids, vocab)
        predictions.append(prediction)
        targets.append(t)

    exact_acc, symbolic_acc = compute_metrics(predictions, targets)
    
    print("\n" + "="*40)
    print(f"📊 EVALUATION RESULTS FOR {model_type.upper()}")
    print("="*40)
    print(f"String Exact Match:   {exact_acc:.2f}%")
    print(f"Symbolic Equivalence: {symbolic_acc:.2f}%")
    print("="*40)
    
    # Print a few examples for visual inspection
    print("\n🔍 SAMPLE PREDICTIONS:")
    for i in range(3):
        print(f"F(x)  : {test_f[i]}")
        print(f"Target: {test_t[i]}")
        print(f"Pred  : {predictions[i]}")
        print("-" * 40)

if __name__ == "__main__":
    evaluate()