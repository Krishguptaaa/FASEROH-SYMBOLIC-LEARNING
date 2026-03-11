import torch
import sys
import json
import pandas as pd
from sympy import sympify, simplify
from sklearn.model_selection import train_test_split

from src.preprocessing.tokenizer import tokenize_expression
from src.preprocessing.vocabulary import encode_tokens
from src.preprocessing.vocabulary import build_vocabulary
from src.preprocessing.encoder import encode_dataset

from src.training.train_lstm import load_dataset, tokenize_data
from src.evaluation.predict import predict_sequence, decode_tokens, load_model


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def compute_exact_match(predictions, targets):

    correct = 0

    for p, t in zip(predictions, targets):

        try:
            if simplify(sympify(p) - sympify(t)) == 0:
                correct += 1
        except:
            pass

    return correct / len(predictions)

def evaluate():
    functions, expansions = load_dataset()

    # Maintain the same split as training
    train_f, test_f, train_t, test_t = train_test_split(
        functions, expansions, test_size=0.2, random_state=42
    )

    # --- THE FIX: Load the saved training vocabulary ---
    with open("models/vocab.json", "r") as f:
        vocab = json.load(f)
    # ---------------------------------------------------

    model_type = "lstm"
    if len(sys.argv) > 1:
        model_type = sys.argv[1].lower()

    if model_type == "lstm":
        model = load_model(vocab, device)
        print("Evaluating LSTM model")
    elif model_type == "transformer":
        from src.models.transformer_seq2seq import TransformerSeq2Seq
        # Initialize model with the exact vocab size from the file
        model = TransformerSeq2Seq(len(vocab)).to(device)
        model.load_state_dict(torch.load("models/transformer_model.pth", weights_only=True))
        model.eval()
        print("Evaluating Transformer model")
    
    predictions = []
    targets = []

    with torch.no_grad():
        for f, t in zip(test_f[:500], test_t[:500]):
            if model_type == "lstm":
                predicted_ids = predict_sequence(model, f, vocab, device)
            else:
                from src.evaluation.predict import predict_sequence_transformer
                predicted_ids = predict_sequence_transformer(model, f, vocab, device)

            prediction = decode_tokens(predicted_ids, vocab)
            predictions.append(prediction)
            targets.append(t)

    symbolic_accuracy = compute_exact_match(predictions, targets)
    print(f"\nEvaluation Results for {model_type.upper()}")
    print("-------------------")
    print("Symbolic Accuracy:", round(symbolic_accuracy, 4))
if __name__ == "__main__":
    evaluate()