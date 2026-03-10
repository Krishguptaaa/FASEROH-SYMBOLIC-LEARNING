import torch
import pandas as pd
from sympy import sympify, simplify

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

    tokenized_inputs, tokenized_outputs = tokenize_data(functions, expansions)

    vocab = build_vocabulary(tokenized_inputs + tokenized_outputs)

    model = load_model(vocab, device)

    predictions = []
    targets = []

    for f, t in zip(functions[:500], expansions[:500]):

        predicted_ids = predict_sequence(model, f, vocab, device)

        prediction = decode_tokens(predicted_ids, vocab)

        predictions.append(prediction)
        targets.append(t)

    symbolic_accuracy = compute_exact_match(predictions, targets)
   

    print("\nEvaluation Results")
    print("-------------------")
    print("symbolic Accuracy:", round(symbolic_accuracy, 4))
    


if __name__ == "__main__":
    evaluate()