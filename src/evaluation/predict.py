import torch

from src.preprocessing.tokenizer import tokenize_expression
from src.preprocessing.vocabulary import encode_tokens

from src.utils.config import MAX_SEQUENCE_LENGTH

from src.models.lstm_seq2seq import Encoder, Decoder, Seq2Seq


def load_model(vocab, device):

    vocab_size = len(vocab)

    encoder = Encoder(vocab_size)
    decoder = Decoder(vocab_size)

    model = Seq2Seq(encoder, decoder, device).to(device)

    model.load_state_dict(torch.load("models/lstm_model.pth"))

    model.eval()

    return model

def predict_sequence(model, input_expr, vocab, device, max_length=30):

    model.eval()

    tokens = tokenize_expression(input_expr)

    tokens = ["<SOS>"] + tokens + ["<EOS>"]

    encoded = encode_tokens(tokens, vocab)

    padded = encoded + [0] * (max_length - len(encoded))

    src = torch.tensor([padded]).to(device)

    hidden, cell = model.encoder(src)

    input_token = torch.tensor([vocab["<SOS>"]]).to(device)

    outputs = []

    for _ in range(max_length):

        output, hidden, cell = model.decoder(input_token, hidden, cell)

        predicted_token = output.argmax(1).item()

        outputs.append(predicted_token)

        input_token = torch.tensor([predicted_token]).to(device)

        if predicted_token == vocab["<EOS>"]:
            break

    return outputs

def decode_tokens(token_ids, vocab):

    inv_vocab = {v:k for k,v in vocab.items()}

    tokens = []

    for idx in token_ids:

        token = inv_vocab.get(idx, "<UNK>")

        if token in ["<SOS>", "<EOS>", "<PAD>"]:
            continue

        tokens.append(token)

    return " ".join(tokens)

def test_prediction(model, vocab, device):

    expr = "sin(x)"

    predicted_ids = predict_sequence(
        model,
        expr,
        vocab,
        device
    )

    prediction = decode_tokens(predicted_ids, vocab)

    print("Input:", expr)
    print("Predicted Taylor:", prediction)

