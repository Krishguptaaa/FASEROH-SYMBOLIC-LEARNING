import torch
import json
from src.preprocessing.tokenizer import tokenize_expression
from src.preprocessing.vocabulary import encode_tokens
from src.preprocessing.encoder import encode_dataset
from src.utils.config import MAX_SEQUENCE_LENGTH
from src.models.lstm_seq2seq import Encoder, Decoder, Seq2Seq
from src.models.transformer_seq2seq import TransformerSeq2Seq

def load_model(vocab, device):
    vocab_size = len(vocab)
    encoder = Encoder(vocab_size)
    decoder = Decoder(vocab_size)
    model = Seq2Seq(encoder, decoder, device).to(device)
    # Added weights_only=True to silence the FutureWarning
    model.load_state_dict(torch.load("models/lstm_model.pth", weights_only=True))
    model.eval()
    return model

def predict_sequence(model, input_expr, vocab, device, max_length=MAX_SEQUENCE_LENGTH):
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
        if predicted_token == vocab["<EOS>"]:
            break
        outputs.append(predicted_token)
        input_token = torch.tensor([predicted_token]).to(device)
    return outputs

def predict_sequence_transformer(model, expression, vocab, device, max_len=MAX_SEQUENCE_LENGTH):
    model.eval()
    tokens = tokenize_expression(expression)
    # Transform into tensor format [1, seq_len]
    encoded = encode_dataset([tokens], vocab, max_len)
    src = encoded.to(device)
    # Start the target sequence with <SOS>
    tgt = torch.tensor([[vocab["<SOS>"]]], device=device)
    outputs = []

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)
            # Take the prediction for the very last token in the sequence
            next_token = output[:, -1, :].argmax(dim=-1)
            token_id = next_token.item()
            if token_id == vocab["<EOS>"]:
                break
            outputs.append(token_id)
            # Append predicted token to tgt for the next step (Autoregressive)
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
    return outputs

def decode_tokens(token_ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = []
    for idx in token_ids:
        token = inv_vocab.get(idx, "<UNK>")
        if token in ["<SOS>", "<EOS>", "<PAD>"]:
            continue
        tokens.append(token)
    return "".join(tokens) # Recombine digits into numbers