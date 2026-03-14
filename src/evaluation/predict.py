import torch
from src.preprocessing.vocabulary import smart_tokenize
from src.utils.config import MAX_SEQUENCE_LENGTH
from src.models.lstm_seq2seq import Encoder, Decoder, Seq2Seq
from src.preprocessing.encoder import encode_dataset

def load_model(vocab, device, model_path="models/lstm_model.pth"):
    emb_dim = 256
    hid_dim = 512
    n_layers = 2
    
    encoder = Encoder(len(vocab), emb_dim, hid_dim, n_layers)
    decoder = Decoder(len(vocab), emb_dim, hid_dim, n_layers)
    model = Seq2Seq(encoder, decoder, device).to(device)
    
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    return model

def predict_sequence(model, input_expr, vocab, device, max_length=MAX_SEQUENCE_LENGTH):
    model.eval()
    
    tokens = smart_tokenize(input_expr)
    
    encoded = encode_dataset([tokens], vocab, max_length)
    
    if not isinstance(encoded, torch.Tensor):
        src = torch.LongTensor(encoded).to(device)
    else:
        src = encoded.to(device)
        
    with torch.no_grad():
        encoder_outputs, hidden, cell = model.encoder(src)
        
        input_token = torch.tensor([vocab["<SOS>"]]).to(device)
        outputs = []

        for _ in range(max_length):
            output, hidden, cell = model.decoder(input_token, hidden, cell, encoder_outputs)
            
            predicted_token = output.argmax(1).item()
            
            if predicted_token == vocab["<EOS>"]:
                break
                
            outputs.append(predicted_token)
            input_token = torch.tensor([predicted_token]).to(device)
            
    return outputs

def predict_sequence_transformer(model, expression, vocab, device, max_len=MAX_SEQUENCE_LENGTH):
    model.eval()
    tokens = smart_tokenize(expression)
    token_ids = [vocab["<SOS>"]] + [vocab.get(t, vocab["<UNK>"]) for t in tokens] + [vocab["<EOS>"]]
    src = torch.LongTensor(token_ids).unsqueeze(0).to(device)
    
    tgt = torch.tensor([[vocab["<SOS>"]]], device=device)
    outputs = []

    with torch.no_grad():
        for _ in range(max_len):
            output = model(src, tgt)
            next_token = output[:, -1, :].argmax(dim=-1)
            token_id = next_token.item()
            
            if token_id == vocab["<EOS>"]:
                break
            outputs.append(token_id)
            tgt = torch.cat([tgt, next_token.unsqueeze(0)], dim=1)
            
    return outputs

def decode_tokens(token_ids, vocab):
    inv_vocab = {v: k for k, v in vocab.items()}
    tokens = [inv_vocab.get(idx, "<UNK>") for idx in token_ids]
    return "".join(tokens)