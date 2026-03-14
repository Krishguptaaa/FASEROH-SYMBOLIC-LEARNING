import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1), :]
        return x

class TransformerSeq2Seq(nn.Module):
    def __init__(self, vocab_size, d_model=256, nhead=8, num_encoder_layers=3, 
                 num_decoder_layers=3, dim_feedforward=1024, dropout=0.1, pad_idx=0):
        super().__init__()
        self.d_model = d_model
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True 
        )
        
        self.fc_out = nn.Linear(d_model, vocab_size)

    def create_mask(self, src, tgt):
        src_seq_len = src.shape[1]
        tgt_seq_len = tgt.shape[1]

        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        src_mask = torch.zeros((src_seq_len, src_seq_len), device=src.device).type(torch.bool)

        src_padding_mask = (src == self.pad_idx)
        tgt_padding_mask = (tgt == self.pad_idx)
        
        return src_mask, tgt_mask, src_padding_mask, tgt_padding_mask

    def forward(self, src, tgt):
        src_mask, tgt_mask, src_padding_mask, tgt_padding_mask = self.create_mask(src, tgt)
        
        src_emb = self.pos_encoder(self.embedding(src) * math.sqrt(self.d_model))
        tgt_emb = self.pos_encoder(self.embedding(tgt) * math.sqrt(self.d_model))
        
        outs = self.transformer(
            src_emb, tgt_emb, 
            src_mask=src_mask, tgt_mask=tgt_mask, 
            memory_mask=None, 
            src_key_padding_mask=src_padding_mask, 
            tgt_key_padding_mask=tgt_padding_mask, 
            memory_key_padding_mask=src_padding_mask
        )
        
        return self.fc_out(outs)