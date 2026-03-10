import torch
import torch.nn as nn

from src.utils.config import MAX_SEQUENCE_LENGTH

class PositionalEncoding(nn.Module):

    def __init__(self, d_model, max_len=5000):
        super().__init__()

        pe = torch.zeros(max_len, d_model)

        position = torch.arange(0, max_len).unsqueeze(1)

        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-torch.log(torch.tensor(10000.0)) / d_model)
        )

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)

        self.register_buffer("pe", pe)

    def forward(self, x):

        x = x + self.pe[:, : x.size(1)]

        return x

class TransformerSeq2Seq(nn.Module):

    def __init__(
        self,
        vocab_size,
        d_model=128,
        nhead=4,
        num_encoder_layers=2,
        num_decoder_layers=2,
        dim_feedforward=256,
        dropout=0.1,
    ):

        super().__init__()

        self.embedding = nn.Embedding(vocab_size, d_model)

        self.positional_encoding = PositionalEncoding(d_model)

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

        self.d_model = d_model

    def forward(self, src, tgt):

        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model))

        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model))

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        output = self.transformer(src, tgt)

        output = self.fc_out(output)

        return output