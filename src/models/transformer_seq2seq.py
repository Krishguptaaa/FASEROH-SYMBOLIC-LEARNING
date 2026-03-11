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
        d_model=512,
        nhead=8,
        num_encoder_layers=4,
        num_decoder_layers=4,
        dim_feedforward=512,
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
        # Position-wise scaling
        src = self.embedding(src) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))
        tgt = self.embedding(tgt) * torch.sqrt(torch.tensor(self.d_model, dtype=torch.float32))

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        # --- THE FIX: Create a Causal Mask for the Decoder ---
        tgt_seq_len = tgt.size(1)
        # Masks future positions with -inf so the model can't "see" them
        tgt_mask = torch.triu(torch.ones(tgt_seq_len, tgt_seq_len, device=tgt.device) * float('-inf'), diagonal=1)
        # -----------------------------------------------------

        # Pass tgt_mask to the transformer
        output = self.transformer(src, tgt, tgt_mask=tgt_mask)
        output = self.fc_out(output)

        return output