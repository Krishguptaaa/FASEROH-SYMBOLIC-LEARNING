import torch
import torch.nn as nn

from src.utils.config import EMBEDDING_DIM, HIDDEN_SIZE, NUM_LAYERS


class Encoder(nn.Module):
    """
    Encoder LSTM for reading the input sequence.
    """

    def __init__(self, vocab_size):
        super(Encoder, self).__init__()

        self.embedding = nn.Embedding(vocab_size, EMBEDDING_DIM)

        self.lstm = nn.LSTM(
            EMBEDDING_DIM,
            HIDDEN_SIZE,
            num_layers=NUM_LAYERS,
            batch_first=True
        )

    def forward(self, input_sequence):
        """
        input_sequence shape:
        [batch_size, sequence_length]
        """

        embedded = self.embedding(input_sequence)

        # embedded shape:
        # [batch_size, sequence_length, embedding_dim]

        outputs, (hidden, cell) = self.lstm(embedded)

        return hidden, cell