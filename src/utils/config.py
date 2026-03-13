"""
Central configuration file for the FASEROH symbolic learning project.
All dataset, model, and training parameters are defined here.
"""

# ======================
# Dataset parameters
# ======================

DATASET_SIZE = 10000

MAX_SEQUENCE_LENGTH = 256


# ======================
# Model parameters
# ======================

EMBEDDING_DIM = 128

HIDDEN_SIZE = 256

NUM_LAYERS = 2


# ======================
# Training parameters
# ======================

BATCH_SIZE = 128

LEARNING_RATE_LSTM = 0.001
LEARNING_RATE_TRANSFORMER = 0.001

EPOCHS = 30


# ======================
# Special tokens
# ======================

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"