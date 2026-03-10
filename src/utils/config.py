"""
Central configuration file for the FASEROH symbolic learning project.
All dataset, model, and training parameters are defined here.
"""

# ======================
# Dataset parameters
# ======================

DATASET_SIZE = 10000

MAX_SEQUENCE_LENGTH = 30


# ======================
# Model parameters
# ======================

EMBEDDING_DIM = 128

HIDDEN_SIZE = 256

NUM_LAYERS = 2


# ======================
# Training parameters
# ======================

BATCH_SIZE = 32

LEARNING_RATE = 0.001

EPOCHS = 30


# ======================
# Special tokens
# ======================

PAD_TOKEN = "<PAD>"
SOS_TOKEN = "<SOS>"
EOS_TOKEN = "<EOS>"
UNK_TOKEN = "<UNK>"