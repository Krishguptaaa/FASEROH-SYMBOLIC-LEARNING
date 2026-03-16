# ML4Sci GSoC 2026 Evaluation: Symbolic Mathematical Learning for FASEROH

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/release/python-3100/)

## Project Overview

This repository is an evaluation project submitted for the **ML4Sci GSoC 2026** contributor selection process, specifically targeting the **FASEROH (Fast Accurate Symbolic Empirical Representation of Histograms)** initiative. 

The objective of this project is to demonstrate competency in handling symbolic mathematics using deep learning pipelines. It focuses on training Sequence-to-Sequence (Seq2Seq) models to compute Taylor series expansions of symbolic mathematical expressions. By treating computational algebra as a neural translation task, this repository serves as a proof-of-concept for the broader symbolic manipulation required in the main FASEROH codebase.

Two distinct neural architectures are implemented and benchmarked:
1. **LSTM-based Seq2Seq** leveraging a dynamic Teacher Forcing schedule.
2. **Transformer-based Seq2Seq** utilizing Multi-Head Attention for global sequence context.

---

## Dataset Engineering & Tokenization

The dataset (`dataset_ready_100k.csv`) was programmatically generated via SymPy. Rather than relying on simple, standard expansions, the generation pipeline introduces rigorous variance to accurately stress-test the models.

### 1. Multi-Point Expansions
To ensure the models learn the underlying calculus rather than memorizing standard Maclaurin series, the dataset mandates expansions around multiple points: `-2, -1, 0, 1, and 2`. 

### 2. Operator Diversity & Normalization
Expressions are constructed using a randomized pool of polynomial terms, `sin`, `cos`, and `exp`. To streamline the vocabulary space and avoid syntax fragmentation, exponentiation operators (`**`) are programmatically normalized to `^` during the preprocessing phase.

### 3. Symbolic Vocabulary Protection
A known challenge in symbolic machine learning is vocabulary explosion caused by trigonometric or exponential functions evaluated at non-zero integer bounds (e.g., sin(1) or exp(-2)). Allowing the tokenizer to split these into arbitrary characters or evaluate them into infinite irrational floats destroys the symbolic integrity of the equation. 

To solve this, these mathematical boundaries are explicitly protected as discrete, frozen tokens (e.g., `SIN_1`, `EXP_-2`) during tokenization.

### Dataset Sample

| Input Function | Target Taylor Expansion |
| :--- | :--- |
| `2*x, -2` | `2*x` |
| `2*x+4, -2` | `2*x+4` |
| `-x*exp(2*x)+2, -2` | `2*x^3*EXP_-4/3+6*x^2*EXP_-4+19*x*EXP_-4+64*EXP_-4/3+2` |
| `-x+sin(x)+8, -1` | `-x^4*SIN_1/24-x^3*SIN_1/6-x^3*COS_1/6-x^2*COS_1/2+x^2*SIN_1/4-x+x*COS_1/2+5*x*SIN_1/6-13*SIN_1/24+5*COS_1/6+8` |
| `3*x+exp(x+5)+cos(x), 1` | `x^4*COS_1/24+x^4*EXP_6/24-x^3*COS_1/6+x^3*SIN_1/6-x^2*SIN_1/2-x^2*COS_1/4+x^2*EXP_6/4-x*SIN_1/2+5*x*COS_1/6+3*x+x*EXP_6/3+13*COS_1/24+5*SIN_1/6+3*EXP_6/8` |

---

## Architectural Benchmarks

Both architectures share a unified embedding space and custom tokenization (`smart_tokenize`) to ensure a strict 1-to-1 baseline comparison. Hyperparameter optimization was conducted systematically via **Optuna** on cloud GPUs to identify optimal latent dimensions, learning rates, and attention heads.

### Evaluation Metrics (100k Dataset)

| Metric | LSTM Seq2Seq | Transformer Seq2Seq | Description |
| :--- | :--- | :--- | :--- |
| **String Exact Match** | `65.37%` | `51.42%` | Perfect character-for-character translation. |
| **Symbolic Equivalence** | `66.10%` | `52.99%` | Verified mathematical equivalence via SymPy `simplify()`. |
| **Sequence Similarity** | `92.00%` | `84.39%` | Levenshtein-based token overlap. |
| **Mean R^2 Score** | `0.6961` | `0.5627` | Numerical fidelity across a continuous domain. |

*Analysis: The LSTM currently outperforms the Transformer in Exact Match accuracy, largely benefiting from a heavily optimized Teacher Forcing decay schedule. However, the Transformer's Multi-Head Attention mechanism demonstrates high Sequence Similarity, proving it successfully maps the global mathematical syntax. The discrepancy highlights the Transformer's sensitivity to sequential ordering in symbolic logic.*

---

## Repository Structure

```text
FASEROH-SYMBOLIC-LEARNING/
│
├── data/
│   ├── raw/                 # Generated raw expressions
│   └── processed/           # Tokenized & protected 100k dataset
│
├── src/
│   ├── models/
│   │   ├── lstm_seq2seq.py         # Encoder/Decoder with TF
│   │   └── transformer_seq2seq.py  # Multi-Head Attention architecture
│   │
│   ├── preprocessing/
│   │   ├── encoder.py              # Tensor conversion
│   │   └── vocabulary.py           # Custom symbolic vocabulary builder
│   │
│   └── training/                   # Model-specific training loops
│
├── experiments/             
│   └── transformer_study.py        # Optuna hyperparameter sweeps
│
├── config.py                # Global hyperparameters & special tokens
└── README.md# FASEROH-SYMBOLIC-LEARNING
```

Limitations & Future Scope:
While this repository successfully establishes a baseline for symbolic translation, there are inherent limitations bounded by the current hardware scope and architectural assumptions.

1. The Inductive Bias of TransformersTransformers rely on positional encodings to understand sequence. However, mathematics is hierarchical, not purely sequential. The Transformer's current Exact Match deficit ($\sim 51\%$) is likely due to standard sinusoidal embeddings failing to capture the strict algebraic order of operations.

Future Scope: Implementing Relative Positional Encodings or replacing the flattened 1D sequence input with Graph Neural Networks (GNNs) or Tree-RNNs to natively ingest the Abstract Syntax Tree (AST) of the equations.

2. Dataset Constriction vs. Compute LimitationsThe dataset is currently capped at 100,000 equations due to single-GPU (RTX 3050) / limited cloud compute boundaries. While sufficient for LSTM convergence, Transformers are notoriously data-hungry and typically require millions of rows to build robust internal representations of complex grammar (like calculus).

Future Scope: Expanding the generation pipeline to 1M+ rows, distributed via PyTorch DDP.

3. Operator ExtensibilityThe current generator is restricted to polynomials, exponents, and basic trigonometry.

Future Scope: Injecting logarithmic (log), hyperbolic (sinh, cosh), and fractional power operators into the generator to verify if the latent space can handle extreme algebraic scaling without catastrophic forgetting.

4. Data Augmentation via CommutativityCurrently, the model treats x + 2 and 2 + x as entirely different input sequences.

Future Scope: Implement a data loader that randomly applies commutative and associative permutations during training. This would artificially expand the dataset's robustness without requiring the generation of new computational graphs.

