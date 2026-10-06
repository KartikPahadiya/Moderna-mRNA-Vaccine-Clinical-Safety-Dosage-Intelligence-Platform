"""Module 2: from-scratch self-attention over an mRNA sequence.

Implements FR-2.1 .. FR-2.5:
  FR-2.1  encode_sequence() turns an A/C/G/U string into a (seq_len, embed_dim) matrix.
  FR-2.2  Invalid characters are rejected with an explicit error.
  FR-2.3  self_attention() returns Q/K/V projections and an (seq_len, seq_len) weight matrix.
  FR-2.4  Every row of the attention matrix sums to 1.0 (softmax sanity check).
  FR-2.5  The weights are rendered as a labeled heatmap in outputs/figures/.

Bonus: multi-head attention (2-4 independent Q/K/V heads, concatenated output).
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / 'outputs' / 'figures'

VOCAB = ['A', 'C', 'G', 'U']
EMBED_DIM = 8
SEED = 42


class InvalidSequenceError(ValueError):
    """Raised when a sequence contains characters outside {A, C, G, U}."""

    def __init__(self, sequence: str):
        bad = sorted(set(sequence) - set(VOCAB))
        super().__init__(
            f'Invalid nucleotide character(s) {bad} in sequence {sequence!r}. '
            f'Only {VOCAB} are allowed.'
        )


def build_embedding_table(embed_dim: int = EMBED_DIM, seed: int = SEED) -> dict[str, np.ndarray]:
    """Toy learnable embedding table: one random vector per nucleotide."""
    rng = np.random.default_rng(seed)
    return {base: rng.standard_normal(embed_dim) for base in VOCAB}


def encode_sequence(sequence: str, embedding_table: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """FR-2.1 / FR-2.2: encode a string into a (seq_len, embed_dim) matrix."""
    sequence = sequence.upper().strip()
    if not (1 <= len(sequence) <= 50):
        raise ValueError(f'Sequence length must be 1-50 bases, got {len(sequence)}')
    if embedding_table is None:
        embedding_table = build_embedding_table()
    try:
        return np.stack([embedding_table[base] for base in sequence])
    except KeyError:
        raise InvalidSequenceError(sequence) from None


def softmax(x: np.ndarray) -> np.ndarray:
    """Row-wise softmax with numerical-stability shift."""
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def self_attention(X: np.ndarray, d_k: int = EMBED_DIM, seed: int = SEED,
                   W_qkv: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None):
    """Scaled dot-product self-attention.

        Attention(Q, K, V) = softmax( (Q · Kᵀ) / √dₖ ) · V

    Returns (attention_weights, output). Each row of the weights matrix is a
    probability distribution over the positions being attended to.
    """
    seq_len, embed_dim = X.shape

    if W_qkv is None:
        rng = np.random.default_rng(seed)
        W_q = rng.standard_normal((embed_dim, d_k)) * 0.1
        W_k = rng.standard_normal((embed_dim, d_k)) * 0.1
        W_v = rng.standard_normal((embed_dim, d_k)) * 0.1
    else:
        W_q, W_k, W_v = W_qkv

    Q = X @ W_q                     # what each position is "looking for"
    K = X @ W_k                     # what each position "offers"
    V = X @ W_v                     # the content each position contributes

    scores = (Q @ K.T) / np.sqrt(d_k)   # pairwise similarity, scaled down
    attention_weights = softmax(scores)  # rows sum to 1
    output = attention_weights @ V
    return attention_weights, output


def multi_head_attention(X: np.ndarray, num_heads: int = 4, d_k: int = EMBED_DIM,
                         seed: int = SEED) -> tuple[list[np.ndarray], np.ndarray]:
    """Bonus: run independent Q/K/V projections per head and concatenate outputs."""
    rng = np.random.default_rng(seed)
    embed_dim = X.shape[1]
    weights, outputs = [], []
    for _ in range(num_heads):
        W = tuple(rng.standard_normal((embed_dim, d_k)) * 0.1 for _ in range(3))
        w, out = self_attention(X, d_k=d_k, W_qkv=W)
        weights.append(w)
        outputs.append(out)
    return weights, np.concatenate(outputs, axis=-1)


def plot_attention(attention_weights: np.ndarray, sequence: str, save_path: str | Path):
    """FR-2.5: labeled heatmap of the attention matrix."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(attention_weights, cmap='viridis')
    ax.set_xticks(range(len(sequence)))
    ax.set_yticks(range(len(sequence)))
    ax.set_xticklabels([f'{i}\n{b}' for i, b in enumerate(sequence)], fontsize=7)
    ax.set_yticklabels([f'{i} {b}' for i, b in enumerate(sequence)], fontsize=7)
    ax.set_xlabel('Attending TO position')
    ax.set_ylabel('Attending FROM position')
    ax.set_title('mRNA Sequence Self-Attention Weights')
    fig.colorbar(im, label='Attention weight')
    fig.tight_layout()
    fig.savefig(save_path, dpi=150)
    plt.close(fig)
    print(f'Saved attention heatmap -> {save_path}')


def interpret(weights: np.ndarray, sequence: str) -> str:
    """Auto-written interpretation of what the heatmap shows."""
    n = len(sequence)
    # Positions that attend most to themselves vs. spread attention out.
    self_attn = np.diag(weights)
    most_self = int(np.argmax(self_attn))
    spread = weights.std(axis=1)
    most_uniform = int(np.argmin(spread))
    return (
        f'Self-attention interpretation ({n} bases)\n'
        f'--------------------------------------\n'
        f'- Every row sums to 1.0, so each base redistributes a fixed budget of\n'
        f'  "attention" across all positions — no information is created or lost.\n'
        f'- Position {most_self} ({sequence[most_self]}) attends most strongly to itself\n'
        f'  (weight {self_attn[most_self]:.2f}); identical bases share the same embedding,\n'
        f'  so same-base pairs score high dot-product similarity in Q·Kᵀ.\n'
        f'- Position {most_uniform} shows the flattest distribution (std {spread[most_uniform]:.3f}),\n'
        f'  spreading attention nearly evenly — it is "unsure" what matters.\n'
        f'- Because W_q/W_k/W_v are randomly initialized, patterns here reflect\n'
        f'  embedding geometry, not learned biology; training would sharpen them.\n'
    )


if __name__ == '__main__':
    sequence = 'ACGUACGUACGGCUA'
    X = encode_sequence(sequence)

    weights, output = self_attention(X)
    # FR-2.4 sanity check: softmax rows sum to 1.
    assert np.allclose(weights.sum(axis=1), 1.0), 'Softmax rows must sum to 1'
    assert weights.shape == (len(sequence), len(sequence))

    plot_attention(weights, sequence, FIG_DIR / 'attention_heatmap.png')
    text = interpret(weights, sequence)
    print(text)
    (ROOT / 'outputs' / 'attention_interpretation.md').write_text(
        '# ' + text.replace('\n', '  \n'), encoding='utf-8')

    # Bonus: multi-head attention runs cleanly and concatenates.
    mh_weights, mh_output = multi_head_attention(X, num_heads=4)
    assert mh_output.shape == (len(sequence), 4 * EMBED_DIM)
    print(f'Multi-head attention OK: {len(mh_weights)} heads, output shape {mh_output.shape}')
