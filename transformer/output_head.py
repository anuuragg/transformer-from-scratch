import numpy as np


class OutputHead:
    def __init__(self, embed_dim: int, vocab_size: int):
        self.embed_dim = embed_dim
        self.vocab_size = vocab_size

        self.W = np.random.randn(
            embed_dim,
            vocab_size
        ) * 0.01

    def forward(self, X: np.ndarray):
        self.X = X

        logits = X @ self.W

        exp_logits = np.exp(
            logits - np.max(logits, axis=-1, keepdims=True)
        )

        self.probs = (
            exp_logits
            / np.sum(exp_logits, axis=-1, keepdims=True)
        )

        return logits, self.probs