import numpy as np

from transformer.embedding import Embedding
from transformer.positional_encoding import PositionalEncoding
from transformer.block import TransformerBlock
from transformer.output_head import OutputHead


class Transformer:
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        hidden_dim: int,
        num_layers: int,
        max_seq_len: int
    ):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.max_seq_len = max_seq_len

        self.embedding = Embedding(vocab_size, embed_dim)

        self.positional_encoding = PositionalEncoding(
            max_seq_len,
            embed_dim
        )

        self.blocks = [
            TransformerBlock(
                embed_dim,
                num_heads,
                hidden_dim
            )
            for _ in range(num_layers)
        ]

        self.output_head = OutputHead(embed_dim, vocab_size)

    def forward(self, token_ids: np.ndarray):
        self.token_ids = token_ids

        X = self.embedding.forward(token_ids)

        X = self.positional_encoding.forward(X)

        for block in self.blocks:
            X = block.forward(X)

        logits, probs = self.output_head.forward(X)

        return logits, probs

    def backward(self, grad_logits):
        grad_X = grad_logits @ self.output_head.W.T

        self.output_head.grad_W = (
            self.output_head.X.T @ grad_logits
        )

        for block in reversed(self.blocks):
            grad_X = block.backward(grad_X)

        self.embedding.grad_weight = self.embedding.backward(
            self.token_ids,
            grad_X
        )

        return grad_X


if __name__ == "__main__":
    model = Transformer(
        vocab_size=65,
        embed_dim=32,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        max_seq_len=5
    )

    token_ids = np.array([1, 2, 3, 4, 5])

    logits, probs = model.forward(token_ids)

    print("logits:", logits.shape)
    print("probs:", probs.shape)