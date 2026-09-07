import numpy as np

from transformer.self_attention import SelfAttention
from transformer.feedforward import FeedForward
from transformer.layer_norm import LayerNorm


class TransformerBlock:
    def __init__(
        self,
        embed_dim: int, num_heads: int, hidden_dim: int):
        self.attention = SelfAttention(embed_dim, num_heads)

        self.feedforward = FeedForward(embed_dim, hidden_dim)

        self.norm1 = LayerNorm(embed_dim)
        self.norm2 = LayerNorm(embed_dim)

    
    def forward(self, X: np.ndarray):
        self.X = X

        norm1_output = self.norm1.forward(X)
        attention_output = self.attention.forward(norm1_output)

        self.attention_residual = X + attention_output


        norm2_output = self.norm2.forward(self.attention_residual)
        feedforward_output = self.feedforward.forward(norm2_output)

        output = self.attention_residual + feedforward_output

        return output


    def backward(self, grad_output):
        grad_feedforward = grad_output
        grad_attention_residual = grad_output.copy()

        grad_norm2 = self.feedforward.backward(grad_feedforward)

        grad_attention_residual += self.norm2.backward(grad_norm2)


        grad_X = grad_attention_residual.copy()
        grad_attention = grad_attention_residual

        grad_norm1 = self.attention.backward(grad_attention)

        grad_X += self.norm1.backward(grad_norm1)

        return grad_X
