import numpy as np


class PositionalEncoding:
    def __init__(self, max_seq_len: int, embed_dim: int):
        self.max_seq_len = max_seq_len
        self.embed_dim = embed_dim

        self.positions = np.arange(max_seq_len)
        self.i = np.arange(embed_dim // 2)

        two_i = 2 * self.i

        exponent = two_i / embed_dim
        denominator = 10000 ** exponent

        angle_rates = 1 / denominator
        angles = np.outer(self.positions, angle_rates)

        self.positional_encoding = np.zeros((max_seq_len, embed_dim))
        self.positional_encoding[:, 0::2] = np.sin(angles)
        self.positional_encoding[:, 1::2] = np.cos(angles)

    
    def forward(self, embeddings):
        positional_embeddings =  self.positional_encoding[:embeddings.shape[0]]
        return embeddings + positional_embeddings

    
    def backward(self, grad_output):
        return grad_output


