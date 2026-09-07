import numpy as np

class Embedding:
    def __init__ (self, vocab_size: int, embed_dim:  int):
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim

        self.weight = np.random.randn(vocab_size, embed_dim) * 0.01

    def forward(self, token_ids: np.ndarray):
        return self.weight[token_ids]

    def backward(self, token_ids, grad_output):
        grad_weights = np.zeros(self.weight.shape)
        np.add.at(grad_weights, token_ids, grad_output)

        return grad_weights
