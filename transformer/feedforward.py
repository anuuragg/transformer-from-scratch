import numpy as np


class FeedForward:
    def __init__(self, embed_dim: int, hidden_dim: int):
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim

        self.W1 = np.random.randn(embed_dim, hidden_dim) * 0.01
        self.b1 = np.zeros(hidden_dim)

        self.W2 = np.random.randn(hidden_dim, embed_dim) * 0.01
        self.b2 = np.zeros(embed_dim)


    def relu(self, x):
        return np.maximum(0, x)


    def forward(self, X: np.ndarray):
        self.X = X

        self.hidden_pre = X @ self.W1 + self.b1

        self.hidden = self.relu(self.hidden_pre)

        output = self.hidden @ self.W2 + self.b2

        return output


    def backward(self, grad_output):
        # 1. Backward through second linear layer
        self.grad_W2 = self.hidden.T @ grad_output
        self.grad_b2 = np.sum(grad_output, axis=0)

        grad_hidden = grad_output @ self.W2.T


        # 2. Backward through ReLU
        relu_grad = (self.hidden_pre > 0).astype(float)

        grad_hidden_pre = grad_hidden * relu_grad


        # 3. Backward through first linear layer
        self.grad_W1 = self.X.T @ grad_hidden_pre
        self.grad_b1 = np.sum(grad_hidden_pre, axis=0)

        grad_X = grad_hidden_pre @ self.W1.T

        return grad_X
