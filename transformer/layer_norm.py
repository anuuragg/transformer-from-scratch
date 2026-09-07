import numpy as np


class LayerNorm:
    def __init__(self, embed_dim: int, eps: float = 1e-5):
        self.embed_dim = embed_dim
        self.eps = eps

        self.gamma = np.ones(embed_dim)
        self.beta = np.zeros(embed_dim)


    def forward(self, X: np.ndarray):
        self.X = X

        self.mean = np.mean(X, axis=-1, keepdims=True)

        self.var = np.var(X, axis=-1, keepdims=True)

        self.normalized = ((X - self.mean) / np.sqrt(self.var + self.eps))

        output = self.gamma * self.normalized + self.beta

        return output


    def backward(self, grad_output):
        N = self.embed_dim

        self.grad_gamma = np.sum(
            grad_output * self.normalized,
            axis=0
        )

        self.grad_beta = np.sum(
            grad_output,
            axis=0
        )

        grad_normalized = grad_output * self.gamma

        inv_std = 1 / np.sqrt(
            self.var + self.eps
        )

        grad_X = (
            (1 / N)
            * inv_std
            * (
                N * grad_normalized
                - np.sum(
                    grad_normalized,
                    axis=-1,
                    keepdims=True
                )
                - self.normalized
                * np.sum(
                    grad_normalized * self.normalized,
                    axis=-1,
                    keepdims=True
                )
            )
        )

        return grad_X


layer_norm = LayerNorm(32)

X = np.random.randn(5, 32)

output = layer_norm.forward(X)

print(output.shape)
print(np.mean(output, axis=-1))
print(np.var(output, axis=-1))