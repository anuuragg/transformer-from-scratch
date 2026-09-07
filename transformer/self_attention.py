import numpy as np

class SelfAttention:
    def __init__(self, embed_dim: int, num_heads: int):
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads")

        self.W_Q = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_K = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_V = np.random.randn(embed_dim, embed_dim) * 0.01
        self.W_O = np.random.randn(embed_dim, embed_dim) * 0.01

    def forward(self,  X: np.ndarray):
        self.Q  = X @ self.W_Q
        self.K  = X @ self.W_K
        self.V  = X @ self.W_V

        self.Q = self.Q.reshape(-1, self.num_heads, self.head_dim).transpose(1, 0, 2)
        self.K = self.K.reshape(-1, self.num_heads, self.head_dim).transpose(1, 0, 2)
        self.V = self.V.reshape(-1, self.num_heads, self.head_dim).transpose(1, 0, 2)

        

        scores = (self.Q @ self.K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)

        mask = np.triu(np.ones((X.shape[0], X.shape[0])), k=1)
        scores = np.where(mask == 1, -np.inf, scores)

        exp_scores = np.exp(scores)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)

        head_outputs = attention_weights @ self.V
        output = head_outputs.reshape(X.shape[0], self.embed_dim)

        output = output @ self.W_O

        return output

