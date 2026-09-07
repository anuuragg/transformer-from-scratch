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
        self.X = X

        self.Q  = X @ self.W_Q
        self.K  = X @ self.W_K
        self.V  = X @ self.W_V

        self.Q = self.Q.reshape(-1, self.num_heads, self.head_dim).transpose(1, 0, 2)
        self.K = self.K.reshape(-1, self.num_heads, self.head_dim).transpose(1, 0, 2)
        self.V = self.V.reshape(-1, self.num_heads, self.head_dim).transpose(1, 0, 2)

        

        scores = (self.Q @ self.K.transpose(0, 2, 1)) / np.sqrt(self.head_dim)
        self.scores = scores

        mask = np.triu(np.ones((X.shape[0], X.shape[0])), k=1)
        self.mask = mask

        scores = np.where(mask == 1, -np.inf, scores)

        exp_scores = np.exp(scores)
        attention_weights = exp_scores / np.sum(exp_scores, axis=-1, keepdims=True)
        self.attention_weights = attention_weights

        head_outputs = attention_weights @ self.V
        self.head_outputs = head_outputs

        head_outputs = head_outputs.transpose(1, 0, 2)
        output = head_outputs.reshape(X.shape[0], self.embed_dim)
        self.concat_output = output

        output = output @ self.W_O

        return output



    def backward(self, grad_output):
        grad_output_input = grad_output @ self.W_O.T

        self.grad_W_O = self.concat_output.T @ grad_output

        grad_heads = grad_output_input.reshape(
            -1,
            self.num_heads,
            self.head_dim
        ).transpose(1, 0, 2)

        grad_attention_weights = (
            grad_heads @ self.V.transpose(0, 2, 1)
        )

        grad_V = (
            self.attention_weights.transpose(0, 2, 1)
            @ grad_heads
        )

        grad_scores = (
            self.attention_weights
            * (
                grad_attention_weights
                - np.sum(
                    grad_attention_weights * self.attention_weights,
                    axis=-1,
                    keepdims=True
                )
            )
        )

        grad_scores = np.where(
            self.mask == 1,
            0,
            grad_scores
        )

        grad_scores = (
            grad_scores / np.sqrt(self.head_dim)
        )

        grad_Q = grad_scores @ self.K

        grad_K = (
            grad_scores.transpose(0, 2, 1)
            @ self.Q
        )

        grad_Q = grad_Q.transpose(1, 0, 2)
        grad_K = grad_K.transpose(1, 0, 2)
        grad_V = grad_V.transpose(1, 0, 2)

        grad_Q = grad_Q.reshape(
            self.X.shape[0],
            self.embed_dim
        )

        grad_K = grad_K.reshape(
            self.X.shape[0],
            self.embed_dim
        )

        grad_V = grad_V.reshape(
            self.X.shape[0],
            self.embed_dim
        )

        self.grad_W_Q = self.X.T @ grad_Q
        self.grad_W_K = self.X.T @ grad_K
        self.grad_W_V = self.X.T @ grad_V

        grad_X_Q = grad_Q @ self.W_Q.T
        grad_X_K = grad_K @ self.W_K.T
        grad_X_V = grad_V @ self.W_V.T

        grad_X = (
            grad_X_Q
            + grad_X_K
            + grad_X_V
        )

        return grad_X
    


    

