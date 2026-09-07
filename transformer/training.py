import numpy as np

from transformer.loss import CrossEntropyLoss
from transformer.optimizer import SGD


class Trainer:
    def __init__(self, model, learning_rate: float):
        self.model = model
        self.loss_fn = CrossEntropyLoss()
        self.optimizer = SGD(learning_rate)

    def train_step(self, token_ids: np.ndarray, targets: np.ndarray):
        # Forward pass
        logits, probs = self.model.forward(token_ids)

        # Calculate loss
        loss = self.loss_fn.forward(probs, targets)

        # Backward pass
        grad_logits = self.loss_fn.backward()

        self.model.backward(grad_logits)

        # Update embedding
        self.optimizer.update(
            self.model.embedding.weight,
            self.model.embedding.grad_weight
        )

        # Update output head
        self.optimizer.update(
            self.model.output_head.W,
            self.model.output_head.grad_W
        )

        # Update Transformer blocks
        for block in self.model.blocks:

            # Self-attention
            self.optimizer.update(
                block.attention.W_Q,
                block.attention.grad_W_Q
            )

            self.optimizer.update(
                block.attention.W_K,
                block.attention.grad_W_K
            )

            self.optimizer.update(
                block.attention.W_V,
                block.attention.grad_W_V
            )

            self.optimizer.update(
                block.attention.W_O,
                block.attention.grad_W_O
            )

            # Feed-forward
            self.optimizer.update(
                block.feedforward.W1,
                block.feedforward.grad_W1
            )

            self.optimizer.update(
                block.feedforward.b1,
                block.feedforward.grad_b1
            )

            self.optimizer.update(
                block.feedforward.W2,
                block.feedforward.grad_W2
            )

            self.optimizer.update(
                block.feedforward.b2,
                block.feedforward.grad_b2
            )

            # LayerNorm 1
            self.optimizer.update(
                block.norm1.gamma,
                block.norm1.grad_gamma
            )

            self.optimizer.update(
                block.norm1.beta,
                block.norm1.grad_beta
            )

            # LayerNorm 2
            self.optimizer.update(
                block.norm2.gamma,
                block.norm2.grad_gamma
            )

            self.optimizer.update(
                block.norm2.beta,
                block.norm2.grad_beta
            )

        return loss



if __name__ == "__main__":
    from transformer.model import Transformer

    model = Transformer(
        vocab_size=65,
        embed_dim=32,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        max_seq_len=5
    )

    trainer = Trainer(model, 0.01)

    token_ids = np.array([1, 2, 3, 4, 5])
    targets = np.array([2, 3, 4, 5, 6])

    loss = trainer.train_step(token_ids, targets)

    print("loss:", loss)