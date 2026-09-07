import numpy as np


class CrossEntropyLoss:
    def __init__(self):
        self.probs = None
        self.targets = None

    def forward(self, probs: np.ndarray, targets: np.ndarray):
        self.probs = probs
        self.targets = targets

        seq_len = probs.shape[0]

        correct_probs = probs[
            np.arange(seq_len),
            targets
        ]

        loss = -np.mean(np.log(correct_probs + 1e-12))

        return loss

    def backward(self):
        seq_len = self.probs.shape[0]

        grad_logits = self.probs.copy()

        grad_logits[
            np.arange(seq_len),
            self.targets
        ] -= 1

        grad_logits /= seq_len

        return grad_logits