import numpy as np


class SGD:
    def __init__(self, learning_rate: float):
        self.learning_rate = learning_rate

    def update(self, parameter: np.ndarray, gradient: np.ndarray):
        parameter -= self.learning_rate * gradient