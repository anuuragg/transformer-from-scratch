import numpy as np
import pytest

from transformer.feedforward import FeedForward


def test_output_shape():
    ffn = FeedForward(
        embed_dim=32,
        hidden_dim=128
    )

    X = np.random.randn(5, 32)

    output = ffn.forward(X)

    assert output.shape == (5, 32)


def test_hidden_shape():
    ffn = FeedForward(
        embed_dim=32,
        hidden_dim=128
    )

    X = np.random.randn(5, 32)

    ffn.forward(X)

    assert ffn.hidden.shape == (5, 128)


def test_relu():
    ffn = FeedForward(
        embed_dim=32,
        hidden_dim=128
    )

    X = np.array([
        [-2.0, -1.0, 0.0, 2.0, 5.0]
    ])

    output = ffn.relu(X)

    expected = np.array([
        [0.0, 0.0, 0.0, 2.0, 5.0]
    ])

    assert np.array_equal(output, expected)


def test_backward_output_shape():
    ffn = FeedForward(
        embed_dim=32,
        hidden_dim=128
    )

    X = np.random.randn(5, 32)

    output = ffn.forward(X)

    grad_output = np.ones_like(output)

    grad_X = ffn.backward(grad_output)

    assert grad_X.shape == X.shape


def test_backward_weight_shapes():
    ffn = FeedForward(
        embed_dim=32,
        hidden_dim=128
    )

    X = np.random.randn(5, 32)

    output = ffn.forward(X)

    grad_output = np.ones_like(output)

    ffn.backward(grad_output)

    assert ffn.grad_W1.shape == (32, 128)
    assert ffn.grad_b1.shape == (128,)
    assert ffn.grad_W2.shape == (128, 32)
    assert ffn.grad_b2.shape == (32,)


def test_backward_gradients_are_finite():
    ffn = FeedForward(
        embed_dim=32,
        hidden_dim=128
    )

    X = np.random.randn(5, 32)

    output = ffn.forward(X)

    grad_output = np.random.randn(*output.shape)

    grad_X = ffn.backward(grad_output)

    assert np.all(np.isfinite(grad_X))
    assert np.all(np.isfinite(ffn.grad_W1))
    assert np.all(np.isfinite(ffn.grad_b1))
    assert np.all(np.isfinite(ffn.grad_W2))
    assert np.all(np.isfinite(ffn.grad_b2))


def test_numerical_gradient():
    np.random.seed(42)

    ffn = FeedForward(
        embed_dim=4,
        hidden_dim=8
    )

    X = np.random.randn(3, 4)

    output = ffn.forward(X)

    grad_output = np.random.randn(*output.shape)

    analytical_grad = ffn.backward(grad_output)

    epsilon = 1e-5

    numerical_grad = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            X_plus = X.copy()
            X_minus = X.copy()

            X_plus[i, j] += epsilon
            X_minus[i, j] -= epsilon

            output_plus = ffn.forward(X_plus)
            output_minus = ffn.forward(X_minus)

            loss_plus = np.sum(output_plus * grad_output)
            loss_minus = np.sum(output_minus * grad_output)

            numerical_grad[i, j] = (
                loss_plus - loss_minus
            ) / (2 * epsilon)

    assert np.allclose(
        analytical_grad,
        numerical_grad,
        rtol=1e-4,
        atol=1e-5
    )