import numpy as np
import pytest

from transformer.self_attention import SelfAttention


def test_invalid_num_heads():
    with pytest.raises(ValueError):
        SelfAttention(embed_dim=32, num_heads=3)


def test_qkv_shapes():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(5, 32)

    attention.forward(X)

    assert attention.Q.shape == (4, 5, 8)
    assert attention.K.shape == (4, 5, 8)
    assert attention.V.shape == (4, 5, 8)


def test_output_shape():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(5, 32)

    output = attention.forward(X)

    assert output.shape == (5, 32)


def test_attention_weights_shape():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(5, 32)

    attention.forward(X)

    assert attention.attention_weights.shape == (4, 5, 5)


def test_softmax_rows_sum_to_one():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(5, 32)

    attention.forward(X)

    row_sums = attention.attention_weights.sum(axis=-1)

    assert np.allclose(row_sums, 1.0)


def test_causal_mask():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(5, 32)

    attention.forward(X)

    weights = attention.attention_weights

    for i in range(5):
        for j in range(i + 1, 5):
            assert np.allclose(weights[:, i, j], 0.0)


def test_different_inputs_produce_different_outputs():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X1 = np.random.randn(5, 32)
    X2 = np.random.randn(5, 32)

    output1 = attention.forward(X1)
    output2 = attention.forward(X2)

    assert not np.allclose(output1, output2)


def test_different_sequence_length():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(8, 32)

    output = attention.forward(X)

    assert output.shape == (8, 32)


def test_backward_output_shape():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(5, 32)

    output = attention.forward(X)

    grad_output = np.ones_like(output)

    grad_X = attention.backward(grad_output)

    assert grad_X.shape == X.shape


def test_backward_weight_shapes():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(5, 32)

    output = attention.forward(X)

    grad_output = np.ones_like(output)

    attention.backward(grad_output)

    assert attention.grad_W_Q.shape == (32, 32)
    assert attention.grad_W_K.shape == (32, 32)
    assert attention.grad_W_V.shape == (32, 32)
    assert attention.grad_W_O.shape == (32, 32)


def test_backward_gradients_are_finite():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(5, 32)

    output = attention.forward(X)

    grad_output = np.ones_like(output)

    grad_X = attention.backward(grad_output)

    assert np.all(np.isfinite(grad_X))
    assert np.all(np.isfinite(attention.grad_W_Q))
    assert np.all(np.isfinite(attention.grad_W_K))
    assert np.all(np.isfinite(attention.grad_W_V))
    assert np.all(np.isfinite(attention.grad_W_O))


def test_backward_with_random_gradient():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(5, 32)

    output = attention.forward(X)

    grad_output = np.random.randn(5, 32)

    grad_X = attention.backward(grad_output)

    assert grad_X.shape == (5, 32)
    assert np.all(np.isfinite(grad_X))
    

def test_backward_gradient_check():
    np.random.seed(42)

    attention = SelfAttention(
        embed_dim=8,
        num_heads=2
    )

    X = np.random.randn(3, 8)

    output = attention.forward(X)

    grad_output = np.random.randn(*output.shape)

    analytical_grad = attention.backward(grad_output)

    epsilon = 1e-5

    numerical_grad = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            X_plus = X.copy()
            X_minus = X.copy()

            X_plus[i, j] += epsilon
            X_minus[i, j] -= epsilon

            output_plus = attention.forward(X_plus)
            output_minus = attention.forward(X_minus)

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