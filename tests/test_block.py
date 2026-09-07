import numpy as np

from transformer.block import TransformerBlock


def test_output_shape():
    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        hidden_dim=128
    )

    X = np.random.randn(5, 32)

    output = block.forward(X)

    assert output.shape == X.shape


def test_different_inputs_produce_different_outputs():
    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        hidden_dim=128
    )

    X1 = np.random.randn(5, 32)
    X2 = np.random.randn(5, 32)

    output1 = block.forward(X1)
    output2 = block.forward(X2)

    assert not np.allclose(output1, output2)


def test_backward_output_shape():
    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        hidden_dim=128
    )

    X = np.random.randn(5, 32)

    output = block.forward(X)

    grad_output = np.random.randn(*output.shape)

    grad_X = block.backward(grad_output)

    assert grad_X.shape == X.shape


def test_backward_gradients_are_finite():
    block = TransformerBlock(
        embed_dim=32,
        num_heads=4,
        hidden_dim=128
    )

    X = np.random.randn(5, 32)

    output = block.forward(X)

    grad_output = np.random.randn(*output.shape)

    grad_X = block.backward(grad_output)

    assert np.all(np.isfinite(grad_X))


def test_numerical_gradient():
    np.random.seed(42)

    block = TransformerBlock(
        embed_dim=4,
        num_heads=2,
        hidden_dim=8
    )

    X = np.random.randn(3, 4)

    output = block.forward(X)

    grad_output = np.random.randn(*output.shape)

    analytical_grad = block.backward(grad_output)

    epsilon = 1e-5

    numerical_grad = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            X_plus = X.copy()
            X_minus = X.copy()

            X_plus[i, j] += epsilon
            X_minus[i, j] -= epsilon

            output_plus = block.forward(X_plus)
            output_minus = block.forward(X_minus)

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