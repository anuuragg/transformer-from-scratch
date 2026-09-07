import numpy as np

from transformer.layer_norm import LayerNorm


def test_output_shape():
    layer_norm = LayerNorm(embed_dim=32)

    X = np.random.randn(5, 32)

    output = layer_norm.forward(X)

    assert output.shape == X.shape


def test_normalized_mean():
    layer_norm = LayerNorm(embed_dim=32)

    X = np.random.randn(5, 32)

    output = layer_norm.forward(X)

    means = np.mean(output, axis=-1)

    assert np.allclose(means, 0.0, atol=1e-5)


def test_normalized_variance():
    layer_norm = LayerNorm(embed_dim=32)

    X = np.random.randn(5, 32)

    output = layer_norm.forward(X)

    variances = np.var(output, axis=-1)

    assert np.allclose(variances, 1.0, atol=1e-4)


def test_gamma_beta_initialization():
    layer_norm = LayerNorm(embed_dim=32)

    assert np.allclose(layer_norm.gamma, 1.0)
    assert np.allclose(layer_norm.beta, 0.0)


def test_backward_output_shape():
    layer_norm = LayerNorm(embed_dim=32)

    X = np.random.randn(5, 32)

    output = layer_norm.forward(X)

    grad_output = np.random.randn(*output.shape)

    grad_X = layer_norm.backward(grad_output)

    assert grad_X.shape == X.shape


def test_backward_parameter_shapes():
    layer_norm = LayerNorm(embed_dim=32)

    X = np.random.randn(5, 32)

    output = layer_norm.forward(X)

    grad_output = np.random.randn(*output.shape)

    layer_norm.backward(grad_output)

    assert layer_norm.grad_gamma.shape == (32,)
    assert layer_norm.grad_beta.shape == (32,)


def test_backward_gradients_are_finite():
    layer_norm = LayerNorm(embed_dim=32)

    X = np.random.randn(5, 32)

    output = layer_norm.forward(X)

    grad_output = np.random.randn(*output.shape)

    grad_X = layer_norm.backward(grad_output)

    assert np.all(np.isfinite(grad_X))
    assert np.all(np.isfinite(layer_norm.grad_gamma))
    assert np.all(np.isfinite(layer_norm.grad_beta))


def test_numerical_gradient():
    np.random.seed(42)

    layer_norm = LayerNorm(embed_dim=4)

    X = np.random.randn(3, 4)

    output = layer_norm.forward(X)

    grad_output = np.random.randn(*output.shape)

    analytical_grad = layer_norm.backward(grad_output)

    epsilon = 1e-5

    numerical_grad = np.zeros_like(X)

    for i in range(X.shape[0]):
        for j in range(X.shape[1]):

            X_plus = X.copy()
            X_minus = X.copy()

            X_plus[i, j] += epsilon
            X_minus[i, j] -= epsilon

            output_plus = layer_norm.forward(X_plus)
            output_minus = layer_norm.forward(X_minus)

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