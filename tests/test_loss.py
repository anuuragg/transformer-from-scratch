import numpy as np

from transformer.loss import CrossEntropyLoss


def test_loss_is_scalar():
    loss_fn = CrossEntropyLoss()

    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1]
    ])

    targets = np.array([0, 1])

    loss = loss_fn.forward(probs, targets)

    assert np.isscalar(loss)


def test_loss_is_positive():
    loss_fn = CrossEntropyLoss()

    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1]
    ])

    targets = np.array([0, 1])

    loss = loss_fn.forward(probs, targets)

    assert loss > 0


def test_loss_gets_smaller_for_better_predictions():
    loss_fn = CrossEntropyLoss()

    good_probs = np.array([
        [0.9, 0.05, 0.05]
    ])

    bad_probs = np.array([
        [0.1, 0.45, 0.45]
    ])

    targets = np.array([0])

    good_loss = loss_fn.forward(good_probs, targets)
    bad_loss = loss_fn.forward(bad_probs, targets)

    assert good_loss < bad_loss


def test_backward_shape():
    loss_fn = CrossEntropyLoss()

    probs = np.array([
        [0.7, 0.2, 0.1],
        [0.1, 0.8, 0.1]
    ])

    targets = np.array([0, 1])

    loss_fn.forward(probs, targets)

    grad = loss_fn.backward()

    assert grad.shape == probs.shape


def test_backward_known_value():
    loss_fn = CrossEntropyLoss()

    probs = np.array([
        [0.7, 0.2, 0.1]
    ])

    targets = np.array([0])

    loss_fn.forward(probs, targets)

    grad = loss_fn.backward()

    expected = np.array([
        [-0.3, 0.2, 0.1]
    ])

    assert np.allclose(grad, expected)