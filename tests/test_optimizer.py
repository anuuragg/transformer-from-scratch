import numpy as np

from transformer.optimizer import SGD


def test_parameter_update():
    optimizer = SGD(learning_rate=0.1)

    parameter = np.array([1.0, 2.0, 3.0])
    gradient = np.array([0.5, -1.0, 2.0])

    optimizer.update(parameter, gradient)

    expected = np.array([
        0.95,
        2.10,
        2.80
    ])

    assert np.allclose(parameter, expected)


def test_zero_gradient():
    optimizer = SGD(learning_rate=0.1)

    parameter = np.array([1.0, 2.0, 3.0])
    gradient = np.zeros(3)

    original = parameter.copy()

    optimizer.update(parameter, gradient)

    assert np.allclose(parameter, original)


def test_update_moves_against_gradient():
    optimizer = SGD(learning_rate=0.1)

    parameter = np.array([1.0])
    gradient = np.array([2.0])

    optimizer.update(parameter, gradient)

    assert parameter[0] < 1.0    