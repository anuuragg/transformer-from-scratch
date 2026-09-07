import numpy as np

from transformer.embedding import Embedding


def test_embedding_initialization():
    embedding = Embedding(65, 32)

    assert embedding.weight.shape == (65, 32)


def test_forward_shape():
    embedding = Embedding(65, 32)

    token_ids = np.array([3, 3, 5])
    output = embedding.forward(token_ids)

    assert output.shape == (3, 32)


def test_forward_lookup():
    embedding = Embedding(65, 32)

    token_ids = np.array([3, 5])
    output = embedding.forward(token_ids)

    assert np.array_equal(output[0], embedding.weight[3])
    assert np.array_equal(output[1], embedding.weight[5])


def test_backward_shape():
    embedding = Embedding(65, 32)

    token_ids = np.array([3, 5])
    grad_output = np.ones((2, 32))

    grad_weights = embedding.backward(token_ids, grad_output)

    assert grad_weights.shape == embedding.weight.shape


def test_backward_accumulates_repeated_tokens():
    embedding = Embedding(6, 2)

    token_ids = np.array([3, 3, 5])

    grad_output = np.array([
        [1, 2],
        [10, 20],
        [100, 200]
    ])

    grad_weights = embedding.backward(token_ids, grad_output)

    assert np.array_equal(grad_weights[3], [11, 22])
    assert np.array_equal(grad_weights[5], [100, 200])


def test_backward_leaves_unused_rows_zero():
    embedding = Embedding(6, 2)

    token_ids = np.array([3, 5])

    grad_output = np.array([
        [1, 2],
        [10, 20]
    ])

    grad_weights = embedding.backward(token_ids, grad_output)

    assert np.array_equal(grad_weights[0], [0, 0])
    assert np.array_equal(grad_weights[1], [0, 0])
    assert np.array_equal(grad_weights[2], [0, 0])
    assert np.array_equal(grad_weights[4], [0, 0])