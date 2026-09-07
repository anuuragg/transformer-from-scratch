import numpy as np

from transformer.positional_encoding import PositionalEncoding


def test_shape():
    encoding = PositionalEncoding(10, 32)

    embeddings = np.zeros((5, 32))
    output = encoding.forward(embeddings)

    assert output.shape == (5, 32)


def test_different_positions_have_different_encodings():
    encoding = PositionalEncoding(10, 32)

    assert not np.array_equal(
        encoding.positional_encoding[0],
        encoding.positional_encoding[1]
    )


def test_encoding_is_deterministic():
    encoding1 = PositionalEncoding(10, 32)
    encoding2 = PositionalEncoding(10, 32)

    assert np.array_equal(
        encoding1.positional_encoding,
        encoding2.positional_encoding
    )


def test_backward_passes_gradient_unchanged():
    encoding = PositionalEncoding(10, 32)

    gradient = np.random.randn(5, 32)
    output = encoding.backward(gradient)

    assert np.array_equal(output, gradient)