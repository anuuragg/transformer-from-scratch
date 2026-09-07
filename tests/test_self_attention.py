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


def test_different_inputs_produce_different_outputs():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X1 = np.random.randn(5, 32)
    X2 = np.random.randn(5, 32)

    output1 = attention.forward(X1)
    output2 = attention.forward(X2)

    assert not np.allclose(output1, output2)


def test_sequence_length():
    attention = SelfAttention(embed_dim=32, num_heads=4)

    X = np.random.randn(8, 32)
    output = attention.forward(X)

    assert output.shape == (8, 32)