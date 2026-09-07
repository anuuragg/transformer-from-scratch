import numpy as np

from transformer.output_head import OutputHead


def test_logits_shape():
    head = OutputHead(
        embed_dim=32,
        vocab_size=65
    )

    X = np.random.randn(5, 32)

    logits, probs = head.forward(X)

    assert logits.shape == (5, 65)


def test_probability_shape():
    head = OutputHead(
        embed_dim=32,
        vocab_size=65
    )

    X = np.random.randn(5, 32)

    logits, probs = head.forward(X)

    assert probs.shape == (5, 65)


def test_probabilities_sum_to_one():
    head = OutputHead(
        embed_dim=32,
        vocab_size=65
    )

    X = np.random.randn(5, 32)

    logits, probs = head.forward(X)

    row_sums = np.sum(probs, axis=-1)

    assert np.allclose(row_sums, 1.0)


def test_probabilities_are_non_negative():
    head = OutputHead(
        embed_dim=32,
        vocab_size=65
    )

    X = np.random.randn(5, 32)

    logits, probs = head.forward(X)

    assert np.all(probs >= 0)


def test_known_input():
    head = OutputHead(
        embed_dim=2,
        vocab_size=3
    )

    head.W = np.array([
        [1.0, 2.0, 3.0],
        [1.0, 2.0, 3.0]
    ])

    X = np.array([
        [1.0, 0.0]
    ])

    logits, probs = head.forward(X)

    expected_logits = np.array([
        [1.0, 2.0, 3.0]
    ])

    exp_values = np.exp(
        expected_logits - np.max(expected_logits)
    )

    expected_probs = (
        exp_values / np.sum(exp_values)
    )

    assert np.allclose(
        logits,
        expected_logits
    )

    assert np.allclose(
        probs,
        expected_probs
    )