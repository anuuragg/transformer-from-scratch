import numpy as np

from transformer.model import Transformer


def create_model():
    return Transformer(
        vocab_size=65,
        embed_dim=32,
        num_heads=4,
        hidden_dim=128,
        num_layers=2,
        max_seq_len=5
    )


def test_logits_shape():
    model = create_model()

    token_ids = np.array([1, 2, 3, 4, 5])

    logits, probs = model.forward(token_ids)

    assert logits.shape == (5, 65)


def test_probability_shape():
    model = create_model()

    token_ids = np.array([1, 2, 3, 4, 5])

    logits, probs = model.forward(token_ids)

    assert probs.shape == (5, 65)


def test_probabilities_sum_to_one():
    model = create_model()

    token_ids = np.array([1, 2, 3, 4, 5])

    logits, probs = model.forward(token_ids)

    row_sums = np.sum(probs, axis=-1)

    assert np.allclose(row_sums, 1.0)


def test_probabilities_are_non_negative():
    model = create_model()

    token_ids = np.array([1, 2, 3, 4, 5])

    logits, probs = model.forward(token_ids)

    assert np.all(probs >= 0)


def test_multiple_blocks():
    model = Transformer(
        vocab_size=65,
        embed_dim=32,
        num_heads=4,
        hidden_dim=128,
        num_layers=3,
        max_seq_len=5
    )

    assert len(model.blocks) == 3

    token_ids = np.array([1, 2, 3, 4, 5])

    logits, probs = model.forward(token_ids)

    assert logits.shape == (5, 65)


def test_different_inputs_produce_different_outputs():
    model = create_model()

    token_ids_1 = np.array([1, 2, 3, 4, 5])
    token_ids_2 = np.array([5, 4, 3, 2, 1])

    logits_1, probs_1 = model.forward(token_ids_1)
    logits_2, probs_2 = model.forward(token_ids_2)

    assert not np.allclose(logits_1, logits_2)


def test_end_to_end_forward():
    model = create_model()

    token_ids = np.array([10, 20, 30, 40, 50])

    logits, probs = model.forward(token_ids)

    assert logits.shape == (5, 65)
    assert probs.shape == (5, 65)

    assert np.all(np.isfinite(logits))
    assert np.all(np.isfinite(probs))