from pathlib import Path

from transformer.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tiny_shakespeare.txt"


with open(DATA_PATH, "r") as f:
    training_text = f.read()


tokenizer = Tokenizer(training_text)


def test_vocabulary():
    assert len(tokenizer.chars) == 65
    assert len(tokenizer.stoi) == 65
    assert len(tokenizer.itos) == 65


def test_encode():
    text = "Hello world!"

    tokens = tokenizer.encode(text)

    assert len(tokens) == len(text)
    assert all(isinstance(token, int) for token in tokens)


def test_decode():
    text = "Hello world!"

    tokens = tokenizer.encode(text)
    decoded_text = tokenizer.decode(tokens)

    assert decoded_text == text


def test_encode_decode_roundtrip():
    text = "To be, or not to be, that is the question."

    assert tokenizer.decode(tokenizer.encode(text)) == text