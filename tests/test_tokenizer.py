from pathlib import Path
from transformer.tokenizer import Tokenizer

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_ROOT / "data" / "tiny_shakespeare.txt"


with open(DATA_PATH, "r") as f:
    training_text = f.read()


tokenizer = Tokenizer(training_text)

print(f"Dataset size: {len(training_text)} characters")
print(f"Vocabulary size: {len(tokenizer.chars)}")
print(f"Vocabulary: {tokenizer.chars}")


text = "Hello world!"

tokens = tokenizer.encode(text)
decoded_text = tokenizer.decode(tokens)

print(tokens)
print(decoded_text)

assert decoded_text == text

print("Tokenizer test passed!")