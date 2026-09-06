class Tokenizer:
    def __init__(self, text: str):
        self.chars = list(sorted(set(text)))

        self.stoi = {
            char: index
            for index, char in enumerate(self.chars)
        }

        self.itos = {
            index: char
            for index, char in enumerate(self.chars)
        }

    def encode(self, raw_text: str):
        encoded_tokens = []

        for char in raw_text:
            token_id = self.stoi.get(char, 0)
            encoded_tokens.append(token_id)

        return encoded_tokens

    def decode(self, encoded_tokens: list):
        string = ""

        for token_id in encoded_tokens:
            string += self.itos.get(token_id, "?")

        return string