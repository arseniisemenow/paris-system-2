"""
Text preprocessing module.
"""

import re
from typing import Optional

import config


class TextPreprocessor:
    """Text preprocessing for topic modeling."""

    def __init__(self):
        self.min_word_length = config.PREPROCESSING["min_word_length"]
        self.max_word_length = config.PREPROCESSING["max_word_length"]
        self.stopwords = set(
            config.PREPROCESSING["russian_stopwords"]
            + config.PREPROCESSING["english_stopwords"]
        )

    def preprocess(self, text: str) -> str:
        """Full preprocessing pipeline."""
        if not text:
            return ""

        # Convert to lowercase
        text = text.lower()

        # Remove URLs
        text = re.sub(r"http\S+|www\.\S+", "", text)

        # Remove email addresses
        text = re.sub(r"\S+@\S+", "", text)

        # Remove numbers (optional, keep for some analyses)
        text = re.sub(r"\d+", "", text)

        # Remove special characters but keep spaces
        text = re.sub(r"[^\w\s]", " ", text)

        # Remove extra whitespace
        text = " ".join(text.split())

        return text

    def tokenize(self, text: str) -> list[str]:
        """Tokenize text into words."""
        if not text:
            return []

        # Simple whitespace tokenization
        tokens = text.split()

        # Filter by length and stopwords
        tokens = [
            token
            for token in tokens
            if self.min_word_length <= len(token) <= self.max_word_length
            and token not in self.stopwords
            and token.isalpha()  # Keep only alphabetic tokens
        ]

        return tokens

    def preprocess_and_tokenize(self, text: str) -> list[str]:
        """Full pipeline: preprocess then tokenize."""
        cleaned = self.preprocess(text)
        return self.tokenize(cleaned)

    def preprocess_corpus(self, texts: list[str]) -> list[str]:
        """Preprocess a list of texts."""
        return [self.preprocess(text) for text in texts]

    def tokenize_corpus(self, texts: list[str]) -> list[list[str]]:
        """Tokenize a list of texts."""
        return [self.tokenize(text) for text in texts]


def preprocess_text(text: str) -> str:
    """Convenience function for preprocessing."""
    preprocessor = TextPreprocessor()
    return preprocessor.preprocess(text)


def tokenize_text(text: str) -> list[str]:
    """Convenience function for tokenization."""
    preprocessor = TextPreprocessor()
    return preprocessor.tokenize(text)


if __name__ == "__main__":
    # Test preprocessor
    preprocessor = TextPreprocessor()

    test_text = """
    Machine learning is a subset of artificial intelligence that 
    enables systems to learn from data. 
    https://example.com article@domain.com
    """

    print("Original:", test_text[:50], "...")
    print("Preprocessed:", preprocessor.preprocess(test_text))
    print("Tokenized:", preprocessor.tokenize(test_text))
