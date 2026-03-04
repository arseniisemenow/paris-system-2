"""
Tests for preprocessing module.
"""

import pytest

from analyzer.preprocessing import TextPreprocessor, preprocess_text, tokenize_text


class TestTextPreprocessor:
    """Test cases for TextPreprocessor class."""

    def setup_method(self):
        """Set up test preprocessor."""
        self.preprocessor = TextPreprocessor()

    def test_preprocess_lowercase(self):
        """Test that text is converted to lowercase."""
        result = self.preprocessor.preprocess("HELLO World")
        assert result == "hello world"

    def test_preprocess_removes_urls(self):
        """Test that URLs are removed."""
        text = "Check out https://example.com for more info"
        result = self.preprocessor.preprocess(text)
        assert "https://example.com" not in result

    def test_preprocess_removes_emails(self):
        """Test that email addresses are removed."""
        text = "Contact us at test@example.com for help"
        result = self.preprocessor.preprocess(text)
        assert "test@example.com" not in result

    def test_preprocess_removes_numbers(self):
        """Test that numbers are removed."""
        text = "The year 2024 has 365 days"
        result = self.preprocessor.preprocess(text)
        assert "2024" not in result
        assert "365" not in result

    def test_preprocess_removes_special_chars(self):
        """Test that special characters are removed."""
        text = "Hello, World! How are you? #test @user"
        result = self.preprocessor.preprocess(text)
        assert "," not in result
        assert "!" not in result
        assert "#" not in result

    def test_preprocess_handles_empty_string(self):
        """Test handling of empty string."""
        result = self.preprocessor.preprocess("")
        assert result == ""

    def test_preprocess_handles_none(self):
        """Test handling of None."""
        result = self.preprocessor.preprocess(None)
        assert result == ""

    def test_tokenize_returns_list(self):
        """Test that tokenize returns a list."""
        result = self.preprocessor.tokenize("hello world test")
        assert isinstance(result, list)

    def test_tokenize_filters_by_length(self):
        """Test that tokens are filtered by length."""
        # "a" is too short, "hello" is ok
        result = self.preprocessor.tokenize("a hello world")
        assert "a" not in result
        assert "hello" in result

    def test_tokenize_filters_stopwords(self):
        """Test that stopwords are filtered."""
        # "the" and "is" are stopwords
        result = self.preprocessor.tokenize("hello the world is good")
        assert "the" not in result
        assert "is" not in result
        assert "hello" in result
        assert "world" in result

    def test_tokenize_filters_non_alpha(self):
        """Test that non-alphabetic tokens are filtered."""
        result = self.preprocessor.tokenize("hello123 world! test_123")
        assert "hello123" not in result
        assert "test_123" not in result

    def test_preprocess_and_tokenize_pipeline(self):
        """Test full preprocessing pipeline."""
        text = "Check out https://example.com - Machine LEARNING is great!"

        result = self.preprocessor.preprocess_and_tokenize(text)

        assert isinstance(result, list)
        # URLs removed
        assert "example" in result or "check" in result

    def test_preprocess_corpus(self):
        """Test processing multiple texts."""
        texts = ["First text", "Second text", "Third"]

        results = self.preprocessor.preprocess_corpus(texts)

        assert len(results) == 3
        assert results[0] == "first text"

    def test_tokenize_corpus(self):
        """Test tokenizing multiple texts."""
        # Note: tokenize expects preprocessed (lowercased) text
        texts = ["hello world", "test document"]

        results = self.preprocessor.tokenize_corpus(texts)

        assert len(results) == 2
        assert "hello" in results[0]
        assert "world" in results[0]


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_preprocess_text(self):
        """Test preprocess_text convenience function."""
        result = preprocess_text("HELLO World")
        assert result == "hello world"

    def test_tokenize_text(self):
        """Test tokenize_text convenience function."""
        # Note: tokenize expects preprocessed (lowercased) text
        result = tokenize_text("hello the world")
        assert "hello" in result
        assert "world" in result
        assert "the" not in result
