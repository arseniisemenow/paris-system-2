"""
Keyword extraction module using KeyBERT.
"""

from typing import Optional

from keybert import KeyBERT

import config


class KeywordExtractor:
    """Extract keywords from text using KeyBERT."""

    def __init__(
        self,
        n_keywords: Optional[int] = None,
        ngram_range: Optional[tuple[int, int]] = None,
        model_name: Optional[str] = None,
    ):
        """Initialize keyword extractor.

        Args:
            n_keywords: Number of keywords to extract (default from config)
            ngram_range: N-gram range (default from config)
            model_name: Custom sentence-transformers model
        """
        self.n_keywords = n_keywords or config.KEYWORDS["n_keywords"]
        self.ngram_range = ngram_range or config.KEYWORDS["ngram_range"]

        # Initialize KeyBERT model
        self.model = KeyBERT(model=model_name)

    def extract(self, text: str) -> list[tuple[str, float]]:
        """Extract keywords from a single text.

        Args:
            text: Input text

        Returns:
            List of (keyword, score) tuples
        """
        if not text or not text.strip():
            return []

        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=self.ngram_range,
            stop_words="english",
            top_n=self.n_keywords,
            min_df=config.KEYWORDS["min_df"],
        )

        return keywords

    def extract_keywords_list(self, text: str) -> list[str]:
        """Extract keywords as a list of strings (no scores).

        Args:
            text: Input text

        Returns:
            List of keyword strings
        """
        keywords = self.extract(text)
        return [kw[0] for kw in keywords]

    def extract_from_corpus(
        self, texts: list[str], top_n: Optional[int] = None
    ) -> list[list[tuple[str, float]]]:
        """Extract keywords from a corpus of texts.

        Args:
            texts: List of input texts
            top_n: Override number of keywords per text

        Returns:
            List of keyword lists (one per text)
        """
        n = top_n or self.n_keywords
        results = []

        for text in texts:
            keywords = self.model.extract_keywords(
                text,
                keyphrase_ngram_range=self.ngram_range,
                stop_words="english",
                top_n=n,
                min_df=config.KEYWORDS["min_df"],
            )
            results.append(keywords)

        return results

    def extract_dominant_keywords(self, texts: list[str]) -> list[list[str]]:
        """Extract keywords from corpus as string lists.

        Args:
            texts: List of input texts

        Returns:
            List of keyword string lists
        """
        results = self.extract_from_corpus(texts)
        return [[kw[0] for kw in kws] for kws in results]


def extract_keywords(
    text: str,
    n_keywords: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[tuple[str, float]]:
    """Convenience function to extract keywords.

    Args:
        text: Input text
        n_keywords: Number of keywords to extract
        ngram_range: N-gram range

    Returns:
        List of (keyword, score) tuples
    """
    extractor = KeywordExtractor(
        n_keywords=n_keywords,
        ngram_range=ngram_range,
    )
    return extractor.extract(text)


def extract_keywords_list(
    text: str,
    n_keywords: int = 10,
    ngram_range: tuple[int, int] = (1, 2),
) -> list[str]:
    """Convenience function to extract keywords as strings.

    Args:
        text: Input text
        n_keywords: Number of keywords to extract
        ngram_range: N-gram range

    Returns:
        List of keyword strings
    """
    extractor = KeywordExtractor(
        n_keywords=n_keywords,
        ngram_range=ngram_range,
    )
    return extractor.extract_keywords_list(text)


if __name__ == "__main__":
    # Test keyword extraction
    test_text = """
    Machine learning is a subset of artificial intelligence that enables 
    systems to learn from data without being explicitly programmed. 
    Deep learning uses neural networks with multiple layers to achieve 
    state-of-the-art results in image recognition and natural language processing.
    """

    print("Testing KeyBERT keyword extraction:")
    print("-" * 40)

    # Test with different ngram ranges
    for ngram in [(1, 1), (1, 2), (1, 3)]:
        keywords = extract_keywords(test_text, n_keywords=5, ngram_range=ngram)
        print(f"\nngram={ngram}:")
        for kw, score in keywords:
            print(f"  {kw:<30} ({score:.4f})")
