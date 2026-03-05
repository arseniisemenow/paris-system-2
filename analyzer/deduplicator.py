"""
Article deduplication module for cross-source deduplication.
"""

from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


class ArticleDeduplicator:
    """Cross-source article deduplication using content similarity."""

    def __init__(
        self,
        title_threshold: float = 0.7,
        content_threshold: float = 0.85,
    ):
        self.title_threshold = title_threshold
        self.content_threshold = content_threshold
        self._vectorizer: Optional[TfidfVectorizer] = None

    def deduplicate(self, articles: list[dict]) -> list[dict]:
        """Deduplicate articles across ALL sources.

        Multi-stage process:
        1. URL exact match - keep highest score
        2. Title similarity (Jaccard)
        3. Content similarity (TF-IDF + Cosine)
        """
        if not articles:
            return []

        print(f"  🔄 Deduplicating {len(articles)} articles...")

        # Stage 1: URL exact match
        unique_by_url = self._dedupe_by_url(articles)
        print(f"    → After URL dedup: {len(unique_by_url)}")

        if len(unique_by_url) <= 1:
            return unique_by_url

        # Stage 2: Title similarity
        unique_by_title = self._dedupe_by_title(unique_by_url)
        print(f"    → After title dedup: {len(unique_by_title)}")

        if len(unique_by_title) <= 1:
            return unique_by_title

        # Stage 3: Content similarity
        unique_by_content = self._dedupe_by_content(unique_by_title)
        print(f"    → After content dedup: {len(unique_by_content)}")

        return unique_by_content

    def _dedupe_by_url(self, articles: list[dict]) -> list[dict]:
        """Stage 1: Remove exact URL duplicates, keep highest score."""
        seen_urls = {}  # url -> article

        for article in articles:
            url = self._normalize_url(article.get("url", ""))

            if not url:
                continue

            if url not in seen_urls:
                seen_urls[url] = article
            else:
                # Keep article with higher score
                existing_score = seen_urls[url].get("score", 0)
                current_score = article.get("score", 0)
                if current_score > existing_score:
                    seen_urls[url] = article

        return list(seen_urls.values())

    def _dedupe_by_title(self, articles: list[dict]) -> list[dict]:
        """Stage 2: Remove near-duplicate titles using Jaccard similarity."""
        if len(articles) <= 1:
            return articles

        unique = []
        seen_titles = []  # (normalized_title, article)

        for article in articles:
            title = article.get("title", "").lower().strip()
            title_tokens = set(title.split())

            is_duplicate = False
            for seen_title, seen_article in seen_titles:
                seen_tokens = set(seen_title.split())
                sim = self._jaccard_similarity(title_tokens, seen_tokens)

                if sim >= self.title_threshold:
                    # Keep one with higher score
                    if article.get("score", 0) > seen_article.get("score", 0):
                        unique.remove(seen_article)
                        seen_titles.remove((seen_title, seen_article))
                        unique.append(article)
                        seen_titles.append((title, article))
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique.append(article)
                seen_titles.append((title, article))

        return unique

    def _dedupe_by_content(self, articles: list[dict]) -> list[dict]:
        """Stage 3: Remove similar content using TF-IDF + Cosine similarity."""
        if len(articles) <= 1:
            return articles

        # Prepare texts (title + content)
        texts = []
        for article in articles:
            title = article.get("title", "")
            content = article.get("content", "")
            # Limit text length for performance
            combined = f"{title} {content}"[:5000]
            texts.append(combined)

        try:
            # TF-IDF vectorization
            self._vectorizer = TfidfVectorizer(
                max_features=5000,
                stop_words="english",
                ngram_range=(1, 2),
            )
            tfidf_matrix = self._vectorizer.fit_transform(texts)

            # Compute cosine similarity matrix
            similarities = cosine_similarity(tfidf_matrix)

            # Find duplicates (upper triangle to avoid self-comparison)
            n = len(articles)
            to_remove = set()

            for i in range(n):
                if i in to_remove:
                    continue
                for j in range(i + 1, n):
                    if j in to_remove:
                        continue

                    content_sim = similarities[i, j]

                    if content_sim >= self.content_threshold:
                        # Keep one with higher score
                        score_i = articles[i].get("score", 0)
                        score_j = articles[j].get("score", 0)

                        if score_i >= score_j:
                            to_remove.add(j)
                        else:
                            to_remove.add(i)

            # Return unique articles
            return [a for i, a in enumerate(articles) if i not in to_remove]

        except ValueError:
            # If vectorization fails (e.g., empty texts), return as-is
            return articles

    def _jaccard_similarity(self, set1: set, set2: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set1 or not set2:
            return 0.0
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        return intersection / union if union > 0 else 0.0

    def _normalize_url(self, url: str) -> str:
        """Normalize URL for comparison."""
        if not url:
            return ""
        # Remove trailing slashes, fragments, query params
        url = url.strip().rstrip("/").split("#")[0].split("?")[0]
        return url.lower()


def deduplicate_articles(articles: list[dict]) -> list[dict]:
    """Convenience function for deduplication."""
    dedup = ArticleDeduplicator(
        title_threshold=config.DEDUPLICATION.get("title_threshold", 0.7),
        content_threshold=config.DEDUPLICATION.get("content_threshold", 0.85),
    )
    return dedup.deduplicate(articles)


if __name__ == "__main__":
    # Test deduplicator
    test_articles = [
        {
            "title": "Machine Learning Introduction",
            "content": "Machine learning is a subset of artificial intelligence.",
            "url": "https://example.com/ml-intro",
            "score": 100,
            "source": "arxiv",
        },
        {
            "title": "Machine Learning Introduction",
            "content": "Machine learning is a subset of artificial intelligence.",
            "url": "https://example.com/ml-intro",
            "score": 50,
            "source": "habr",
        },
        {
            "title": "Deep Learning Overview",
            "content": "Deep learning is a part of machine learning.",
            "url": "https://example.com/dl",
            "score": 75,
            "source": "hn",
        },
        {
            "title": "Introduction to Machine Learning",
            "content": "ML basics and fundamentals.",
            "url": "https://example.com/ml-basics",
            "score": 80,
            "source": "arxiv",
        },
    ]

    dedup = ArticleDeduplicator()
    result = dedup.deduplicate(test_articles)

    print(f"\nInput: {len(test_articles)} articles")
    print(f"Output: {len(result)} unique articles\n")

    for i, a in enumerate(result, 1):
        print(f"{i}. {a['title'][:40]}... (score: {a.get('score')})")
