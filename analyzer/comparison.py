"""
Topic comparison module.
"""

from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import config


class TopicComparator:
    """Compare topics across different sources."""

    def __init__(
        self,
        jaccard_threshold: Optional[float] = None,
        cosine_threshold: Optional[float] = None,
    ):
        self.jaccard_threshold = (
            jaccard_threshold or config.COMPARISON["jaccard_threshold"]
        )
        self.cosine_threshold = (
            cosine_threshold or config.COMPARISON["cosine_threshold"]
        )

    def jaccard_similarity(self, set_a: set, set_b: set) -> float:
        """Calculate Jaccard similarity between two sets."""
        if not set_a or not set_b:
            return 0.0

        intersection = len(set_a & set_b)
        union = len(set_a | set_b)

        return intersection / union if union > 0 else 0.0

    def compare_topics(
        self,
        topics_a: list[dict],
        topics_b: list[dict],
        source_a: str = "source_a",
        source_b: str = "source_b",
    ) -> list[dict]:
        """Compare topics from two sources.

        Args:
            topics_a: List of topic dicts from source A
            topics_b: List of topic dicts from source B
            source_a: Name of source A
            source_b: Name of source B

        Returns:
            List of comparison dicts
        """
        comparisons = []

        for topic_a in topics_a:
            # Handle both string (comma-separated) and list keywords
            kw_a_raw = topic_a.get("keywords", "")
            if isinstance(kw_a_raw, str) and kw_a_raw:
                keywords_a = set(kw_a_raw.split(","))
            elif isinstance(kw_a_raw, list):
                keywords_a = set(kw_a_raw)
            else:
                keywords_a = set()
            topic_a_name = topic_a.get("name", f"Topic {topic_a.get('topic_id')}")

            for topic_b in topics_b:
                # Handle both string (comma-separated) and list keywords
                kw_b_raw = topic_b.get("keywords", "")
                if isinstance(kw_b_raw, str) and kw_b_raw:
                    keywords_b = set(kw_b_raw.split(","))
                elif isinstance(kw_b_raw, list):
                    keywords_b = set(kw_b_raw)
                else:
                    keywords_b = set()
                topic_b_name = topic_b.get("name", f"Topic {topic_b.get('topic_id')}")

                # Calculate similarities
                jaccard = self.jaccard_similarity(keywords_a, keywords_b)

                # Calculate cosine similarity using TF-IDF vectors
                cosine = self._cosine_similarity_keywords(keywords_a, keywords_b)

                # Determine if common
                is_common = (
                    jaccard >= self.jaccard_threshold or cosine >= self.cosine_threshold
                )

                comparisons.append(
                    {
                        "source_a": source_a,
                        "source_b": source_b,
                        "topic_a": topic_a_name,
                        "topic_b": topic_b_name,
                        "keywords_a": list(keywords_a),
                        "keywords_b": list(keywords_b),
                        "jaccard_similarity": round(jaccard, 3),
                        "cosine_similarity": round(cosine, 3),
                        "is_common": is_common,
                    }
                )

        return comparisons

    def _cosine_similarity_keywords(self, keywords_a: set, keywords_b: set) -> float:
        """Calculate cosine similarity between keyword sets using TF-IDF."""
        if not keywords_a or not keywords_b:
            return 0.0

        # Convert to sorted lists for consistent ordering
        all_keywords = sorted(keywords_a | keywords_b)

        # Create vectors
        vec_a = np.array([1.0 if kw in keywords_a else 0.0 for kw in all_keywords])
        vec_b = np.array([1.0 if kw in keywords_b else 0.0 for kw in all_keywords])

        # Calculate cosine similarity
        dot_product = np.dot(vec_a, vec_b)
        norm_a = np.linalg.norm(vec_a)
        norm_b = np.linalg.norm(vec_b)

        if norm_a == 0 or norm_b == 0:
            return 0.0

        return dot_product / (norm_a * norm_b)

    def get_common_topics(self, comparisons: list[dict]) -> list[dict]:
        """Filter to only common topics."""
        return [c for c in comparisons if c.get("is_common", False)]

    def get_unique_to_source(
        self,
        comparisons: list[dict],
        source: str,
    ) -> list[dict]:
        """Get topics unique to a source (no matches above threshold)."""
        source_field = f"source_{'a' if comparisons and comparisons[0].get('source_a') == source else 'b'}"

        unique = []
        for comp in comparisons:
            if not comp.get("is_common", False):
                if source == comp.get("source_a"):
                    unique.append(
                        {
                            "source": source,
                            "topic": comp.get("topic_a"),
                            "keywords": comp.get("keywords_a", []),
                            "similar_to": comp.get("topic_b"),
                            "similarity": max(
                                comp.get("jaccard_similarity", 0),
                                comp.get("cosine_similarity", 0),
                            ),
                        }
                    )
                elif source == comp.get("source_b"):
                    unique.append(
                        {
                            "source": source,
                            "topic": comp.get("topic_b"),
                            "keywords": comp.get("keywords_b", []),
                            "similar_to": comp.get("topic_a"),
                            "similarity": max(
                                comp.get("jaccard_similarity", 0),
                                comp.get("cosine_similarity", 0),
                            ),
                        }
                    )

        return unique

    def build_comparison_matrix(self, comparisons: list[dict]) -> np.ndarray:
        """Build similarity matrix from comparisons."""
        # Get unique sources and topics
        sources = list(set(c["source_a"] for c in comparisons))

        # This is a simplified matrix builder
        # For MVP, return similarity scores
        if not comparisons:
            return np.array([])

        # Get max similarities
        max_sim = {}
        for comp in comparisons:
            pair = (comp["source_a"], comp["source_b"])
            max_sim[pair] = max(max_sim.get(pair, 0), comp["cosine_similarity"])

        return max_sim


def compare_topic_lists(
    topics_a: list[dict],
    topics_b: list[dict],
    source_a: str = "source_a",
    source_b: str = "source_b",
) -> list[dict]:
    """Convenience function to compare topic lists."""
    comparator = TopicComparator()
    return comparator.compare_topics(topics_a, topics_b, source_a, source_b)


if __name__ == "__main__":
    # Test comparator
    topics_arxiv = [
        {
            "topic_id": 0,
            "keywords": ["machine", "learning", "neural", "network"],
            "name": "Topic 0",
        },
        {
            "topic_id": 1,
            "keywords": ["natural", "language", "processing", "text"],
            "name": "Topic 1",
        },
    ]

    topics_habr = [
        {
            "topic_id": 0,
            "keywords": ["ml", "machine", "learning", "ai"],
            "name": "Topic 0",
        },
        {
            "topic_id": 1,
            "keywords": ["nlp", "natural", "language", "chatgpt"],
            "name": "Topic 1",
        },
    ]

    comparator = TopicComparator()
    comparisons = comparator.compare_topics(topics_arxiv, topics_habr, "arXiv", "Habr")

    print("Topic comparisons:")
    for comp in comparisons:
        print(f"\n{comp['topic_a']} <-> {comp['topic_b']}")
        print(
            f"  Jaccard: {comp['jaccard_similarity']}, Cosine: {comp['cosine_similarity']}"
        )
        print(f"  Common: {comp['is_common']}")
