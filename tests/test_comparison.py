"""
Tests for comparison module.
"""

import pytest

from analyzer.comparison import TopicComparator, compare_topic_lists


class TestTopicComparator:
    """Test cases for TopicComparator class."""

    def setup_method(self):
        """Set up test comparator."""
        self.comparator = TopicComparator()

    def test_jaccard_similarity_identical_sets(self):
        """Test Jaccard similarity with identical sets."""
        set_a = {"a", "b", "c"}
        set_b = {"a", "b", "c"}

        result = self.comparator.jaccard_similarity(set_a, set_b)

        assert result == 1.0

    def test_jaccard_similarity_disjoint_sets(self):
        """Test Jaccard similarity with disjoint sets."""
        set_a = {"a", "b", "c"}
        set_b = {"d", "e", "f"}

        result = self.comparator.jaccard_similarity(set_a, set_b)

        assert result == 0.0

    def test_jaccard_similarity_partial_overlap(self):
        """Test Jaccard similarity with partial overlap."""
        set_a = {"a", "b", "c"}
        set_b = {"b", "c", "d"}

        # Intersection: {b, c} = 2
        # Union: {a, b, c, d} = 4
        # Jaccard: 2/4 = 0.5

        result = self.comparator.jaccard_similarity(set_a, set_b)

        assert result == 0.5

    def test_jaccard_similarity_empty_set(self):
        """Test Jaccard similarity with empty set."""
        set_a = {"a", "b"}
        set_b = set()

        result = self.comparator.jaccard_similarity(set_a, set_b)

        assert result == 0.0

    def test_compare_topics_returns_list(self):
        """Test that compare_topics returns a list."""
        topics_a = [
            {"topic_id": 0, "keywords": ["machine", "learning"], "name": "Topic 0"}
        ]
        topics_b = [{"topic_id": 0, "keywords": ["ml", "ai"], "name": "Topic 0"}]

        result = self.comparator.compare_topics(topics_a, topics_b, "A", "B")

        assert isinstance(result, list)
        assert len(result) == 1

    def test_compare_topics_structure(self):
        """Test structure of comparison result."""
        topics_a = [{"topic_id": 0, "keywords": ["machine"], "name": "Topic A"}]
        topics_b = [{"topic_id": 0, "keywords": ["machine"], "name": "Topic B"}]

        result = self.comparator.compare_topics(
            topics_a, topics_b, "SourceA", "SourceB"
        )

        assert result[0]["source_a"] == "SourceA"
        assert result[0]["source_b"] == "SourceB"
        assert result[0]["topic_a"] == "Topic A"
        assert result[0]["topic_b"] == "Topic B"
        assert "jaccard_similarity" in result[0]
        assert "cosine_similarity" in result[0]

    def test_compare_topics_identical_keywords(self):
        """Test comparison with identical keywords."""
        topics_a = [
            {"topic_id": 0, "keywords": ["machine", "learning", "neural"], "name": "A"}
        ]
        topics_b = [
            {"topic_id": 0, "keywords": ["machine", "learning", "neural"], "name": "B"}
        ]

        result = self.comparator.compare_topics(topics_a, topics_b, "A", "B")

        assert result[0]["jaccard_similarity"] == 1.0
        assert result[0]["cosine_similarity"] == 1.0

    def test_compare_topics_different_keywords(self):
        """Test comparison with different keywords."""
        topics_a = [{"topic_id": 0, "keywords": ["machine"], "name": "A"}]
        topics_b = [{"topic_id": 0, "keywords": ["cooking"], "name": "B"}]

        result = self.comparator.compare_topics(topics_a, topics_b, "A", "B")

        assert result[0]["jaccard_similarity"] == 0.0

    def test_compare_topics_multiple_topics(self):
        """Test comparison with multiple topics."""
        topics_a = [
            {"topic_id": 0, "keywords": ["machine"], "name": "A0"},
            {"topic_id": 1, "keywords": ["nlp"], "name": "A1"},
        ]
        topics_b = [
            {"topic_id": 0, "keywords": ["ml"], "name": "B0"},
            {"topic_id": 1, "keywords": ["text"], "name": "B1"},
        ]

        result = self.comparator.compare_topics(topics_a, topics_b, "A", "B")

        # 2 x 2 = 4 comparisons
        assert len(result) == 4

    def test_is_common_flag(self):
        """Test is_common flag is set correctly."""
        # High similarity - should be common
        topics_a = [
            {"topic_id": 0, "keywords": ["machine", "learning", "neural"], "name": "A"}
        ]
        topics_b = [
            {"topic_id": 0, "keywords": ["machine", "learning", "network"], "name": "B"}
        ]

        result = self.comparator.compare_topics(topics_a, topics_b, "A", "B")

        # With default threshold of 0.15, high similarity should be common
        assert result[0]["is_common"] is True

    def test_get_common_topics(self):
        """Test filtering common topics."""
        comparisons = [
            {
                "source_a": "A",
                "source_b": "B",
                "topic_a": "Common",
                "topic_b": "Common",
                "jaccard_similarity": 0.5,
                "cosine_similarity": 0.7,
                "is_common": True,
            },
            {
                "source_a": "A",
                "source_b": "B",
                "topic_a": "Different",
                "topic_b": "Other",
                "jaccard_similarity": 0.1,
                "cosine_similarity": 0.1,
                "is_common": False,
            },
        ]

        result = self.comparator.get_common_topics(comparisons)

        assert len(result) == 1
        assert result[0]["topic_a"] == "Common"

    def test_get_unique_to_source(self):
        """Test getting unique topics for a source."""
        comparisons = [
            {
                "source_a": "arXiv",
                "source_b": "Habr",
                "topic_a": "arXiv Unique",
                "topic_b": "",
                "keywords_a": ["quantum"],
                "keywords_b": [],
                "jaccard_similarity": 0.1,
                "cosine_similarity": 0.1,
                "is_common": False,
            },
        ]

        result = self.comparator.get_unique_to_source(comparisons, "arXiv")

        assert len(result) == 1
        assert result[0]["source"] == "arXiv"


class TestConvenienceFunction:
    """Test convenience function."""

    def test_compare_topic_lists(self):
        """Test compare_topic_lists convenience function."""
        topics_a = [{"topic_id": 0, "keywords": ["ai"], "name": "A"}]
        topics_b = [{"topic_id": 0, "keywords": ["ml"], "name": "B"}]

        result = compare_topic_lists(topics_a, topics_b, "X", "Y")

        assert len(result) == 1
        assert result[0]["source_a"] == "X"
        assert result[0]["source_b"] == "Y"
