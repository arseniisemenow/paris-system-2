"""
Tests for database module.
"""

import os
import tempfile
from pathlib import Path

import pytest

from database import Database
from models import Article, Topic, Source


@pytest.fixture
def temp_db():
    """Create a temporary database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)

    db = Database(db_path)

    yield db

    # Cleanup
    if db_path.exists():
        db_path.unlink()


class TestDatabase:
    """Test cases for Database class."""

    def test_init_creates_tables(self, temp_db):
        """Test that database initialization creates tables."""
        # Query sqlite_master for table existence
        import sqlite3

        conn = sqlite3.connect(temp_db.db_path)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN ('sources', 'articles', 'topics', 'topic_comparisons')
        """)

        tables = {row[0] for row in cursor.fetchall()}

        assert "sources" in tables
        assert "articles" in tables
        assert "topics" in tables
        assert "topic_comparisons" in tables

        conn.close()

    def test_get_or_create_source_creates_new(self, temp_db):
        """Test creating a new source."""
        source_id = temp_db.get_or_create_source(
            name="Test Source", source_type="academic", url="https://example.com"
        )

        assert source_id is not None
        assert source_id > 0

    def test_get_or_create_source_returns_existing(self, temp_db):
        """Test that getting existing source returns same ID."""
        source_id_1 = temp_db.get_or_create_source(
            name="Test Source", source_type="academic", url="https://example.com"
        )

        source_id_2 = temp_db.get_or_create_source(
            name="Test Source",
            source_type="different_type",
            url="https://different.com",
        )

        assert source_id_1 == source_id_2

    def test_insert_single_article(self, temp_db):
        """Test inserting a single article."""
        source_id = temp_db.get_or_create_source("Test", "academic", "https://test.com")

        article_id = temp_db.insert_article(
            source_id=source_id,
            title="Test Article",
            content="Test content",
            url="https://example.com/article",
            published_at="2024-01-01T00:00:00",
        )

        assert article_id is not None

        # Verify article was inserted
        articles = temp_db.get_articles_by_source("Test")
        assert len(articles) == 1
        assert articles[0]["title"] == "Test Article"

    def test_insert_articles_bulk(self, temp_db):
        """Test bulk inserting articles."""
        source_id = temp_db.get_or_create_source("Test", "academic", "https://test.com")

        articles = [
            {
                "source_id": source_id,
                "title": f"Article {i}",
                "content": f"Content {i}",
                "url": f"https://example.com/{i}",
                "published_at": None,
            }
            for i in range(10)
        ]

        temp_db.insert_articles_bulk(articles)

        retrieved = temp_db.get_articles_by_source("Test")
        assert len(retrieved) == 10

    def test_insert_topics(self, temp_db):
        """Test inserting topics."""
        source_id = temp_db.get_or_create_source("Test", "academic", "https://test.com")

        topics = [
            {
                "source_id": source_id,
                "topic_id": 0,
                "name": "Topic 1",
                "keywords": "machine,learning,ai",
                "article_count": 10,
            },
            {
                "source_id": source_id,
                "topic_id": 1,
                "name": "Topic 2",
                "keywords": "nlp,text,language",
                "article_count": 8,
            },
        ]

        temp_db.insert_topics(topics)

        retrieved = temp_db.get_topics_by_source("Test")
        assert len(retrieved) == 2
        assert retrieved[0]["name"] == "Topic 1"

    def test_insert_comparisons(self, temp_db):
        """Test inserting topic comparisons."""
        comparisons = [
            {
                "source_a": "arXiv",
                "source_b": "Habr",
                "topic_a": "ML Topic",
                "topic_b": "ML Topic",
                "jaccard_similarity": 0.5,
                "cosine_similarity": 0.7,
                "is_common": 1,
            },
        ]

        temp_db.insert_comparisons(comparisons)

        retrieved = temp_db.get_comparisons()
        assert len(retrieved) == 1
        assert retrieved[0]["source_a"] == "arXiv"
        assert retrieved[0]["is_common"] == 1

    def test_get_common_topics(self, temp_db):
        """Test filtering common topics."""
        comparisons = [
            {
                "source_a": "arXiv",
                "source_b": "Habr",
                "topic_a": "Common",
                "topic_b": "Common",
                "jaccard_similarity": 0.5,
                "cosine_similarity": 0.7,
                "is_common": 1,
            },
            {
                "source_a": "arXiv",
                "source_b": "Habr",
                "topic_a": "Unique",
                "topic_b": "Different",
                "jaccard_similarity": 0.1,
                "cosine_similarity": 0.2,
                "is_common": 0,
            },
        ]

        temp_db.insert_comparisons(comparisons)

        common = temp_db.get_common_topics()
        assert len(common) == 1
        assert common[0]["topic_a"] == "Common"

    def test_clear_articles(self, temp_db):
        """Test clearing articles for a source."""
        source_id = temp_db.get_or_create_source("Test", "academic", "https://test.com")

        temp_db.insert_article(source_id, "Test", "Content", "https://test.com", None)

        assert len(temp_db.get_articles_by_source("Test")) == 1

        temp_db.clear_articles("Test")

        assert len(temp_db.get_articles_by_source("Test")) == 0

    def test_get_all_sources(self, temp_db):
        """Test getting all sources."""
        temp_db.get_or_create_source("Source 1", "academic", "https://1.com")
        temp_db.get_or_create_source("Source 2", "professional", "https://2.com")

        sources = temp_db.get_all_sources()

        assert len(sources) == 2
        assert {s["name"] for s in sources} == {"Source 1", "Source 2"}
