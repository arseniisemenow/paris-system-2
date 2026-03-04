"""
Database module for Topic Matcher using SQLite.
"""

import sqlite3
from pathlib import Path
from typing import Optional
from contextlib import contextmanager

import config


class Database:
    """SQLite database manager."""

    def __init__(self, db_path: Optional[Path] = None):
        self.db_path = db_path or config.DATABASE_PATH
        self._ensure_data_dir()
        self._init_db()

    def _ensure_data_dir(self) -> None:
        """Ensure data directory exists."""
        self.db_path.parent.mkdir(parents=True, exist_ok=True)

    @contextmanager
    def _get_connection(self):
        """Context manager for database connection."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def _init_db(self) -> None:
        """Initialize database schema."""
        with self._get_connection() as conn:
            cursor = conn.cursor()

            # Sources table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS sources (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT UNIQUE NOT NULL,
                    source_type TEXT NOT NULL,
                    url TEXT,
                    last_fetch TIMESTAMP,
                    article_count INTEGER DEFAULT 0
                )
            """)

            # Articles table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS articles (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT,
                    url TEXT,
                    published_at TIMESTAMP,
                    keywords TEXT,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                )
            """)

            # Topics table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_id INTEGER NOT NULL,
                    topic_id INTEGER NOT NULL,
                    name TEXT NOT NULL,
                    keywords TEXT,
                    article_count INTEGER DEFAULT 0,
                    FOREIGN KEY (source_id) REFERENCES sources(id)
                )
            """)

            # Topic comparisons table
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS topic_comparisons (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    source_a TEXT NOT NULL,
                    source_b TEXT NOT NULL,
                    topic_a TEXT NOT NULL,
                    topic_b TEXT NOT NULL,
                    jaccard_similarity REAL,
                    cosine_similarity REAL,
                    is_common INTEGER DEFAULT 0
                )
            """)

            conn.commit()

    # ============ Sources ============

    def get_or_create_source(self, name: str, source_type: str, url: str) -> int:
        """Get source ID or create new one."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT id FROM sources WHERE name = ?", (name,))
            row = cursor.fetchone()
            if row:
                return row["id"]
            cursor.execute(
                "INSERT INTO sources (name, source_type, url) VALUES (?, ?, ?)",
                (name, source_type, url),
            )
            return cursor.lastrowid or 0

    def update_source_fetch(self, source_id: int, article_count: int) -> None:
        """Update source after fetching."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                UPDATE sources 
                SET last_fetch = CURRENT_TIMESTAMP, article_count = ?
                WHERE id = ?
            """,
                (article_count, source_id),
            )

    def get_all_sources(self) -> list[dict]:
        """Get all sources."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM sources")
            return [dict(row) for row in cursor.fetchall()]

    # ============ Articles ============

    def insert_article(
        self,
        source_id: int,
        title: str,
        content: str,
        url: str,
        published_at: Optional[str] = None,
    ) -> int:
        """Insert a single article."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO articles (source_id, title, content, url, published_at)
                VALUES (?, ?, ?, ?, ?)
            """,
                (source_id, title, content, url, published_at),
            )
            return cursor.lastrowid or 0

    def insert_articles_bulk(self, articles: list[dict]) -> None:
        """Insert multiple articles at once."""
        if not articles:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT INTO articles (source_id, title, content, url, published_at)
                VALUES (:source_id, :title, :content, :url, :published_at)
            """,
                articles,
            )

    def get_articles_by_source(self, source_name: str) -> list[dict]:
        """Get all articles from a specific source."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT a.* FROM articles a
                JOIN sources s ON a.source_id = s.id
                WHERE s.name = ?
            """,
                (source_name,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_articles(self) -> list[dict]:
        """Get all articles."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM articles")
            return [dict(row) for row in cursor.fetchall()]

    def get_article_count_by_source(self, source_name: str) -> int:
        """Get article count for a source."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT COUNT(*) as count FROM articles a
                JOIN sources s ON a.source_id = s.id
                WHERE s.name = ?
            """,
                (source_name,),
            )
            return cursor.fetchone()["count"]

    # ============ Topics ============

    def insert_topics(self, topics: list[dict]) -> None:
        """Insert multiple topics."""
        if not topics:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                """
                INSERT OR REPLACE INTO topics (source_id, topic_id, name, keywords, article_count)
                VALUES (:source_id, :topic_id, :name, :keywords, :article_count)
            """,
                topics,
            )

    def get_topics_by_source(self, source_name: str) -> list[dict]:
        """Get all topics for a source."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT t.* FROM topics t
                JOIN sources s ON t.source_id = s.id
                WHERE s.name = ?
                ORDER BY t.topic_id
            """,
                (source_name,),
            )
            return [dict(row) for row in cursor.fetchall()]

    def get_all_topics(self) -> list[dict]:
        """Get all topics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM topics")
            return [dict(row) for row in cursor.fetchall()]

    # ============ Comparisons ============

    def insert_comparisons(self, comparisons: list[dict]) -> None:
        """Insert topic comparisons."""
        if not comparisons:
            return
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM topic_comparisons")
            cursor.executemany(
                """
                INSERT INTO topic_comparisons 
                (source_a, source_b, topic_a, topic_b, jaccard_similarity, cosine_similarity, is_common)
                VALUES (:source_a, :source_b, :topic_a, :topic_b, :jaccard_similarity, :cosine_similarity, :is_common)
            """,
                comparisons,
            )

    def get_comparisons(self) -> list[dict]:
        """Get all topic comparisons."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM topic_comparisons")
            return [dict(row) for row in cursor.fetchall()]

    def get_common_topics(self) -> list[dict]:
        """Get topics that are common between sources."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM topic_comparisons WHERE is_common = 1")
            return [dict(row) for row in cursor.fetchall()]

    # ============ Utility ============

    def clear_articles(self, source_name: str) -> None:
        """Clear articles for a source."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                DELETE FROM articles WHERE source_id = 
                (SELECT id FROM sources WHERE name = ?)
            """,
                (source_name,),
            )

    def clear_topics(self) -> None:
        """Clear all topics."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM topics")

    def clear_comparisons(self) -> None:
        """Clear all comparisons."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute("DELETE FROM topic_comparisons")
