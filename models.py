"""
Data models for Topic Matcher project.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    """Represents an article from any source."""

    id: Optional[int] = None
    source: str = ""
    title: str = ""
    content: str = ""
    url: str = ""
    published_at: Optional[datetime] = None
    topics: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "title": self.title,
            "content": self.content,
            "url": self.url,
            "published_at": self.published_at.isoformat()
            if self.published_at
            else None,
            "topics": ",".join(self.topics) if self.topics else "",
            "keywords": ",".join(self.keywords) if self.keywords else "",
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Topic:
    """Represents a topic extracted from articles."""

    id: Optional[int] = None
    source: str = ""
    topic_id: int = 0
    name: str = ""
    keywords: list[str] = field(default_factory=list)
    article_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "source": self.source,
            "topic_id": self.topic_id,
            "name": self.name,
            "keywords": ",".join(self.keywords) if self.keywords else "",
            "article_count": self.article_count,
        }


@dataclass
class Source:
    """Represents a data source."""

    id: Optional[int] = None
    name: str = ""
    source_type: str = ""  # academic, professional, mass_media
    url: str = ""
    last_fetch: Optional[datetime] = None
    article_count: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name,
            "source_type": self.source_type,
            "url": self.url,
            "last_fetch": self.last_fetch.isoformat() if self.last_fetch else None,
            "article_count": self.article_count,
        }


@dataclass
class TopicComparison:
    """Represents a comparison between topics from different sources."""

    source_a: str = ""
    source_b: str = ""
    topic_a: str = ""
    topic_b: str = ""
    jaccard_similarity: float = 0.0
    cosine_similarity: float = 0.0
    is_common: bool = False

    def to_dict(self) -> dict:
        return {
            "source_a": self.source_a,
            "source_b": self.source_b,
            "topic_a": self.topic_a,
            "topic_b": self.topic_b,
            "jaccard_similarity": self.jaccard_similarity,
            "cosine_similarity": self.cosine_similarity,
            "is_common": self.is_common,
        }
