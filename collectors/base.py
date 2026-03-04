"""
Base collector class.
"""

import time
from abc import ABC, abstractmethod
from typing import Optional

import config


class BaseCollector(ABC):
    """Base class for data collectors."""

    def __init__(self, source_name: str):
        self.source_name = source_name
        self.source_config = config.SOURCES.get(source_name, {})

    @abstractmethod
    def fetch_articles(self, max_results: int = 100) -> list[dict]:
        """Fetch articles from the source.

        Returns:
            List of article dictionaries with keys:
            - title: str
            - content: str
            - url: str
            - published_at: Optional[str] (ISO format)
        """
        pass

    def _rate_limit(self, delay: float = 1.0) -> None:
        """Apply rate limiting between requests."""
        time.sleep(delay)
