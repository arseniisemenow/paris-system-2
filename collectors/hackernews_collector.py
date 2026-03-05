"""
Hacker News API collector.
Uses Algolia HN API for faster fetching.
"""

import time
from datetime import datetime
from typing import Optional

import requests

import config
from collectors.base import BaseCollector


class HackerNewsCollector(BaseCollector):
    """Collector for Hacker News via Algolia API (faster than Firebase)."""

    def __init__(self):
        super().__init__("hackernews")
        self.api_url = "https://hn.algolia.com/api/v1"
        self.max_results = self.source_config.get("max_results", 50)
        self.story_type = self.source_config.get("story_type", "top")

    def fetch_articles(self, max_results: Optional[int] = None) -> list[dict]:
        """Fetch articles from Hacker News Algolia API.

        Uses Algolia API which is faster than Firebase.
        """
        max_results = max_results or self.max_results

        # Choose endpoint based on story type
        if self.story_type == "new":
            endpoint = "/search_by_date"
        else:
            endpoint = "/search"

        params = {
            "tags": "story",
            "hitsPerPage": max_results,
        }

        response = requests.get(
            f"{self.api_url}{endpoint}",
            params=params,
            headers={"User-Agent": "TopicMatcher/1.0"},
            timeout=30,
        )
        response.raise_for_status()

        data = response.json()
        hits = data.get("hits", [])

        # Filter to articles with URLs and deduplicate
        articles = []
        seen_urls = set()

        for hit in hits:
            url = hit.get("url", "")
            if not url:
                continue
            if url in seen_urls:
                continue
            seen_urls.add(url)

            # Parse date
            created_at = hit.get("created_at", "")
            published_at = (
                datetime.fromisoformat(created_at.replace("Z", "+00:00")).isoformat()
                if created_at
                else None
            )

            articles.append(
                {
                    "title": hit.get("title", ""),
                    "content": "",  # Algolia doesn't provide full content
                    "url": url,
                    "published_at": published_at,
                    "score": hit.get("points", 0),
                }
            )

        return articles


def fetch_hackernews_articles(max_results: int = 50) -> list[dict]:
    """Convenience function to fetch Hacker News articles."""
    collector = HackerNewsCollector()
    return collector.fetch_articles(max_results)


if __name__ == "__main__":
    # Test collector
    print("Testing Hacker News collector...")
    articles = fetch_hackernews_articles(10)
    print(f"Fetched {len(articles)} articles\n")

    for i, article in enumerate(articles[:5], 1):
        print(f"{i}. {article['title'][:60]}...")
        print(f"   URL: {article['url'][:50]}...")
        print(f"   Score: {article.get('score', 'N/A')}")
        print()
