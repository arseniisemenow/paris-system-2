"""
TechCrunch RSS collector.
"""

from datetime import datetime
from typing import Optional

import requests
from bs4 import BeautifulSoup

import config
from collectors.base import BaseCollector


class TechCrunchCollector(BaseCollector):
    """Collector for TechCrunch RSS feed."""

    def __init__(self):
        super().__init__("techcrunch")
        self.feed_url = self.source_config.get(
            "feed_url", "https://techcrunch.com/feed/"
        )

    def fetch_articles(
        self, max_results: Optional[int] = None, topic: Optional[str] = None
    ) -> list[dict]:
        """Fetch articles from TechCrunch RSS feed.

        Args:
            max_results: Maximum number of articles to fetch
            topic: Optional topic/keyword to filter by

        Uses RSS which is allowed for scraping.
        """
        response = requests.get(
            self.feed_url,
            headers={
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "application/rss+xml, application/xml, text/xml",
            },
            timeout=30,
        )
        response.raise_for_status()

        articles = self._parse_rss(response.text)

        # Filter by topic if provided
        if topic:
            topic_lower = topic.lower()
            articles = [
                a for a in articles if topic_lower in a.get("title", "").lower()
            ]

        # Limit results
        if max_results:
            articles = articles[:max_results]

        return articles

    def _parse_rss(self, xml_content: str) -> list[dict]:
        """Parse RSS feed into article dictionaries."""
        articles = []
        soup = BeautifulSoup(xml_content, "xml")

        items = soup.find_all("item")

        for item in items:
            try:
                title_elem = item.find("title")
                title = (
                    self._clean_text(title_elem.text)
                    if title_elem and title_elem.text
                    else ""
                )

                link_elem = item.find("link")
                link = link_elem.text if link_elem and link_elem.text else ""

                # Description (content)
                description = ""
                desc_elem = item.find("description")
                if desc_elem and desc_elem.text:
                    description = self._clean_html(desc_elem.text)

                # Published date
                pub_date = item.find("pubDate")
                published_at = None
                if pub_date and pub_date.text:
                    published_at = self._parse_date(pub_date.text)

                articles.append(
                    {
                        "title": title,
                        "content": description,
                        "url": link,
                        "published_at": published_at,
                    }
                )
            except (AttributeError, ValueError):
                continue

        return articles

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace."""
        if not text:
            return ""
        return " ".join(text.split())

    def _clean_html(self, html_content: str) -> str:
        """Remove HTML tags and clean text."""
        if not html_content:
            return ""
        soup = BeautifulSoup(html_content, "html.parser")
        text = soup.get_text()
        return self._clean_text(text)

    def _parse_date(self, date_str: str) -> Optional[str]:
        """Parse various date formats to ISO string."""
        formats = [
            "%a, %d %b %Y %H:%M:%S %z",
            "%Y-%m-%dT%H:%M:%S%z",
            "%Y-%m-%d %H:%M:%S",
        ]

        for fmt in formats:
            try:
                dt = datetime.strptime(date_str.strip(), fmt)
                return dt.isoformat()
            except ValueError:
                continue

        return None


def fetch_techcrunch_articles(max_results: int = 50) -> list[dict]:
    """Convenience function to fetch TechCrunch articles."""
    collector = TechCrunchCollector()
    return collector.fetch_articles(max_results)


if __name__ == "__main__":
    articles = fetch_techcrunch_articles(10)
    print(f"Fetched {len(articles)} articles from TechCrunch")
    for i, article in enumerate(articles[:3], 1):
        print(f"\n{i}. {article['title'][:80]}")
        print(f"   URL: {article['url']}")
