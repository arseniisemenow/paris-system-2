"""
arXiv API collector.
"""

from typing import Optional
import urllib.parse
import xml.etree.ElementTree as ET

import requests

import config
from collectors.base import BaseCollector


class ArxivCollector(BaseCollector):
    """Collector for arXiv API."""

    def __init__(self):
        super().__init__("arxiv")
        self.api_url = self.source_config.get(
            "api_url", "http://export.arxiv.org/api/query"
        )
        self.categories = self.source_config.get("categories", ["cs.LG"])
        self.max_results = self.source_config.get("max_results", 100)

    def fetch_articles(self, max_results: Optional[int] = None) -> list[dict]:
        """Fetch articles from arXiv API.

        Uses official arXiv API - no rate limiting issues.
        """
        max_results = max_results or self.max_results

        # Build query for multiple categories
        query = " OR ".join(f"cat:{cat}" for cat in self.categories)
        params = {
            "search_query": query,
            "start": 0,
            "max_results": max_results,
            "sortBy": "submittedDate",
            "sortOrder": "descending",
        }

        response = requests.get(self.api_url, params=params, timeout=30)
        response.raise_for_status()

        return self._parse_atom(response.text)

    def _parse_atom(self, xml_content: str) -> list[dict]:
        """Parse arXiv Atom feed into article dictionaries."""
        articles = []

        # Define namespaces
        ns = {
            "atom": "http://www.w3.org/2005/Atom",
            "arxiv": "http://arxiv.org/schemas/atom",
        }

        root = ET.fromstring(xml_content)

        for entry in root.findall("atom:entry", ns):
            try:
                # Extract fields
                title = self._clean_text(entry.find("atom:title", ns).text or "")
                summary = self._clean_text(entry.find("atom:summary", ns).text or "")
                url = entry.find("atom:id", ns).text or ""

                # Published date
                published = entry.find("atom:published", ns)
                published_at = published.text if published is not None else None

                # Get PDF link
                pdf_link = entry.find("arxiv:pdf", ns)
                pdf_url = pdf_link.text if pdf_link is not None else url

                articles.append(
                    {
                        "title": title,
                        "content": summary,
                        "url": pdf_url or url,
                        "published_at": published_at,
                    }
                )
            except (AttributeError, ET.ParseError) as e:
                # Skip malformed entries
                continue

        return articles

    def _clean_text(self, text: str) -> str:
        """Clean text by removing extra whitespace."""
        if not text:
            return ""
        # Normalize whitespace
        return " ".join(text.split())


def fetch_arxiv_articles(max_results: int = 100) -> list[dict]:
    """Convenience function to fetch arXiv articles."""
    collector = ArxivCollector()
    return collector.fetch_articles(max_results)


if __name__ == "__main__":
    # Test collector
    articles = fetch_arxiv_articles(10)
    print(f"Fetched {len(articles)} articles from arXiv")
    for i, article in enumerate(articles[:3], 1):
        print(f"\n{i}. {article['title'][:80]}")
        print(f"   URL: {article['url']}")
        print(f"   Published: {article['published_at']}")
