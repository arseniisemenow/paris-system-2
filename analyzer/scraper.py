"""
Full article scraper module.
Fetches full article content from URLs.
"""

import re
from typing import Optional

import requests
from bs4 import BeautifulSoup

import config


class ArticleScraper:
    """Scrape full article content from URLs."""

    def __init__(
        self,
        timeout: Optional[int] = None,
        max_length: Optional[int] = None,
    ):
        self.timeout = timeout or config.FULL_TEXT["timeout"]
        self.max_length = max_length or config.FULL_TEXT["max_length"]
        self.session = requests.Session()
        self.session.headers.update(
            {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
                "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
                "Accept-Language": "en-US,en;q=0.5",
            }
        )

    def scrape(self, url: str) -> str:
        """Scrape full article content from URL.

        Args:
            url: Article URL to scrape

        Returns:
            Full article text content
        """
        if not url:
            return ""

        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()

            soup = BeautifulSoup(response.text, "html.parser")

            # Try to find article content based on site
            content = self._extract_content(soup, url)

            # Truncate to max_length
            if len(content) > self.max_length:
                content = content[: self.max_length]

            return content

        except requests.RequestException:
            return ""

    def _extract_content(self, soup: BeautifulSoup, url: str) -> str:
        """Extract content based on the site."""
        url_lower = url.lower()

        if "habr.com" in url_lower:
            return self._extract_habr(soup)
        elif "arxiv.org" in url_lower:
            return self._extract_arxiv(soup)
        elif "techcrunch.com" in url_lower:
            return self._extract_techcrunch(soup)
        elif "news.ycombinator.com" in url_lower:
            return self._extract_hn(soup)
        else:
            return self._extract_generic(soup)

    def _extract_habr(self, soup: BeautifulSoup) -> str:
        """Extract content from Habr article."""
        # Try article tag first
        article = soup.find("article")
        if article:
            # Remove scripts, styles, and other noise
            for tag in article.find_all(["script", "style", "nav", "footer", "header"]):
                tag.decompose()
            return article.get_text(separator="\n", strip=True)

        # Fallback
        return self._extract_generic(soup)

    def _extract_arxiv(self, soup: BeautifulSoup) -> str:
        """Extract content from arXiv page."""
        # arXiv pages have abstract in specific div
        abstract = soup.find("div", class_="abstract")
        if abstract:
            return abstract.get_text(strip=True)

        # Fallback
        return self._extract_generic(soup)

    def _extract_techcrunch(self, soup: BeautifulSoup) -> str:
        """Extract content from TechCrunch article."""
        article = soup.find("article")
        if article:
            for tag in article.find_all(
                ["script", "style", "nav", "footer", "header", "aside"]
            ):
                tag.decompose()
            return article.get_text(separator="\n", strip=True)

        return self._extract_generic(soup)

    def _extract_hn(self, soup: BeautifulSoup) -> str:
        """Extract content from Hacker News discussion."""
        # HN doesn't have full articles, just return empty
        return ""

    def _extract_generic(self, soup: BeautifulSoup) -> str:
        """Generic content extraction."""
        # Remove scripts and styles
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        # Try to find main content
        main = (
            soup.find("main")
            or soup.find("article")
            or soup.find("div", class_=re.compile(r"content|article|post", re.I))
        )
        if main:
            return main.get_text(separator="\n", strip=True)

        # Last resort: body
        body = soup.find("body")
        if body:
            return body.get_text(separator="\n", strip=True)

        return ""


def scrape_article(url: str) -> str:
    """Convenience function to scrape article content."""
    scraper = ArticleScraper()
    return scraper.scrape(url)


if __name__ == "__main__":
    # Test scraper
    import sys

    if len(sys.argv) > 1:
        url = sys.argv[1]
    else:
        url = "https://habr.com/ru/articles/850970/"

    print(f"Scraping: {url}")
    content = scrape_article(url)
    print(f"Content length: {len(content)} chars")
    print(f"First 500 chars: {content[:500]}")
