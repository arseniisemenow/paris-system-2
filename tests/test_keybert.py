"""
Test KeyBERT keyword extraction on a random Habr article.
"""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from keybert import KeyBERT

from collectors.habr_collector import HabrCollector


def test_keybert_on_habr_article():
    """Fetch a random Habr article and extract keywords using KeyBERT."""

    # Fetch articles from Habr
    collector = HabrCollector()
    articles = collector.fetch_articles(max_results=10)

    if not articles:
        print("No articles fetched from Habr")
        return

    # Take first article
    article = articles[0]

    print("=" * 60)
    print("ARTICLE TITLE:")
    print("=" * 60)
    print(article["title"])
    print()

    print("=" * 60)
    print("ARTICLE CONTENT (full):")
    print("=" * 60)
    content = article["content"]
    print(content)
    print()

    # Extract keywords using KeyBERT
    kw_model = KeyBERT()

    for ngram in [(1, 1), (1, 2), (1, 3)]:
        print("=" * 60)
        print(f"KEYWORDS (ngram={ngram}):")
        print("=" * 60)

        keywords = kw_model.extract_keywords(
            content,
            keyphrase_ngram_range=ngram,
            stop_words="english",
            top_n=15,
            min_df=1,
        )

        for i, (keyword, score) in enumerate(keywords, 1):
            print(f"  {i:2}. {keyword:<40} (score: {score:.4f})")

        print()

    print("=" * 60)
    print("URL:", article.get("url", "N/A"))
    print("=" * 60)


if __name__ == "__main__":
    test_keybert_on_habr_article()
