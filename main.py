"""
Main entry point for Topic Matcher.
"""

import argparse
import sys
from datetime import datetime

from database import Database
from collectors.arxiv_collector import ArxivCollector
from collectors.habr_collector import HabrCollector
from collectors.hackernews_collector import HackerNewsCollector
from analyzer.preprocessing import TextPreprocessor
from analyzer.topic_model import TopicModeler
from analyzer.comparison import TopicComparator
from analyzer.deduplicator import deduplicate_articles
import config


def collect_data(db: Database) -> dict:
    """Collect data from all sources with incremental parsing and deduplication."""
    print("📥 Сбор данных...")

    all_new_articles = []  # For cross-source deduplication

    # Collect from arXiv
    print("  → arXiv...")
    try:
        collector = ArxivCollector()
        articles = collector.fetch_articles(
            max_results=config.SOURCES["arxiv"]["max_results"]
        )

        source_id = db.get_or_create_source(
            name="arXiv", source_type="academic", url=config.SOURCES["arxiv"]["api_url"]
        )

        # Incremental: Get existing URLs, filter new ones
        existing_urls = db.get_existing_urls("arXiv")
        new_articles = [a for a in articles if a.get("url") not in existing_urls]

        # Prepare articles for bulk insert
        article_dicts = [
            {
                "source_id": source_id,
                "title": a["title"],
                "content": a["content"],
                "url": a["url"],
                "published_at": a["published_at"],
            }
            for a in new_articles
        ]

        if article_dicts:
            db.insert_articles_bulk(article_dicts)

        # Update source with total count
        total_count = db.get_article_count_by_source("arXiv")
        db.update_source_fetch(source_id, total_count)

        # Add to all articles for deduplication
        for a in new_articles:
            a["source"] = "arXiv"
        all_new_articles.extend(new_articles)

        print(f"    Собрано {len(new_articles)} новых статей (всего: {total_count})")

    except Exception as e:
        print(f"    ❌ Ошибка сбора arXiv: {e}")

    # Collect from Habr
    print("  → Habr...")
    try:
        collector = HabrCollector()
        articles = collector.fetch_articles(max_results=50)

        source_id = db.get_or_create_source(
            name="Habr",
            source_type="professional",
            url=config.SOURCES["habr"]["feed_url"],
        )

        # Incremental: Get existing URLs, filter new ones
        existing_urls = db.get_existing_urls("Habr")
        new_articles = [a for a in articles if a.get("url") not in existing_urls]

        article_dicts = [
            {
                "source_id": source_id,
                "title": a["title"],
                "content": a["content"],
                "url": a["url"],
                "published_at": a["published_at"],
            }
            for a in new_articles
        ]

        if article_dicts:
            db.insert_articles_bulk(article_dicts)

        # Update source with total count
        total_count = db.get_article_count_by_source("Habr")
        db.update_source_fetch(source_id, total_count)

        # Add to all articles for deduplication
        for a in new_articles:
            a["source"] = "Habr"
        all_new_articles.extend(new_articles)

        print(f"    Собрано {len(new_articles)} новых статей (всего: {total_count})")

    except Exception as e:
        print(f"    ❌ Ошибка сбора Habr: {e}")

    # Collect from Hacker News
    print("  → Hacker News...")
    try:
        collector = HackerNewsCollector()
        articles = collector.fetch_articles(
            max_results=config.SOURCES["hackernews"]["max_results"]
        )

        source_id = db.get_or_create_source(
            name="Hacker News",
            source_type="mass_media",
            url=config.SOURCES["hackernews"]["api_url"],
        )

        # Incremental: Get existing URLs, filter new ones
        existing_urls = db.get_existing_urls("Hacker News")
        new_articles = [a for a in articles if a.get("url") not in existing_urls]

        article_dicts = [
            {
                "source_id": source_id,
                "title": a["title"],
                "content": a["content"],
                "url": a["url"],
                "published_at": a["published_at"],
            }
            for a in new_articles
        ]

        if article_dicts:
            db.insert_articles_bulk(article_dicts)

        # Update source with total count
        total_count = db.get_article_count_by_source("Hacker News")
        db.update_source_fetch(source_id, total_count)

        # Add to all articles for deduplication
        for a in new_articles:
            a["source"] = "Hacker News"
        all_new_articles.extend(new_articles)

        print(f"    Собрано {len(new_articles)} новых статей (всего: {total_count})")

    except Exception as e:
        print(f"    ❌ Ошибка сбора Hacker News: {e}")

    # Cross-source deduplication (only for newly collected articles)
    if all_new_articles:
        print(f"\n🔄 Кросс-источник дедупликация...")
        unique_articles = deduplicate_articles(all_new_articles)
        dupes_count = len(all_new_articles) - len(unique_articles)

        if dupes_count > 0:
            print(f"    Удалено {dupes_count} дубликатов")
            # Note: In production, you'd remove duplicates from DB here
            # For now, we just report the count

    total_articles = sum(
        db.get_article_count_by_source(s["name"]) for s in db.get_all_sources()
    )
    print(f"✅ Сбор данных завершён: {total_articles} статей всего")

    results = {
        "arxiv": db.get_article_count_by_source("arXiv"),
        "habr": db.get_article_count_by_source("Habr"),
        "hackernews": db.get_article_count_by_source("Hacker News"),
    }
    return results


def analyze_topics(db: Database) -> None:
    """Analyze topics for each source."""
    print("\n🔍 Тематическое моделирование...")

    # Clear old topics
    db.clear_topics()

    # Preprocessor
    preprocessor = TextPreprocessor()

    # Get sources
    sources = db.get_all_sources()

    for source in sources:
        source_name = source["name"]
        print(f"  → Анализ {source_name}...")

        # Get articles
        articles = db.get_articles_by_source(source_name)

        if not articles:
            print(f"    Нет статей для {source_name}")
            continue

        # Preprocess content
        texts = []
        for article in articles:
            content = article.get("content", "")
            title = article.get("title", "")
            combined = f"{title} {content}"
            texts.append(preprocessor.preprocess(combined))

        # Filter empty texts
        texts = [t for t in texts if t.strip()]

        if not texts:
            print(f"    Нет текста для анализа")
            continue

        # Topic modeling
        modeler = TopicModeler(n_topics=config.TOPIC_MODEL["n_topics"])
        modeler.fit(texts)

        topics = modeler.get_topics(n_words=config.TOPIC_MODEL["n_top_words"])

        # Save to database
        topic_dicts = [
            {
                "source_id": source["id"],
                "topic_id": t["topic_id"],
                "name": t["name"],
                "keywords": ",".join(t["keywords"]),
                "article_count": len(texts),
            }
            for t in topics
        ]

        db.insert_topics(topic_dicts)

        print(f"    Найдено {len(topics)} тем")

    print("✅ Тематическое моделирование завершено")


def compare_topics(db: Database) -> None:
    """Compare topics across sources."""
    print("\n🔄 Сопоставление тем...")

    # Clear old comparisons
    db.clear_comparisons()

    sources = db.get_all_sources()

    if len(sources) < 2:
        print("  Недостаточно источников для сравнения")
        return

    # Get all topics
    all_topics = db.get_all_topics()

    if not all_topics:
        print("  Нет тем для сравнения")
        return

    # Group by source
    topics_by_source = {}
    for topic in all_topics:
        source_id = topic["source_id"]
        source_name = next(
            (s["name"] for s in sources if s["id"] == source_id), "unknown"
        )

        if source_name not in topics_by_source:
            topics_by_source[source_name] = []

        topics_by_source[source_name].append(topic)

    # Compare all pairs
    comparator = TopicComparator()
    all_comparisons = []

    source_names = list(topics_by_source.keys())

    for i, source_a in enumerate(source_names):
        for source_b in source_names[i + 1 :]:
            print(f"  → Сравнение {source_a} ↔ {source_b}...")

            topics_a = topics_by_source[source_a]
            topics_b = topics_by_source[source_b]

            comparisons = comparator.compare_topics(
                topics_a,
                topics_b,
                source_a,
                source_b,
            )

            all_comparisons.extend(comparisons)

    if all_comparisons:
        # Convert keyword lists to strings for DB
        comp_dicts = [
            {
                "source_a": c["source_a"],
                "source_b": c["source_b"],
                "topic_a": c["topic_a"],
                "topic_b": c["topic_b"],
                "jaccard_similarity": c["jaccard_similarity"],
                "cosine_similarity": c["cosine_similarity"],
                "is_common": 1 if c["is_common"] else 0,
            }
            for c in all_comparisons
        ]

        db.insert_comparisons(comp_dicts)

        common_count = len([c for c in all_comparisons if c["is_common"]])
        print(f"  Найдено {common_count} общих тем")

    print("✅ Сопоставление завершено")


def run_pipeline(db: Database) -> None:
    """Run full analysis pipeline."""
    print("🚀 Запуск анализа Topic Matcher")
    print(f"   Время: {datetime.now().isoformat()}")
    print("-" * 40)

    # Step 1: Collect
    collect_data(db)

    # Step 2: Analyze
    analyze_topics(db)

    # Step 3: Compare
    compare_topics(db)

    print("-" * 40)
    print("✅ Анализ завершён!")
    print(f"\n💡 Запустите UI: streamlit run ui/app.py")


def main():
    """CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Topic Matcher - Analyze topics across sources"
    )

    parser.add_argument(
        "--collect-only",
        action="store_true",
        help="Only collect data, skip analysis",
    )

    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Only analyze existing data",
    )

    parser.add_argument(
        "--compare-only",
        action="store_true",
        help="Only compare existing topics",
    )

    args = parser.parse_args()

    # Initialize database
    db = Database(config.DATABASE_PATH)

    if args.collect_only:
        collect_data(db)
    elif args.analyze_only:
        analyze_topics(db)
    elif args.compare_only:
        compare_topics(db)
    else:
        run_pipeline(db)


def collect_by_topic(
    db: Database,
    topic: str,
    sources: list[str] = None,
    max_per_source: int = 100,
) -> dict:
    """Collect articles by topic from selected sources.

    Args:
        db: Database instance
        topic: Topic/keyword to search for
        sources: List of source names (default: all)
        max_per_source: Max articles per source

    Returns:
        Dict with collection results per source
    """
    from collectors.arxiv_collector import ArxivCollector
    from collectors.habr_collector import HabrCollector
    from collectors.hackernews_collector import HackerNewsCollector

    if sources is None:
        sources = ["arXiv", "Habr", "Hacker News"]

    results = {}
    all_articles = []

    source_collectors = {
        "arXiv": ArxivCollector(),
        "Habr": HabrCollector(),
        "Hacker News": HackerNewsCollector(),
    }

    print(f"\n🔍 Поиск статей по теме: '{topic}'")
    print("-" * 40)

    for source_name in sources:
        if source_name not in source_collectors:
            print(f"  ⚠️ Неизвестный источник: {source_name}")
            continue

        print(f"  → {source_name}...")
        try:
            collector = source_collectors[source_name]
            articles = collector.fetch_articles(max_per_source, topic=topic)

            # Get source ID
            source_id = db.get_or_create_source(
                name=source_name,
                source_type=config.SOURCES.get(
                    source_name.lower().replace(" ", ""), {"type": "unknown"}
                ).get("type", "unknown"),
                url=config.SOURCES.get(source_name.lower().replace(" ", ""), {}).get(
                    "api_url", ""
                ),
            )

            # Get existing URLs for incremental
            existing_urls = db.get_existing_urls(source_name)
            new_articles = [a for a in articles if a.get("url") not in existing_urls]

            # Insert new articles
            article_dicts = [
                {
                    "source_id": source_id,
                    "title": a["title"],
                    "content": a["content"],
                    "url": a["url"],
                    "published_at": a["published_at"],
                }
                for a in new_articles
            ]

            if article_dicts:
                db.insert_articles_bulk(article_dicts)

            # Update source
            total_count = db.get_article_count_by_source(source_name)
            db.update_source_fetch(source_id, total_count)

            # Add to all for deduplication
            for a in new_articles:
                a["source"] = source_name
            all_articles.extend(new_articles)

            results[source_name] = len(new_articles)
            print(
                f"    Найдено {len(new_articles)} новых статей (всего: {total_count})"
            )

        except Exception as e:
            print(f"    ❌ Ошибка: {e}")
            results[source_name] = 0

    # Cross-source deduplication
    if all_articles:
        print(f"\n🔄 Дедупликация...")
        unique = deduplicate_articles(all_articles)
        dupes = len(all_articles) - len(unique)
        if dupes > 0:
            print(f"    Удалено {dupes} дубликатов")

    total = sum(results.values())
    print(f"\n✅ Собрано {total} статей по теме '{topic}'")

    return results


if __name__ == "__main__":
    main()
