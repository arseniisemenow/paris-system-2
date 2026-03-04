"""
Main entry point for Topic Matcher.
"""

import argparse
import sys
from datetime import datetime

from database import Database
from collectors.arxiv_collector import ArxivCollector
from collectors.habr_collector import HabrCollector
from analyzer.preprocessing import TextPreprocessor
from analyzer.topic_model import TopicModeler
from analyzer.comparison import TopicComparator
import config


def collect_data(db: Database) -> dict:
    """Collect data from all sources."""
    print("📥 Сбор данных...")

    results = {"arxiv": 0, "habr": 0}

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

        # Clear old articles if refreshing
        db.clear_articles("arXiv")

        # Prepare articles for bulk insert
        article_dicts = [
            {
                "source_id": source_id,
                "title": a["title"],
                "content": a["content"],
                "url": a["url"],
                "published_at": a["published_at"],
            }
            for a in articles
        ]

        db.insert_articles_bulk(article_dicts)
        db.update_source_fetch(source_id, len(articles))

        results["arxiv"] = len(articles)
        print(f"    Собрано {len(articles)} статей")

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

        # Clear old articles
        db.clear_articles("Habr")

        article_dicts = [
            {
                "source_id": source_id,
                "title": a["title"],
                "content": a["content"],
                "url": a["url"],
                "published_at": a["published_at"],
            }
            for a in articles
        ]

        db.insert_articles_bulk(article_dicts)
        db.update_source_fetch(source_id, len(articles))

        results["habr"] = len(articles)
        print(f"    Собрано {len(articles)} статей")

    except Exception as e:
        print(f"    ❌ Ошибка сбора Habr: {e}")

    print(f"✅ Сбор данных завершён: {sum(results.values())} статей")
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


if __name__ == "__main__":
    main()
