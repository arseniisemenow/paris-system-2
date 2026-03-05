"""
Streamlit UI for Topic Matcher.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from database import Database
from config import DATABASE_PATH
from main import collect_by_topic
from analyzer.preprocessing import TextPreprocessor
from analyzer.topic_model import TopicModeler
from analyzer.comparison import TopicComparator
import config


def init_session_state():
    """Initialize Streamlit session state."""
    if "db" not in st.session_state:
        st.session_state.db = Database(DATABASE_PATH)
    if "sources" not in st.session_state:
        st.session_state.sources = ["arXiv", "Habr", "Hacker News"]


def load_data():
    """Load all data from database."""
    db = st.session_state.db

    sources = db.get_all_sources()
    articles_count = {s["name"]: s["article_count"] for s in sources}

    topics = db.get_all_topics()
    comparisons = db.get_comparisons()

    return {
        "sources": sources,
        "articles_count": articles_count,
        "topics": topics,
        "comparisons": comparisons,
    }


def render_overview(data: dict):
    """Render overview page."""
    st.header("📊 Обзор")

    # Stats cards
    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Источники", len(data["sources"]))

    with col2:
        total_articles = sum(data["articles_count"].values())
        st.metric("Всего статей", total_articles)

    with col3:
        common_topics = [c for c in data["comparisons"] if c.get("is_common")]
        st.metric("Общих тем", len(common_topics))

    st.divider()

    # Source details
    st.subheader("📁 Источники данных")

    if data["sources"]:
        sources_df = pd.DataFrame(data["sources"])
        st.dataframe(
            sources_df[["name", "source_type", "article_count", "last_fetch"]],
            use_container_width=True,
        )
    else:
        st.info("Данные ещё не собраны. Запустите сбор данных.")


def render_topics(data: dict):
    """Render topics page."""
    st.header("🏷️ Темы")

    if not data["topics"]:
        st.info("Темы ещё не извлечены. Запустите анализ.")
        return

    # Group topics by source
    topics_by_source = {}
    for topic in data["topics"]:
        source = topic.get("source_id")  # Need to join with sources
        if source not in topics_by_source:
            topics_by_source[source] = []
        topics_by_source[source].append(topic)

    # Display topics by source
    for source in data["sources"]:
        st.subheader(f"📚 {source['name']}")

        source_topics = [
            t for t in data["topics"] if t.get("source_id") == source["id"]
        ]

        if source_topics:
            for topic in source_topics:
                keywords = (
                    topic.get("keywords", "").split(",")
                    if topic.get("keywords")
                    else []
                )
                with st.expander(
                    f"**{topic['name']}** ({topic['article_count']} статей)"
                ):
                    st.write("**Ключевые слова:**")
                    st.write(", ".join(keywords[:10]))
        else:
            st.write("Нет тем")

        st.divider()


def render_comparisons(data: dict):
    """Render topic comparisons page."""
    st.header("🔄 Сопоставление тем")

    if not data["comparisons"]:
        st.info("Сравнение ещё не выполнено.")
        return

    comparisons_df = pd.DataFrame(data["comparisons"])

    # Filter options
    col1, col2 = st.columns(2)

    with col1:
        show_common = st.toggle("Только общие темы", value=False)

    with col2:
        min_similarity = st.slider("Минимальная схожесть", 0.0, 1.0, 0.15)

    # Filter
    filtered = comparisons_df[
        (comparisons_df["jaccard_similarity"] >= min_similarity)
        | (comparisons_df["cosine_similarity"] >= min_similarity)
    ]

    if show_common:
        filtered = filtered[filtered["is_common"] == 1]

    st.subheader(f"Найдено сопоставлений: {len(filtered)}")

    # Display as table
    if not filtered.empty:
        display_cols = [
            "source_a",
            "topic_a",
            "source_b",
            "topic_b",
            "jaccard_similarity",
            "cosine_similarity",
            "is_common",
        ]
        available_cols = [c for c in display_cols if c in filtered.columns]
        st.dataframe(filtered[available_cols], use_container_width=True)

        # Heatmap of similarities
        if len(filtered) > 0:
            st.subheader("🗺️ Тепловая карта схожести")

            # Create pivot table
            sources = list(
                set(filtered["source_a"].unique()) | set(filtered["source_b"].unique())
            )

            # Simplified heatmap data
            heatmap_data = []
            for _, row in filtered.iterrows():
                heatmap_data.append(
                    {
                        "x": row["topic_a"][:30] if row["topic_a"] else "",
                        "y": row["topic_b"][:30] if row["topic_b"] else "",
                        "similarity": row["cosine_similarity"],
                    }
                )

            if heatmap_data:
                heatmap_df = pd.DataFrame(heatmap_data)

                fig = px.density_heatmap(
                    heatmap_df,
                    x="x",
                    y="y",
                    z="similarity",
                    color_continuous_scale="Viridis",
                    title="Cosine Similarity between Topics",
                )
                st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("Нет сопоставлений с выбранными параметрами.")


def render_articles(data: dict):
    """Render articles page."""
    st.header("📄 Статьи")

    db = st.session_state.db

    # Source selector
    source_names = [s["name"] for s in data["sources"]]

    if not source_names:
        st.info("Нет данных о статьях.")
        return

    selected_source = st.selectbox("Выберите источник", source_names)

    articles = db.get_articles_by_source(selected_source)

    st.write(f"Статей в {selected_source}: {len(articles)}")

    if articles:
        articles_df = pd.DataFrame(articles)

        # Show article titles
        st.subheader("Список статей")

        for i, article in enumerate(articles[:50], 1):
            with st.expander(f"{i}. {article.get('title', 'Без названия')[:80]}..."):
                st.write(f"**URL:** {article.get('url', 'N/A')}")
                st.write(f"**Опубликовано:** {article.get('published_at', 'N/A')}")
                content = article.get("content", "")
                if content:
                    st.write(f"**Содержание:** {content[:300]}...")


def render_collect_by_topic(data: dict):
    """Render topic-based collection page."""
    st.header("🔍 Сбор по теме")

    # Topic input
    topic = st.text_input("Введите тему для поиска", value="AI").strip()

    # Source selection
    st.subheader("Выберите источники")
    col1, col2, col3 = st.columns(3)

    with col1:
        use_arxiv = st.checkbox("arXiv (академические)", value=True)
    with col2:
        use_habr = st.checkbox("Habr (профессиональные)", value=True)
    with col3:
        use_hn = st.checkbox("Hacker News (СМИ)", value=True)

    sources = []
    if use_arxiv:
        sources.append("arXiv")
    if use_habr:
        sources.append("Habr")
    if use_hn:
        sources.append("Hacker News")

    # Max articles per source
    max_per_source = st.slider("Макс. статей на источник", 10, 100, 50, step=10)

    # Collect button
    if st.button("🚀 Собрать статьи", type="primary"):
        if not topic:
            st.error("Введите тему для поиска")
            return

        if not sources:
            st.error("Выберите хотя бы один источник")
            return

        with st.spinner("Сбор статей..."):
            db = st.session_state.db

            try:
                results = collect_by_topic(
                    db,
                    topic=topic,
                    sources=sources,
                    max_per_source=max_per_source,
                )

                # Show results
                st.success(f"✅ Собрано статей: {sum(results.values())}")

                # Run analysis
                st.info("Запуск анализа тем...")
                analyze_topics_for_sources(db, sources)

                st.success("✅ Анализ завершён!")

                # Show collected articles
                st.subheader("Собранные статьи")
                for source_name, count in results.items():
                    if count > 0:
                        st.write(f"**{source_name}:** {count} статей")

            except Exception as e:
                st.error(f"❌ Ошибка: {e}")

    # Show current stats
    st.divider()
    st.subheader("📊 Текущая статистика")

    db = st.session_state.db
    sources_data = db.get_all_sources()

    for s in sources_data:
        count = db.get_article_count_by_source(s["name"])
        st.write(f"  **{s['name']}**: {count} статей ({s['source_type']})")


def analyze_topics_for_sources(db: Database, source_names: list[str]):
    """Analyze topics for selected sources."""
    # Clear old topics and comparisons
    db.clear_topics()
    db.clear_comparisons()

    preprocessor = TextPreprocessor()
    sources = db.get_all_sources()

    for source in sources:
        if source["name"] not in source_names:
            continue

        articles = db.get_articles_by_source(source["name"])
        if not articles:
            continue

        # Preprocess
        texts = []
        for article in articles:
            content = article.get("content", "")
            title = article.get("title", "")
            combined = f"{title} {content}"
            texts.append(preprocessor.preprocess(combined))

        texts = [t for t in texts if t.strip()]

        if len(texts) < 3:
            continue

        # Topic modeling
        modeler = TopicModeler(n_topics=min(7, len(texts) // 3))
        modeler.fit(texts)
        topics = modeler.get_topics(n_words=10)

        # Save
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

    # Compare topics
    all_topics = db.get_all_topics()
    if not all_topics:
        return

    topics_by_source = {}
    for topic in all_topics:
        source_id = topic["source_id"]
        source_name = next(
            (s["name"] for s in sources if s["id"] == source_id), "unknown"
        )
        if source_name not in topics_by_source:
            topics_by_source[source_name] = []
        topics_by_source[source_name].append(topic)

    comparator = TopicComparator()
    all_comparisons = []
    source_names_list = list(topics_by_source.keys())

    for i, source_a in enumerate(source_names_list):
        for source_b in source_names_list[i + 1 :]:
            topics_a = topics_by_source[source_a]
            topics_b = topics_by_source[source_b]

            comparisons = comparator.compare_topics(
                topics_a, topics_b, source_a, source_b
            )
            all_comparisons.extend(comparisons)

    if all_comparisons:
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


def main():
    """Main Streamlit app."""
    st.set_page_config(
        page_title="Topic Matcher",
        page_icon="🔍",
        layout="wide",
    )

    init_session_state()

    # Load data
    data = load_data()

    # Sidebar navigation
    st.sidebar.title("🔍 Topic Matcher")

    pages = {
        "Обзор": render_overview,
        "Темы": render_topics,
        "Сопоставление": render_comparisons,
        "Статьи": render_articles,
        "Сбор по теме": render_collect_by_topic,
    }

    selected_page = st.sidebar.radio("Навигация", list(pages.keys()))

    # Render selected page
    pages[selected_page](data)

    # Refresh button
    if st.sidebar.button("🔄 Обновить данные"):
        st.cache_data.clear()
        st.rerun()


if __name__ == "__main__":
    main()
