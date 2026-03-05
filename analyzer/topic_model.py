"""
Topic modeling module using LDA.
"""

from typing import Optional

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation, NMF

import config


class TopicModeler:
    """Topic modeling using LDA or NMF."""

    def __init__(
        self,
        n_topics: Optional[int] = None,
        model_type: str = "lda",
    ):
        self.n_topics = n_topics or config.TOPIC_MODEL["n_topics"]
        self.model_type = model_type.lower()

        # Vectorizer configuration
        self.vectorizer = CountVectorizer(
            max_df=config.TOPIC_MODEL["max_document_frequency"],
            min_df=config.TOPIC_MODEL["min_document_frequency"],
            max_features=5000,
            stop_words="english",
        )

        # Model
        self.model = None
        self.feature_names = None
        self.doc_topic_matrix = None
        self.is_fitted = False
        self._last_texts = None  # Store for get_topics

    def fit(self, texts: list[str]) -> "TopicModeler":
        """Fit the topic model on texts."""
        if not texts:
            raise ValueError("Cannot fit on empty text corpus")

        # Store texts for later use in get_topics
        self._last_texts = texts

        # Vectorize
        doc_term_matrix = self.vectorizer.fit_transform(texts)
        self.feature_names = self.vectorizer.get_feature_names_out()

        # Choose model
        if self.model_type == "lda":
            self.model = LatentDirichletAllocation(
                n_components=self.n_topics,
                random_state=42,
                max_iter=20,
                learning_method="online",
                n_jobs=-1,
            )
        else:  # nmf
            self.model = NMF(
                n_components=self.n_topics,
                random_state=42,
                max_iter=200,
            )

        # Fit model
        self.doc_topic_matrix = self.model.fit_transform(doc_term_matrix)
        self.is_fitted = True

        return self

    def get_topics(self, n_words: int = 10) -> list[dict]:
        """Get top words for each topic with article counts."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        # Get dominant topic for each document
        doc_topic_matrix = self.model.transform(
            self.vectorizer.transform(self._last_texts)
        )
        dominant_topics = doc_topic_matrix.argmax(axis=1)

        topics = []

        for topic_idx, topic in enumerate(self.model.components_):
            # Get top word indices
            top_word_indices = topic.argsort()[: -n_words - 1 : -1]
            top_words = [self.feature_names[i] for i in top_word_indices]

            # Count articles belonging to this topic
            topic_article_count = int((dominant_topics == topic_idx).sum())

            topics.append(
                {
                    "topic_id": topic_idx,
                    "keywords": top_words,
                    "name": f"Topic {topic_idx}: {', '.join(top_words[:3])}",
                    "article_count": topic_article_count,
                }
            )

        return topics

    def get_document_topics(self, text: str) -> list[tuple[int, float]]:
        """Get topic distribution for a single document."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet")

        # Transform the text
        doc_vector = self.vectorizer.transform([text])
        topic_distribution = self.model.transform(doc_vector)[0]

        # Return as list of (topic_id, probability)
        return list(enumerate(topic_distribution))

    def get_dominant_topic(self, text: str) -> int:
        """Get the dominant topic for a document."""
        topics = self.get_document_topics(text)
        return max(topics, key=lambda x: x[1])[0]

    def get_topic_labels(self) -> list[str]:
        """Get human-readable topic labels."""
        topics = self.get_topics(n_words=3)
        return [f"Topic {t['topic_id']}: {', '.join(t['keywords'])}" for t in topics]


def extract_topics(
    texts: list[str],
    n_topics: int = 7,
    model_type: str = "lda",
) -> list[dict]:
    """Convenience function to extract topics from texts.

    Args:
        texts: List of text documents
        n_topics: Number of topics to extract
        model_type: 'lda' or 'nmf'

    Returns:
        List of topic dictionaries with keywords
    """
    modeler = TopicModeler(n_topics=n_topics, model_type=model_type)
    modeler.fit(texts)
    return modeler.get_topics()


if __name__ == "__main__":
    # Test topic modeling
    sample_texts = [
        "Machine learning algorithms process data to make predictions",
        "Deep learning neural networks train on large datasets",
        "Natural language processing analyzes text and speech",
        "Computer vision recognizes objects in images",
        "Reinforcement learning trains agents through rewards",
        "Data science uses statistics and programming to analyze data",
        "Artificial intelligence enables machines to think",
    ]

    topics = extract_topics(sample_texts, n_topics=3)

    print("Extracted topics:")
    for topic in topics:
        print(f"\n{topic['name']}")
        print(f"  Keywords: {', '.join(topic['keywords'])}")
