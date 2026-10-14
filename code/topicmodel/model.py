"""
BERTopic construction, fitting and post-processing.
====================================================
Pipeline inside BERTopic (all settings from config.py):
  embeddings (pre-computed) -> UMAP (5 dims) -> HDBSCAN (density clusters)
  -> CountVectorizer + c-TF-IDF (distinctive words per topic)
  -> KeyBERTInspired + MaximalMarginalRelevance (readable, diverse keywords)
Afterwards: optional outlier reduction, topics over time, optional reduced
model when there are many topics, and era alignment via merge_models.
"""
from __future__ import annotations
import logging
import numpy as np
import pandas as pd
from . import config as C

log = logging.getLogger("topicmodel")


def min_cluster_size(n_docs: int) -> int:
    return max(C.HDBSCAN_MIN_CLUSTER_FLOOR, round(n_docs * C.HDBSCAN_MIN_CLUSTER_FRACTION))


def build_topic_model(n_docs: int, embedding_model_name: str, device: str):
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from hdbscan import HDBSCAN
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from umap import UMAP
    from .embed import load_embedding_model

    umap_model = UMAP(**C.UMAP_PARAMS)
    hdbscan_model = HDBSCAN(min_cluster_size=min_cluster_size(n_docs), min_samples=C.HDBSCAN_MIN_SAMPLES,
                            metric="euclidean", cluster_selection_method="eom", prediction_data=True)
    stop = list(ENGLISH_STOP_WORDS) + C.EXTRA_STOPWORDS
    vectorizer = CountVectorizer(ngram_range=C.NGRAM_RANGE, stop_words=stop, min_df=2)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    # A chain: KeyBERT-style re-ranking of candidate words by similarity to the
    # topic, then MMR to penalise near-duplicate keywords.
    representation = [KeyBERTInspired(top_n_words=C.TOP_N_WORDS * 2),
                      MaximalMarginalRelevance(diversity=C.MMR_DIVERSITY, top_n_words=C.TOP_N_WORDS)]
    st_model = load_embedding_model(embedding_model_name, device)   # needed by KeyBERTInspired
    model = BERTopic(embedding_model=st_model, umap_model=umap_model, hdbscan_model=hdbscan_model,
                     vectorizer_model=vectorizer, ctfidf_model=ctfidf, representation_model=representation,
                     top_n_words=C.TOP_N_WORDS, calculate_probabilities=False, verbose=True)
    log.info("BERTopic built: min_cluster_size=%d, n_docs=%d", min_cluster_size(n_docs), n_docs)
    return model


def fit_model(model, docs: list[str], embeddings: np.ndarray) -> tuple[list[int], dict]:
    """Fit and return (topics per doc, stats)."""
    topics, _ = model.fit_transform(docs, embeddings)
    topics = np.asarray(topics)
    stats = {"n_docs": len(docs), "n_topics_raw": int((model.get_topic_info()["Topic"] != -1).sum()),
             "outlier_share_raw": float((topics == -1).mean())}
    if C.REDUCE_OUTLIERS and (topics == -1).any():
        # Reassign outliers to the closest topic by c-TF-IDF similarity (documented strategy).
        new_topics = model.reduce_outliers(docs, topics.tolist(), strategy="c-tf-idf")
        model.update_topics(docs, topics=new_topics, vectorizer_model=model.vectorizer_model,
                            ctfidf_model=model.ctfidf_model, representation_model=model.representation_model)
        topics = np.asarray(new_topics)
    stats["outlier_share_final"] = float((topics == -1).mean())
    stats["n_topics_final"] = int((model.get_topic_info()["Topic"] != -1).sum())
    log.info("fit: %s", stats)
    return topics.tolist(), stats


def topic_table(model, docs: pd.DataFrame, topics: list[int]) -> pd.DataFrame:
    """One row per topic: size, share, keywords, representative documents."""
    info = model.get_topic_info()
    rows = []
    d = docs.assign(topic=topics)
    for _, r in info.iterrows():
        t = int(r["Topic"])
        words = [w for w, _ in model.get_topic(t)] if t != -1 else []
        sub = d[d["topic"] == t]
        # representative = documents closest to the topic's c-TF-IDF representation
        rep_ids, rep_titles = [], []
        if t != -1:
            reps = model.get_representative_docs(t) or []
            for text in reps[: C.N_REPRESENTATIVE_DOCS]:
                m = sub[sub["document"] == text]
                if len(m):
                    rep_ids.append(m.iloc[0]["id"]); rep_titles.append(m.iloc[0]["title"][:140])
        rows.append({"topic": t, "size": int(len(sub)), "share": round(len(sub) / len(d), 5),
                     "keywords": ", ".join(words), "representative_ids": "|".join(rep_ids),
                     "representative_titles": " || ".join(rep_titles),
                     "share_match_thread": round(float((sub["post_type"] == "match_thread").mean()), 4) if len(sub) else 0.0,
                     "mean_comments": round(float(pd.to_numeric(sub["num_comments"], errors="coerce").fillna(0).mean()), 2) if len(sub) else 0.0})
    return pd.DataFrame(rows)


def topics_over_time(model, docs: pd.DataFrame, topics: list[int], run_type: str) -> pd.DataFrame:
    """Topic frequency per time bin (month for year/era, week for windows)."""
    freq = C.TIME_BIN[run_type]
    stamps = docs["date"].dt.tz_localize(None).dt.to_period("W" if freq == "W" else "M").dt.start_time
    tot = model.topics_over_time(docs["document"].tolist(), stamps.tolist(), topics=topics, global_tuning=False, evolution_tuning=False)
    tot = tot.rename(columns={"Topic": "topic", "Words": "keywords", "Frequency": "n_docs", "Timestamp": "period"})
    return tot[["topic", "period", "n_docs", "keywords"]]


def save_model(model, path, embedding_model_name: str):
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path), serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model_name)


def load_model(path):
    from bertopic import BERTopic
    return BERTopic.load(str(path))


def merge_year_models(models: list, min_similarity: float = C.ERA_MERGE_MIN_SIMILARITY):
    """Align several year models into one era model (topics sharing >= min_similarity are unified)."""
    from bertopic import BERTopic
    return BERTopic.merge_models(models, min_similarity=min_similarity)
