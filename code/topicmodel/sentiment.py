"""
Sentiment (lexicon baseline).
=============================
VADER (Hutto & Gilbert 2014, "VADER: A Parsimonious Rule-based Model for
Sentiment Analysis of Social Media Text", ICWSM) is a lexicon-and-rules
scorer designed for social-media text.  Its `compound` score lies in [-1, 1];
we use the authors' conventional thresholds (+/-0.05) to call a post
positive / neutral / negative.  This is a first, cheap layer agreed on
2026-10-13; it is to be validated or replaced (e.g. by a transformer
classifier) once the topic approach is settled with co-authors.
"""
from __future__ import annotations
import pandas as pd
from . import config as C


def score_documents(texts: list[str]) -> pd.DataFrame:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    comp = [an.polarity_scores(t)["compound"] for t in texts]
    lab = ["positive" if c > C.SENTIMENT_POS_THRESHOLD else "negative" if c < C.SENTIMENT_NEG_THRESHOLD else "neutral" for c in comp]
    return pd.DataFrame({"sentiment_compound": comp, "sentiment_label": lab})


def by_topic(docs: pd.DataFrame) -> pd.DataFrame:
    g = docs.groupby("topic")
    out = pd.DataFrame({
        "n_docs": g.size(),
        "mean_compound": g["sentiment_compound"].mean().round(4),
        "share_positive": g["sentiment_label"].apply(lambda s: (s == "positive").mean()).round(4),
        "share_negative": g["sentiment_label"].apply(lambda s: (s == "negative").mean()).round(4),
    }).reset_index()
    return out


def by_period(docs: pd.DataFrame, run_type: str) -> pd.DataFrame:
    freq = C.TIME_BIN[run_type]
    per = docs["date"].dt.tz_localize(None).dt.to_period("W" if freq == "W" else "M").dt.start_time
    g = docs.assign(period=per).groupby(["period", "topic"])
    return pd.DataFrame({"n_docs": g.size(), "mean_compound": g["sentiment_compound"].mean().round(4),
                         "share_positive": g["sentiment_label"].apply(lambda s: (s == "positive").mean()).round(4),
                         "share_negative": g["sentiment_label"].apply(lambda s: (s == "negative").mean()).round(4)}).reset_index()
