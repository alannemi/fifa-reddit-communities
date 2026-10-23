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


# ===========================================================================
# Stage S4 of the analytical pipeline (session 05; design §8, decision F18):
#   post_sentiment       VADER compound of the cleaned TITLE (the unit every post has)
#   post_sentiment_full  VADER compound of title + body (<= BODY_MAX_CHARS)
#   post_sentiment_label positive > 0.05, negative < -0.05, else neutral (on post_sentiment)
# Scored per calendar year and cached in data/analysis/sentiment/<year>.csv (resumable); merged into
# data/analysis/sentiment.csv.  A 300-post hand-coding sample is written alongside (refinement N9).
# ===========================================================================
import logging as _logging
from pathlib import Path as _Path
import numpy as _np
from . import data as _data

_log = _logging.getLogger("topicmodel")


def label_of(compound: float) -> str:
    return "positive" if compound > C.SENTIMENT_POS_THRESHOLD else "negative" if compound < C.SENTIMENT_NEG_THRESHOLD else "neutral"


def score_year(year: int, cache_dir: _Path) -> pd.DataFrame:
    """id, post_sentiment, post_sentiment_full, post_sentiment_label for one yearly file (cached)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{year}.csv"
    if p.exists():
        return pd.read_csv(p, dtype={"id": str, "post_sentiment_label": str}, keep_default_na=False)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    df = _data.load_year(year)
    title = df["title"].map(_data.clean_text)
    body = df["selftext"].str.slice(0, C.BODY_MAX_CHARS).map(_data.clean_text)
    comp_t = _np.array([an.polarity_scores(t)["compound"] for t in title])
    comp_f = comp_t.copy()
    has_body = body.str.strip() != ""
    full = (title[has_body] + "\n" + body[has_body]).str.strip()
    comp_f[has_body.to_numpy()] = [an.polarity_scores(t)["compound"] for t in full]
    out = pd.DataFrame({"id": df["id"], "post_sentiment": comp_t.round(C.SENTIMENT_DECIMALS),
                        "post_sentiment_full": comp_f.round(C.SENTIMENT_DECIMALS)})
    out["post_sentiment_label"] = out["post_sentiment"].map(label_of)
    out.to_csv(p, index=False, float_format=f"%.{C.SENTIMENT_DECIMALS}f")
    _log.info("sentiment: %d scored %d posts (%d with body) -> %s", year, len(out), int(has_body.sum()), p)
    return out


def run_sentiment(years: list[int], analysis_data_dir: _Path, period_bins: pd.DataFrame | None) -> pd.DataFrame:
    """Score the given years (default all), merge to sentiment.csv and write the validation sample."""
    cache = analysis_data_dir / "sentiment"
    parts = [score_year(y, cache) for y in years]
    sent = pd.concat(parts, ignore_index=True)
    sent.to_csv(analysis_data_dir / "sentiment.csv", index=False, float_format=f"%.{C.SENTIMENT_DECIMALS}f")
    _log.info("sentiment: %d posts -> %s", len(sent), analysis_data_dir / "sentiment.csv")
    if period_bins is not None:
        validation_sample(sent, period_bins, years, analysis_data_dir / "sentiment_validation_sample.csv")
    return sent


def validation_sample(sent: pd.DataFrame, period_bins: pd.DataFrame, years: list[int], path: _Path,
                      n: int = C.SENTIMENT_VALIDATION_SAMPLE) -> pd.DataFrame:
    """Stratified by period_bin × label × is_self, proportional to stratum size with at least two posts per
    non-empty stratum; titles are read back from the yearly files; coder columns are blank (design §8)."""
    meta = []
    for y in years:
        df = pd.read_csv(C.YEARLY_DIR / f"soccer.data.cleaned-{y}.csv", dtype=str, keep_default_na=False,
                         usecols=["id", "title", "is_self"])
        meta.append(df)
    meta = pd.concat(meta, ignore_index=True)
    d = sent.merge(meta, on="id", how="inner").merge(period_bins[["id", "period_bin"]], on="id", how="inner")
    rng = _np.random.RandomState(C.SEED)
    strata = d.groupby(["period_bin", "post_sentiment_label", "is_self"], sort=True)
    sizes = strata.size()
    alloc = (sizes / sizes.sum() * n).round().astype(int).clip(lower=2)
    picks = []
    for key, grp in strata:
        k = min(int(alloc[key]), len(grp))
        picks.append(grp.sample(k, random_state=rng))
    samp = pd.concat(picks).sample(frac=1, random_state=C.SEED)
    out = samp[["id", "period_bin", "is_self", "title", "post_sentiment", "post_sentiment_full", "post_sentiment_label"]].copy()
    out["coder_1"] = ""; out["coder_2"] = ""; out["notes"] = ""
    out.to_csv(path, index=False)
    _log.info("sentiment: validation sample of %d posts (%d strata) -> %s", len(out), len(sizes), path)
    return out
