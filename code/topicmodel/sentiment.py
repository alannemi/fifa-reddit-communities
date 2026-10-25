"""
Stage S4 — VADER sentiment per post (design §8; decision F18).
==============================================================
VADER (Hutto & Gilbert 2014) is the lexicon-and-rules scorer the brief specifies; its `compound` score lies in
[-1, 1].  Per post:
  post_sentiment        compound of the cleaned TITLE — the unit every post has, comparable across bins and post types
  post_sentiment_full   compound of title + body (body <= BODY_MAX_CHARS) — the second measure, for self posts
  post_sentiment_label  positive above SENTIMENT_POS_THRESHOLD, negative below SENTIMENT_NEG_THRESHOLD, else neutral
Scored per calendar year and cached in <analysis data dir>/sentiment/<year>.csv (resumable), then merged into
sentiment.csv; the 300-post hand-coding sample is written alongside (refinement N9).  Known limits (design F10):
headline-length text, sarcasm and football vocabulary that a general lexicon mis-scores; a transformer-based measure
can be added later as a further column without changing anything else.
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C
from . import data

log = logging.getLogger("topicmodel")


def label_of(compound: float) -> str:
    """The single definition of post_sentiment_label (the S5 checks use it too)."""
    return "positive" if compound > C.SENTIMENT_POS_THRESHOLD else "negative" if compound < C.SENTIMENT_NEG_THRESHOLD else "neutral"


def score_year(year: int, cache_dir: Path) -> pd.DataFrame:
    """id, post_sentiment, post_sentiment_full, post_sentiment_label for one yearly file (cached)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    p = cache_dir / f"{year}.csv"
    if p.exists():
        return pd.read_csv(p, dtype={"id": str, "post_sentiment_label": str}, keep_default_na=False)
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    an = SentimentIntensityAnalyzer()
    df = data.load_year(year)
    title = df["title"].map(data.clean_text)
    body = df["selftext"].str.slice(0, C.BODY_MAX_CHARS).map(data.clean_text)
    comp_t = np.array([an.polarity_scores(t)["compound"] for t in title])
    comp_f = comp_t.copy()
    has_body = body.str.strip() != ""
    full = (title[has_body] + "\n" + body[has_body]).str.strip()
    comp_f[has_body.to_numpy()] = [an.polarity_scores(t)["compound"] for t in full]
    out = pd.DataFrame({"id": df["id"], "post_sentiment": comp_t.round(C.SENTIMENT_DECIMALS),
                        "post_sentiment_full": comp_f.round(C.SENTIMENT_DECIMALS)})
    out["post_sentiment_label"] = out["post_sentiment"].map(label_of)
    out.to_csv(p, index=False, float_format=f"%.{C.SENTIMENT_DECIMALS}f")
    log.info("sentiment: %d scored %d posts (%d with body) -> %s", year, len(out), int(has_body.sum()), p)
    return out


def run_sentiment(years: list[int], analysis_data_dir: Path, period_bins: pd.DataFrame | None) -> pd.DataFrame:
    """Score the given years, merge to sentiment.csv and write the validation sample."""
    cache = analysis_data_dir / "sentiment"
    parts = [score_year(y, cache) for y in years]
    sent = pd.concat(parts, ignore_index=True)
    sent.to_csv(analysis_data_dir / "sentiment.csv", index=False, float_format=f"%.{C.SENTIMENT_DECIMALS}f")
    log.info("sentiment: %d posts -> %s", len(sent), analysis_data_dir / "sentiment.csv")
    if period_bins is not None:
        validation_sample(sent, period_bins, years, analysis_data_dir / "sentiment_validation_sample.csv")
    return sent


def validation_sample(sent: pd.DataFrame, period_bins: pd.DataFrame, years: list[int], path: Path,
                      n: int = C.SENTIMENT_VALIDATION_SAMPLE) -> pd.DataFrame:
    """Stratified by period_bin × label × is_self, proportional to stratum size with at least two posts per
    non-empty stratum; titles are read back from the yearly files; coder columns are blank (design §8)."""
    meta = []
    for y in years:
        df = pd.read_csv(data.yearly_file(y), dtype=str, keep_default_na=False, usecols=["id", "title", "is_self"])
        meta.append(df)
    meta = pd.concat(meta, ignore_index=True)
    d = sent.merge(meta, on="id", how="inner").merge(period_bins[["id", "period_bin"]], on="id", how="inner")
    rng = np.random.RandomState(C.SEED)
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
    log.info("sentiment: validation sample of %d posts (%d strata) -> %s", len(out), len(sizes), path)
    return out
