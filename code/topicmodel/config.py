"""
Configuration for the topic-modelling pipeline.
=================================================
Every tunable setting lives here so that (a) a run is fully described by this
file plus the command line, (b) each run writes its effective settings to
results/.../config.json, and (c) co-authors can review or change a setting
without reading the pipeline code.

Settings were agreed with A. Nemirovski on 2026-10-13 (see
docs/session-02_2026-10-13/02_topic-modelling-design.md for the plain-language
explanation of each one).
"""
from __future__ import annotations
import datetime as dt
import json
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths (relative to the repository root = two levels above this file)
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
YEARLY_DIR = DATA_DIR / "yearly datasets"
RESULTS_DIR = ROOT / "results" / "topic-models"
EMBED_CACHE_DIR = RESULTS_DIR / "embeddings"      # <model-slug>/<year>.npz
LLM_CACHE_DIR = ROOT / ".hf-cache"                # Hugging Face weights (gitignored / machine-local)

# ---------------------------------------------------------------------------
# Documents
# ---------------------------------------------------------------------------
BODY_MAX_CHARS = 1000      # body (selftext) is truncated to this many characters
MIN_TOKENS = 3             # documents with fewer word tokens are flagged `too_short` and not modelled
COLUMNS_KEPT = ["id", "created_datetime", "year", "post_type", "lang", "author",
                "link_flair_text", "num_comments", "score", "domain", "is_self"]

# ---------------------------------------------------------------------------
# World Cups, windows and eras
# ---------------------------------------------------------------------------
# Opening match and final of each tournament (UTC dates).
WORLD_CUPS = {
    2010: (dt.date(2010, 6, 11), dt.date(2010, 7, 11)),   # South Africa
    2014: (dt.date(2014, 6, 12), dt.date(2014, 7, 13)),   # Brazil
    2018: (dt.date(2018, 6, 14), dt.date(2018, 7, 15)),   # Russia
    2022: (dt.date(2022, 11, 20), dt.date(2022, 12, 18)), # Qatar
    2026: (dt.date(2026, 6, 11), dt.date(2026, 7, 19)),   # Canada/Mexico/USA
}
WINDOW_MONTHS = 2          # tournament window = kickoff - 2 months .. final + 2 months (confirmed 2026-10-13)
DATA_END = dt.date(2026, 7, 31)   # last day covered by the corpus

# Year models are grouped (aligned) within these four-year World Cup eras.
ERAS = {
    2010: [2010, 2011, 2012, 2013],
    2014: [2014, 2015, 2016, 2017],
    2018: [2018, 2019, 2020, 2021],
    2022: [2022, 2023, 2024, 2025],
    2026: [2026],
}


def _shift_months(d: dt.date, months: int) -> dt.date:
    """Shift a date by whole months, clamping the day to the target month's length."""
    y, m = d.year, d.month + months
    while m > 12: y, m = y + 1, m - 12
    while m < 1: y, m = y - 1, m + 12
    import calendar
    return dt.date(y, m, min(d.day, calendar.monthrange(y, m)[1]))


def world_cup_window(year: int) -> tuple[dt.date, dt.date]:
    """Return (start, end) of the tournament window, clamped to the data end."""
    kick, final = WORLD_CUPS[year]
    return _shift_months(kick, -WINDOW_MONTHS), min(_shift_months(final, WINDOW_MONTHS), DATA_END)

# ---------------------------------------------------------------------------
# Embeddings
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"   # fast; good for headline-length text
EMBEDDING_MODEL_HQ = "sentence-transformers/all-mpnet-base-v2"  # higher quality; use on a GPU (Colab)
EMBEDDING_BATCH_SIZE = 256
EMBEDDING_MAX_SEQ_LENGTH = 256   # tokens; title + 1,000 chars of body fits comfortably

# ---------------------------------------------------------------------------
# BERTopic
# ---------------------------------------------------------------------------
SEED = 42
UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
HDBSCAN_MIN_CLUSTER_FRACTION = 0.0025   # min topic size = 0.25 % of modelled documents ...
HDBSCAN_MIN_CLUSTER_FLOOR = 15          # ... but never fewer than 15 posts
HDBSCAN_MIN_SAMPLES = 10
NGRAM_RANGE = (1, 2)
TOP_N_WORDS = 15
MMR_DIVERSITY = 0.3
MAX_TOPICS_BEFORE_REDUCE = 60           # above this, an additional reduced model (nr_topics="auto") is also saved
REDUCE_OUTLIERS = True                  # reassign outliers with the c-TF-IDF strategy (before/after share reported)
N_REPRESENTATIVE_DOCS = 5

# Reddit / r/soccer scaffolding terms removed ONLY from the keyword descriptions
# (never from the documents or the embeddings). No football vocabulary is removed.
EXTRA_STOPWORDS = [
    "thread", "match thread", "post match", "pre match", "discussion", "daily discussion",
    "amp", "http", "https", "www", "com", "html", "reddit", "redd", "imgur", "youtube", "youtu",
    "twitter", "streamable", "gfycat", "edit", "deleted", "removed",
]

# ---------------------------------------------------------------------------
# Topics over time
# ---------------------------------------------------------------------------
TIME_BIN = {"year": "M", "era": "M", "window": "W"}   # monthly for year/era runs, weekly for tournament windows
ERA_MERGE_MIN_SIMILARITY = 0.85

# ---------------------------------------------------------------------------
# LLM labelling (two-analyst debate, as in the template paper)
# ---------------------------------------------------------------------------
LLM_MODEL = "Qwen/Qwen3-4B"            # recommended (Apache-2.0, same family as the template paper)
LLM_MODEL_FAST = "Qwen/Qwen3-1.7B"     # fallback for speed
LLM_MODEL_ALT = "microsoft/Phi-4-mini-instruct"   # optional comparison (MIT licence)
LLM_MAX_NEW_TOKENS = 500
LLM_TEMPERATURE = 0.2
LLM_SAMPLE_POSTS = 40                  # posts shown to the analysts per topic
LLM_MAX_TURNS = 3                      # initial answer + up to 2 revision turns
LLM_AGENTS = ("Analyst A", "Analyst B")
# Free-text tones from the LLM are also mapped to this controlled set for counting.
TONE_SET = ["positive", "negative", "neutral", "mixed", "excited", "enthusiastic", "frustrated",
            "angry", "anxious", "sad", "humorous", "critical", "supportive", "curious", "cautious"]

# ---------------------------------------------------------------------------
# Sentiment (lexicon baseline; to be validated/replaced after co-author review)
# ---------------------------------------------------------------------------
SENTIMENT_POS_THRESHOLD = 0.05
SENTIMENT_NEG_THRESHOLD = -0.05


def effective_settings(**overrides) -> dict:
    """All settings as a JSON-serialisable dict (written to each run's config.json)."""
    d = {k: v for k, v in globals().items()
         if k.isupper() and not k.startswith("_") and isinstance(v, (str, int, float, list, dict, tuple, type(None)))}
    d = json.loads(json.dumps(d, default=str))
    d.update(overrides)
    return d
