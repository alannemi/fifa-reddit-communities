"""
Configuration for the topic-modelling pipeline.
=================================================
Every tunable setting lives here so that (a) a run is fully described by this
file plus the command line, (b) each run writes its effective settings to
results/.../config.json, and (c) co-authors can review or change a setting
without reading the pipeline code.

Settings were agreed with A. Nemirovski on 2026-10-13 (see
docs/session-02_2026-10-13/02_topic-modelling-design.md for the plain-language
explanation of each one).  The section "Analytical pipeline" at the end holds
the settings of the period-bin pipeline designed in session 04
(docs/session-04_2026-10-22/01_analytical-pipeline-design.md, approved
2026-10-22) and implemented in session 05 (code/06-analysis-pipeline.py);
the approvals of 2026-10-23 are in docs/session-05_2026-10-23/00_approvals.md and those of
2026-10-24 (two-step taxonomy protocol D5, prompt v1.2) in docs/session-06_2026-10-24/00_approvals.md.
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
    "vs", "soccer",   # added 2026-10-22 (design F11): scaffolding seen in the 2010 keyword lists
]

# ---------------------------------------------------------------------------
# Topics over time
# ---------------------------------------------------------------------------
TIME_BIN = {"year": "M", "era": "M", "window": "W"}   # monthly for year/era runs, weekly for tournament windows
ERA_MERGE_MIN_SIMILARITY = 0.85

# ---------------------------------------------------------------------------
# LLM labelling (two-analyst debate, as in the template paper)
# ---------------------------------------------------------------------------
LLM_MODEL = "Qwen/Qwen3-1.7B"          # the brief's model (design §7.1, 2026-10-22); Apache-2.0; revision pinned below
LLM_MODEL_LARGE = "Qwen/Qwen3-4B"      # NOT USED: the optional 4B comparison (design F9/§12.4) was dropped by the author on
                                       # 2026-10-24 — the pipeline stays with Qwen3-1.7B; kept only so --llm can name it
LLM_MODEL_ALT = "microsoft/Phi-4-mini-instruct"   # optional comparison (MIT licence)
LLM_MAX_NEW_TOKENS = 500               # used by the exploratory tool (05); the bin pipeline uses the *_LABEL/_TAXONOMY values
LLM_TEMPERATURE = 0                    # greedy decoding for reproducibility (design F12; was 0.2 in session 02)
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


# ===========================================================================
# Analytical pipeline (period bins → BERTopic per bin → two agents → VADER → tagging)
# Design: docs/session-04_2026-10-22/01_analytical-pipeline-design.md (v1.0, approved 2026-10-22)
# Approvals/refinements: docs/session-05_2026-10-23/00_approvals.md
# Entry point: code/06-analysis-pipeline.py (stages bins, embed, topics, reassign, label, taxonomy,
# sentiment, collect, report).  Everything below is written to each bin's config.json.
# ===========================================================================
ANALYSIS_DIR = ROOT / "results" / "analysis"          # results/analysis/bin-N/ + aggregated CSVs
ANALYSIS_DATA_DIR = DATA_DIR / "analysis"             # join tables by post id (period_bins, sentiment, post_variables)

# --- S0: period bins (design §4; the brief's nine batches). The UTC calendar date of created_datetime decides.
PERIOD_BINS = [
    dict(bin=1, start=dt.date(2010, 1, 1),  end=dt.date(2010, 9, 30), period_type="tournament", world_cup=2010),
    dict(bin=2, start=dt.date(2010, 10, 1), end=dt.date(2014, 1, 31), period_type="interim",    world_cup=None),
    dict(bin=3, start=dt.date(2014, 2, 1),  end=dt.date(2014, 9, 30), period_type="tournament", world_cup=2014),
    dict(bin=4, start=dt.date(2014, 10, 1), end=dt.date(2018, 1, 31), period_type="interim",    world_cup=None),
    dict(bin=5, start=dt.date(2018, 2, 1),  end=dt.date(2018, 9, 30), period_type="tournament", world_cup=2018),
    dict(bin=6, start=dt.date(2018, 10, 1), end=dt.date(2022, 6, 30), period_type="interim",    world_cup=None),
    dict(bin=7, start=dt.date(2022, 7, 1),  end=dt.date(2023, 2, 28), period_type="tournament", world_cup=2022),
    dict(bin=8, start=dt.date(2023, 3, 1),  end=dt.date(2026, 1, 31), period_type="interim",    world_cup=None),
    dict(bin=9, start=dt.date(2026, 2, 1),  end=dt.date(2026, 7, 31), period_type="tournament", world_cup=2026),
]
# Bin sizes computed from the v2.1 yearly files on 2026-10-22 (design §4); S0 compares and logs any difference.
PERIOD_BIN_EXPECTED_N = {1: 10002, 2: 186988, 3: 83147, 4: 331216, 5: 88648, 6: 305653, 7: 41438, 8: 217303, 9: 37626}
BIN_TIME_BIN = {"tournament": "W", "interim": "M"}   # topics-over-time resolution per period type

# --- S1: embeddings (design §5) — mpnet on the GPU; the cache in EMBED_CACHE_DIR is keyed by model name.
ANALYSIS_EMBEDDING_MODEL = EMBEDDING_MODEL_HQ

# --- S2: BERTopic per bin (design §6). UMAP_PARAMS, HDBSCAN_MIN_SAMPLES, NGRAM_RANGE, TOP_N_WORDS, MMR_DIVERSITY,
#     N_REPRESENTATIVE_DOCS and EXTRA_STOPWORDS above are shared with the exploratory tool.
BIN_MIN_CLUSTER_FRACTION = 0.0025      # starting min_cluster_size = 0.25 % of the bin's modelled posts ...
BIN_MIN_CLUSTER_FLOOR = 25             # ... but never fewer than 25 (05-topic-model.py keeps its floor of 15)
TOPIC_COUNT_RANGE = (50, 100)          # topic-count guard target (raw topics per bin)
GUARD_MAX_TRIALS = 4                   # at most four HDBSCAN fits on the frozen 5-d array
GUARD_STEP = 1.5                       # first move ×1.5 / ÷1.5, then bisection between bracketing values
OUTLIER_STRATEGY = "embeddings"        # decision D3: cosine similarity to the topic embedding, thresholded
OUTLIER_THRESHOLD: dict = {}           # {"default": T} or per bin {1: T1, 4: T4}; empty = not calibrated -> no reassignment yet
OUTLIER_CALIBRATION_QUANTILES = (0.50, 0.65, 0.80)   # candidate thresholds = these quantiles of outlier similarity
OUTLIER_SAMPLE_PER_CANDIDATE = 100     # posts per candidate threshold for the co-author judgement sample

# --- S3: two-agent labelling (design §7). LLM_MODEL, LLM_TEMPERATURE, LLM_SAMPLE_POSTS, LLM_AGENTS, TONE_SET above.
LLM_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"   # Hugging Face snapshot of Qwen/Qwen3-1.7B used throughout
LLM_SAMPLING_SEEDED = False            # True = seeded sampling (comparison on bin 1, F17); False = greedy
LLM_SAMPLING_TEMPERATURE = 0.7
LLM_BATCH_SIZE = {"cuda": 16, "mps": 4, "cpu": 1}   # prompts generated together (greedy mode only)
MAX_ROUNDS = 5                         # decision D2: cap on debate rounds
LLM_MAX_NEW_TOKENS_LABEL = 600         # S3a answer
LLM_MAX_NEW_TOKENS_TAXONOMY = 8000     # S3b answer (design said 4,000; raised 2026-10-23, review finding R2)
LLM_POST_MAX_CHARS = 300               # each sample post shown to the analysts is truncated to this length
TAXONOMY_KEYWORDS_PER_TOPIC = 8        # keywords per topic line in the S3b items
# S3b protocol (decision D5, 2026-10-24; dry-run finding D2: Qwen3-1.7B cannot partition 92 topics in one prompt):
#   "two_step"  = the analysts first agree on the bin's category definitions (no topic ids), then assign the topics to
#                 those categories, one debate item per TAXONOMY_ASSIGN_BATCH topics (1 = one debate per topic);
#   "partition" = the v1.1 single-prompt protocol (categories with explicit `topics` lists; chunking fallback above
#                 TAXONOMY_MAX_ITEMS), kept for comparison and replay of the session-05 dry run.
TAXONOMY_PROTOCOL = "two_step"
TAXONOMY_ASSIGN_BATCH = 1              # topics per assignment prompt (author's decision 2026-10-24: one topic per item)
LLM_MAX_NEW_TOKENS_DEFINE = 4000       # step-1 answer (10-20 categories x ~100 tokens, with margin)
LLM_MAX_NEW_TOKENS_ASSIGN = 800        # step-2 answer (one {topic_id, category_id} pair per topic in the item)
TAXONOMY_MAX_ITEMS = 100               # partition protocol only: above this many topics S3b uses the chunking fallback (design §7.4)
TAXONOMY_CHUNK_SIZE = 80               # partition protocol only
LLM_JSON_SALVAGE = True                # recover answers whose only defect is bracket syntax (N11, 2026-10-23 dry run);
                                       # strict JSON is tried first; recovered answers are marked parsed_via="salvage"
                                       # in the transcript. False = design §7.1 literally (second failure = no answer).

# --- S4: sentiment (design §8, F18). SENTIMENT_POS/NEG_THRESHOLD above.
SENTIMENT_TEXT = "title"               # post_sentiment is scored on the cleaned title; post_sentiment_full on title + body
SENTIMENT_VALIDATION_SAMPLE = 300      # hand-coding sample (stratified by bin × label × is_self)
SENTIMENT_DECIMALS = 4
