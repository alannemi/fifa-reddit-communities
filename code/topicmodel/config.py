"""
Settings of the analytical pipeline — period bins → BERTopic per bin → two Qwen3-1.7B analysts → VADER → tagging.
==================================================================================================================
Every tunable setting lives here so that (a) a run is fully described by this file plus the command line, (b) every
bin writes its effective settings to its config.json (`effective_settings()`), and (c) co-authors can review or change
a setting without reading the pipeline code.  Nothing below is read from the environment except the path root.

Design: docs/session-04_2026-10-22/01_analytical-pipeline-design.md (v1.2; decisions D1–D5 in §14, flags in §13).
Approvals: docs/session-05_2026-10-23/00_approvals.md (N1–N12, R1–R13), docs/session-06_2026-10-24/00_approvals.md
(D5 two-step taxonomy, prompt v1.2), docs/session-07_2026-10-25/00_approvals.md (one pipeline, PIPELINE_ROOT).
Entry point: code/05-analysis-pipeline.py (stages bins, embed, topics, reassign, label, taxonomy, sentiment, collect,
report).  Stage S6 (write-back into the data files) is code/06-tag-corpus.R + code/07-validate-analysis.R.
"""
from __future__ import annotations
import datetime as dt
import json
import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths — one root setting (session 07, decisions H2/H3)
# ---------------------------------------------------------------------------
# Unset: the repository layout (local work).  PIPELINE_ROOT=<folder>: the flat layout of the shared Google Drive
# folder MyDrive/fifa-communities/, used by the Colab notebook (code/05b-analysis-pipeline-colab.ipynb).
# PIPELINE_YEARLY_DIR overrides the input folder alone (on Colab the 17 CSVs are copied to the machine's local disk
# for speed); PIPELINE_HF_CACHE_DIR overrides the model-weight cache alone (on Colab the weights must be on the local
# disk: safetensors memory-maps them and Drive's FUSE mount returns garbage — "header too large" — for such reads).
# The stage program's --out-root option redirects the two OUTPUT folders for dry runs
# (repository-shaped: <out>/results/analysis and <out>/data/analysis) and wins over PIPELINE_ROOT for those two.
#
#   setting            repository layout (unset)    PIPELINE_ROOT=<R> (flat, Drive)                    holds
#   YEARLY_DIR         data/yearly datasets         <R>/yearly datasets  (or PIPELINE_YEARLY_DIR)       the 17 input CSVs
#   ANALYSIS_DATA_DIR  data/analysis                <R>/analysis                                       join tables by post id
#   ANALYSIS_DIR       results/analysis             <R>/results                                        bin-N/ + aggregated tables
#   EMBED_CACHE_DIR    results/embeddings           <R>/embeddings                                     <model-slug>/<year>.npz
#   LLM_CACHE_DIR      .hf-cache                    <R>/hf-cache  (or PIPELINE_HF_CACHE_DIR)             Hugging Face weights
ROOT = Path(__file__).resolve().parents[2]        # the repository root (two levels above this file)
YEARLY_FILE = "soccer.data.cleaned-{year}.csv"    # one cleaned file per calendar year (codebook §1)


def resolve_paths(pipeline_root: str | None, yearly_dir: str | None = None, hf_cache_dir: str | None = None) -> dict[str, Path]:
    """The five persisted paths for a root setting (pure function; the module constants below are one call of it)."""
    if pipeline_root:
        r = Path(pipeline_root).expanduser().resolve()
        p = {"YEARLY_DIR": r / "yearly datasets", "ANALYSIS_DATA_DIR": r / "analysis", "ANALYSIS_DIR": r / "results",
             "EMBED_CACHE_DIR": r / "embeddings", "LLM_CACHE_DIR": r / "hf-cache"}
    else:
        p = {"YEARLY_DIR": ROOT / "data" / "yearly datasets", "ANALYSIS_DATA_DIR": ROOT / "data" / "analysis",
             "ANALYSIS_DIR": ROOT / "results" / "analysis", "EMBED_CACHE_DIR": ROOT / "results" / "embeddings",
             "LLM_CACHE_DIR": ROOT / ".hf-cache"}
    if yearly_dir:
        p["YEARLY_DIR"] = Path(yearly_dir).expanduser().resolve()
    if hf_cache_dir:
        p["LLM_CACHE_DIR"] = Path(hf_cache_dir).expanduser().resolve()
    return p


PIPELINE_ROOT = os.environ.get("PIPELINE_ROOT") or None
PIPELINE_YEARLY_DIR = os.environ.get("PIPELINE_YEARLY_DIR") or None
PIPELINE_HF_CACHE_DIR = os.environ.get("PIPELINE_HF_CACHE_DIR") or None
_paths = resolve_paths(PIPELINE_ROOT, PIPELINE_YEARLY_DIR, PIPELINE_HF_CACHE_DIR)
YEARLY_DIR = _paths["YEARLY_DIR"]
ANALYSIS_DATA_DIR = _paths["ANALYSIS_DATA_DIR"]
ANALYSIS_DIR = _paths["ANALYSIS_DIR"]
EMBED_CACHE_DIR = _paths["EMBED_CACHE_DIR"]
LLM_CACHE_DIR = _paths["LLM_CACHE_DIR"]

# ---------------------------------------------------------------------------
# Documents (design §2): one post = one document
# ---------------------------------------------------------------------------
BODY_MAX_CHARS = 1000      # the body (selftext) is truncated to this many characters
MIN_TOKENS = 3             # posts with fewer word tokens in title + body are kept in every table but not modelled
COLUMNS_KEPT = ["id", "created_datetime", "year", "post_type", "lang", "author",
                "link_flair_text", "num_comments", "score", "domain", "is_self"]

# ---------------------------------------------------------------------------
# S0 — period bins (design §4; the brief's nine batches; boundaries approved as F6 on 2026-10-23)
# ---------------------------------------------------------------------------
# The UTC calendar date of created_datetime decides the bin; the bins are contiguous and cover the whole corpus.
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
# Kick-off and final of each tournament (UTC dates): the markers drawn in the reports of the tournament bins.
WORLD_CUPS = {
    2010: (dt.date(2010, 6, 11), dt.date(2010, 7, 11)),   # South Africa
    2014: (dt.date(2014, 6, 12), dt.date(2014, 7, 13)),   # Brazil
    2018: (dt.date(2018, 6, 14), dt.date(2018, 7, 15)),   # Russia
    2022: (dt.date(2022, 11, 20), dt.date(2022, 12, 18)), # Qatar
    2026: (dt.date(2026, 6, 11), dt.date(2026, 7, 19)),   # Canada/Mexico/USA
}
# Bin sizes computed from the v2.1 yearly files (design §4, 2026-10-22); stage S0 compares and logs any difference.
PERIOD_BIN_EXPECTED_N = {1: 10002, 2: 186988, 3: 83147, 4: 331216, 5: 88648, 6: 305653, 7: 41438, 8: 217303, 9: 37626}
TIME_BIN = {"tournament": "W", "interim": "M"}   # topics-over-time resolution: weekly in tournament bins, monthly in interim bins

# ---------------------------------------------------------------------------
# S1 — sentence embeddings (design §5), cached per calendar year under EMBED_CACHE_DIR/<model-slug>/<year>.npz
# ---------------------------------------------------------------------------
EMBEDDING_MODEL = "sentence-transformers/all-mpnet-base-v2"          # the pipeline's model (768-d; GPU on Colab)
EMBEDDING_MODEL_DRY_RUN = "sentence-transformers/all-MiniLM-L6-v2"  # local dry runs only (384-d, minutes on a Mac);
                                                                    # never for results that enter the analysis
EMBEDDING_BATCH_SIZE = 256
EMBEDDING_MAX_SEQ_LENGTH = 256   # tokens; title + 1,000 characters of body fit comfortably

# ---------------------------------------------------------------------------
# S2 — BERTopic per bin (design §6): frozen UMAP, HDBSCAN with the topic-count guard, c-TF-IDF keywords
# ---------------------------------------------------------------------------
SEED = 42                               # UMAP, post sampling and shuffles
UMAP_PARAMS = dict(n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=SEED)
MIN_CLUSTER_FRACTION = 0.0025           # starting HDBSCAN min_cluster_size = 0.25 % of the bin's modelled posts ...
MIN_CLUSTER_FLOOR = 25                  # ... but never fewer than 25 (design §6.1; the guard never goes below it)
HDBSCAN_MIN_SAMPLES = 10                # revisited only if the large-bin gate shows > 45 % raw outliers (design §12.2)
TOPIC_COUNT_RANGE = (50, 100)           # topic-count guard target: raw topics per bin (design §6.2)
GUARD_MAX_TRIALS = 4                    # at most four HDBSCAN fits on the frozen 5-d array
GUARD_STEP = 1.5                        # first move ×1.5 / ÷1.5, then bisection between the bracketing values
NGRAM_RANGE = (1, 2)                    # keyword candidates: unigrams and bigrams
TOP_N_WORDS = 15                        # keywords per topic
MMR_DIVERSITY = 0.3                     # maximal-marginal-relevance re-ranking of the keywords
N_REPRESENTATIVE_DOCS = 5               # representative posts per topic (BERTopic keeps 3 by default; review R3)
# r/soccer and Reddit scaffolding removed ONLY from the keyword descriptions (never from the documents or the
# embeddings); no football vocabulary is removed.  `vs` and `soccer` were added by design F11.
EXTRA_STOPWORDS = [
    "thread", "match thread", "post match", "pre match", "discussion", "daily discussion",
    "amp", "http", "https", "www", "com", "html", "reddit", "redd", "imgur", "youtube", "youtu",
    "twitter", "streamable", "gfycat", "edit", "deleted", "removed",
    "vs", "soccer",
]

# ---------------------------------------------------------------------------
# D3 — thresholded outlier reassignment (design §6.3; stage `reassign`, refinement N1)
# ---------------------------------------------------------------------------
# An outlier joins its most similar topic (cosine to the topic embedding) only if the similarity reaches T; T is
# calibrated at the quality gates from the judgement samples that stage S2 writes (outlier_calibration/).
OUTLIER_THRESHOLD: dict = {}            # {"default": T} or per bin {1: T1, 4: T4}; empty = not calibrated -> no reassignment yet
OUTLIER_CALIBRATION_QUANTILES = (0.50, 0.65, 0.80)   # candidate thresholds = these quantiles of the outlier similarity
OUTLIER_SAMPLE_PER_CANDIDATE = 100      # posts per candidate threshold in the co-author judgement sample

# ---------------------------------------------------------------------------
# S3 — the two analysts (design §7; agreement protocol = decision D2)
# ---------------------------------------------------------------------------
LLM_MODEL = "Qwen/Qwen3-1.7B"           # the brief's model (Apache-2.0) and the only model of the pipeline (the Qwen3-4B
                                        # comparison was dropped on 2026-10-24); --llm accepts another id for tests
LLM_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"   # Hugging Face snapshot of Qwen/Qwen3-1.7B used throughout
LLM_SAMPLING_SEEDED = False             # False = greedy decoding, reproducible (design F12); True = seeded sampling
LLM_SAMPLING_TEMPERATURE = 0.7          #   for the F17 comparison on bin 1 (per-call SHA-256 seeds, unbatched)
LLM_BATCH_SIZE = {"cuda": 16, "mps": 4, "cpu": 1}   # prompts generated together (greedy mode only)
MAX_ROUNDS = 5                          # decision D2: cap on debate rounds
LLM_SAMPLE_POSTS = 40                   # posts shown to the analysts per topic (5 representative + 35 seeded random)
LLM_POST_MAX_CHARS = 300                # each sample post is truncated to this length
LLM_MAX_NEW_TOKENS_LABEL = 600          # S3a answer
LLM_JSON_SALVAGE = True                 # recover answers whose only defect is bracket syntax (refinement N11, confirmed
                                        # 2026-10-24): strict JSON is tried first; recovered answers carry parsed_via =
                                        # "salvage" in the transcript.  False = a second syntax failure is "no answer"

# ---------------------------------------------------------------------------
# S3b — discourse taxonomy protocol (decision D5, 2026-10-24; design §7.4 v1.2; prompt v1.2)
# ---------------------------------------------------------------------------
#   "two_step"  = the analysts first agree on the bin's category definitions (no topic ids), then assign the topics
#                 to those categories, one debate item per TAXONOMY_ASSIGN_BATCH topics (1 = one debate per topic);
#   "partition" = the superseded v1.1 single prompt (categories with explicit `topics` lists; chunking above
#                 TAXONOMY_MAX_ITEMS), kept for comparison and replay of the session-05 dry run (flag F26).
TAXONOMY_PROTOCOL = "two_step"
TAXONOMY_ASSIGN_BATCH = 1               # topics per assignment prompt (author's decision 2026-10-24: one topic per item)
TAXONOMY_KEYWORDS_PER_TOPIC = 8         # keywords per topic line shown to the analysts
LLM_MAX_NEW_TOKENS_DEFINE = 4000        # step-1 answer (10–20 categories × ≈ 100 tokens, with margin)
LLM_MAX_NEW_TOKENS_ASSIGN = 800         # step-2 answer (one {topic_id, category_id} pair per topic in the item)
LLM_MAX_NEW_TOKENS_PARTITION = 8000     # partition protocol only (review finding R2 raised it from 4,000)
TAXONOMY_MAX_ITEMS = 100                # partition protocol only: above this many topics the chunking fallback runs
TAXONOMY_CHUNK_SIZE = 80                # partition protocol only

# ---------------------------------------------------------------------------
# Controlled tone vocabulary (design §7.3): the analysts' free-text emotion_tone is mapped to one category for counting
# ---------------------------------------------------------------------------
TONE_SET = ["positive", "negative", "neutral", "mixed", "excited", "enthusiastic", "frustrated",
            "angry", "anxious", "sad", "humorous", "critical", "supportive", "curious", "cautious"]
# Word stem -> category, checked in order (first match wins), so that "frustration and anger" -> frustrated and
# "anxious and desperate" -> anxious; a tone that matches no stem is counted as "other".
TONE_STEMS = [("frustrat", "frustrated"), ("anger", "angry"), ("angry", "angry"), ("anxi", "anxious"),
              ("despair", "sad"), ("desperat", "anxious"), ("sad", "sad"), ("grie", "sad"),
              ("enthusias", "enthusiastic"), ("excit", "excited"), ("critic", "critical"),
              ("support", "supportive"), ("curio", "curious"), ("inquisit", "curious"), ("cautio", "cautious"),
              ("humor", "humorous"), ("humour", "humorous"), ("funny", "humorous"), ("mixed", "mixed"),
              ("neutral", "neutral"), ("negativ", "negative"), ("positiv", "positive"), ("celebrat", "positive"),
              ("hopeful", "positive"), ("optimis", "positive"), ("disappoint", "negative"), ("dissatisf", "negative")]
assert {cat for _, cat in TONE_STEMS} <= set(TONE_SET), "every tone stem must map into TONE_SET"

# ---------------------------------------------------------------------------
# S4 — VADER sentiment (design §8, decision F18): post_sentiment on the cleaned TITLE, post_sentiment_full on title + body
# ---------------------------------------------------------------------------
SENTIMENT_POS_THRESHOLD = 0.05          # post_sentiment_label: positive above this, negative below the next one,
SENTIMENT_NEG_THRESHOLD = -0.05         #   otherwise neutral (VADER's conventional thresholds)
SENTIMENT_VALIDATION_SAMPLE = 300       # hand-coding sample (stratified by bin × label × is_self; refinement N9)
SENTIMENT_DECIMALS = 4


def effective_settings(**overrides) -> dict:
    """All settings as a JSON-serialisable dict (written to each bin's config.json), the effective paths included."""
    d = {k: v for k, v in globals().items()
         if k.isupper() and not k.startswith("_") and isinstance(v, (str, int, float, list, dict, tuple, Path, type(None)))}
    d = json.loads(json.dumps(d, default=str))
    d.update(overrides)
    return d
