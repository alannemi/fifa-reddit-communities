"""
Stage S0 — period bins.
=======================
The brief divides the corpus into nine contiguous periods (five World Cup
"tournament" bins and four "interim" bins between them; design §4).  A post's
bin is decided by the UTC calendar date of `created_datetime`.  The bins cover
the whole corpus, so every post gets exactly one bin; the stage checks that
the bin sizes add up to the number of posts read.

Outputs (config.ANALYSIS_DATA_DIR; <out-root>/data/analysis/ in dry runs):
  period_bins.csv          id, period_bin, period_type      -- the join table used by every later stage
  period_bins_summary.csv  one row per bin with the counts that the design table (§4) reports

`load_bin()` is how the later stages obtain a bin's documents: the yearly
files the bin touches are loaded with `data.build_documents()` and filtered
through period_bins.csv (never by recomputing dates), so S0 is the single
source of truth for `period_bin` (refinement N8).
"""
from __future__ import annotations
import logging
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C
from . import data

log = logging.getLogger("topicmodel")

_STARTS = np.array([np.datetime64(b["start"], "D") for b in C.PERIOD_BINS])
_ENDS = np.array([np.datetime64(b["end"], "D") for b in C.PERIOD_BINS])
_IDS = np.array([b["bin"] for b in C.PERIOD_BINS])


def bin_info(b: int) -> dict:
    return next(x for x in C.PERIOD_BINS if x["bin"] == int(b))


def period_type_of(b: int) -> str:
    return bin_info(b)["period_type"]


def bin_years(b: int) -> list[int]:
    """Calendar years whose yearly files intersect the bin (bin 7 = 2022 and 2023)."""
    info = bin_info(b)
    return list(range(info["start"].year, info["end"].year + 1))


def bin_of(dates: pd.Series) -> np.ndarray:
    """period_bin for a Series of UTC datetimes (tz-aware or naive UTC). Raises if any date falls outside the bins."""
    d = pd.to_datetime(dates, utc=True).dt.tz_localize(None).to_numpy().astype("datetime64[D]")
    idx = np.searchsorted(_STARTS, d, side="right") - 1
    bad = (idx < 0) | (d > _ENDS[np.clip(idx, 0, len(_ENDS) - 1)])
    if bad.any():
        raise ValueError(f"{int(bad.sum())} dates fall outside the period bins, e.g. {d[bad][:5]}")
    return _IDS[idx]


# ---------------------------------------------------------------------------
# S0: build the join table from the yearly files
# ---------------------------------------------------------------------------
_USECOLS = ["id", "created_datetime", "title", "selftext", "is_self", "post_type", "lang"]


def build_period_bins(out_dir: Path) -> pd.DataFrame:
    """Read the 17 yearly files one at a time (text dropped after deriving flags, R10), assign bins, write the
    join table and the summary; return the summary."""
    out_dir.mkdir(parents=True, exist_ok=True)
    years = data.available_years()
    parts = []
    for y in years:
        df = pd.read_csv(data.yearly_file(y), dtype=str, keep_default_na=False,
                         usecols=_USECOLS, encoding="utf-8")
        part = pd.DataFrame({
            "id": df["id"],
            "period_bin": bin_of(df["created_datetime"]),
            "is_self": df["is_self"].eq("True"),
            "has_body": df["selftext"].str.strip().ne(""),
            "match_thread": df["post_type"].eq("match_thread"),
            "lang_en": df["lang"].eq("en"),
            "title_short": df["title"].map(data.clean_text).map(data.n_tokens) < C.MIN_TOKENS,
        })
        parts.append(part)
        log.info("bins: %d -> %d posts read", y, len(part))
        del df
    allp = pd.concat(parts, ignore_index=True)
    n_total = len(allp)
    allp["period_type"] = allp["period_bin"].map(period_type_of)
    if allp["id"].duplicated().any():
        raise RuntimeError("duplicate post ids across the yearly files")

    # join table
    allp[["id", "period_bin", "period_type"]].to_csv(out_dir / "period_bins.csv", index=False)

    # summary (the design §4 table)
    g = allp.groupby("period_bin")
    summ = pd.DataFrame({
        "period_bin": g.size().index,
        "period_type": [period_type_of(b) for b in g.size().index],
        "start": [bin_info(b)["start"].isoformat() for b in g.size().index],
        "end": [bin_info(b)["end"].isoformat() for b in g.size().index],
        "n_posts": g.size().to_numpy(),
        "share_self": g["is_self"].mean().round(4).to_numpy(),
        "share_with_body": g["has_body"].mean().round(4).to_numpy(),
        "share_match_thread": g["match_thread"].mean().round(4).to_numpy(),
        "share_lang_en": g["lang_en"].mean().round(4).to_numpy(),
        "n_title_short": g["title_short"].sum().to_numpy(),
    })
    tr = [C.target_topic_range(int(n)) for n in summ["n_posts"]]           # D8: 50–75 topics per 10,000 posts (approx., on n_posts)
    summ["target_topics_min"] = [lo for lo, _ in tr]; summ["target_topics_max"] = [hi for _, hi in tr]
    summ["expected_n_design"] = summ["period_bin"].map(C.PERIOD_BIN_EXPECTED_N)
    summ.to_csv(out_dir / "period_bins_summary.csv", index=False)

    # checks: the bins partition the corpus
    if int(summ["n_posts"].sum()) != n_total:
        raise RuntimeError(f"bin sizes sum to {summ['n_posts'].sum()} but {n_total} posts were read")
    diff = summ[summ["n_posts"] != summ["expected_n_design"]]
    if len(diff):
        log.warning("bins: %d bin(s) differ from the design table (data changed?):\n%s", len(diff),
                    diff[["period_bin", "n_posts", "expected_n_design"]].to_string(index=False))
    else:
        log.info("bins: all nine bin sizes equal the design table; Σ = %s", f"{n_total:,}")
    log.info("bins: wrote %s and %s", out_dir / "period_bins.csv", out_dir / "period_bins_summary.csv")
    return summ


def read_period_bins(analysis_data_dir: Path) -> pd.DataFrame:
    p = analysis_data_dir / "period_bins.csv"
    if not p.exists():
        raise SystemExit(f"{p} not found: run `--stage bins` first")
    pb = pd.read_csv(p, dtype=str, keep_default_na=False)
    pb["period_bin"] = pb["period_bin"].astype(int)
    return pb


def load_bin(b: int, analysis_data_dir: Path) -> tuple[pd.DataFrame, dict[int, pd.DataFrame]]:
    """Documents of one bin (with `period_bin` column) and the full-year document tables for the embedding cache."""
    pb = read_period_bins(analysis_data_dir)
    full = {y: data.build_documents(data.load_year(y)) for y in bin_years(b)}
    docs = pd.concat(full.values(), ignore_index=True)
    docs = docs.merge(pb[["id", "period_bin"]], on="id", how="left")
    if docs["period_bin"].isna().any():
        raise RuntimeError("posts without a period_bin: period_bins.csv is stale; re-run `--stage bins --force`")
    docs = docs[docs["period_bin"] == int(b)].reset_index(drop=True)
    log.info("bin %d: %d documents from years %s", b, len(docs), bin_years(b))
    return docs, full
