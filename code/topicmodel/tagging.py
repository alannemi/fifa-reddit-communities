"""
Stage S5 — collect every post's analysis variables and aggregate the topic labels (design §9.1, §7.7).
======================================================================================================
Inputs (all join tables by post id):  data/analysis/period_bins.csv, data/analysis/sentiment.csv and, per bin,
results/analysis/bin-N/{documents.csv, topics.csv, topics_labelled.csv, topic_overrides.csv, topic_categories.csv,
categories.csv, category_overrides.csv, topic_category_overrides.csv}.  Overrides are applied here (never in place), so
the agents' values stay auditable (`topic_id_raw`, `category_id_agent`, `name_agent`).  Order of the category
overrides (F28, 2026-10-24): `category_id_agent` = the agents' value; then `final_category_id` moves of
topic_category_overrides.csv on the topic that survives `merge_into`; then the category merges of
category_overrides.csv; then topic lists and sizes are recomputed (a category emptied by moves stays with n_topics 0).
Outputs:
  data/analysis/post_variables.csv (+ .json sidecar)  one row per clean post, the eight codebook §4.9 columns plus
                                                       the two audit columns; the input of 07-tag-corpus.R
  results/analysis/topic_labels_all_bins.csv           one row per topic cluster (the brief's "separate CSV")
  results/analysis/categories_all_bins.csv             one row per category of each bin's taxonomy
  results/analysis/category_crosswalk.csv              empty template for the manual cross-bin crosswalk (F4)
S5 may run before all bins exist (refinement N6): columns stay empty for bins without outputs, the sidecar lists
`bins_complete`, and 07-tag-corpus.R refuses a partial write-back outside a dry run.
"""
from __future__ import annotations
import hashlib
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import bins as BINS
from .agents import apply_topic_overrides, apply_category_overrides, read_overrides, read_final_categories, ASSIGN_COLS

log = logging.getLogger("topicmodel")

POST_COLUMNS = ["id", "period_bin", "period_type", "topic_id", "topic_assignment", "category_id",
                "post_sentiment", "post_sentiment_full", "post_sentiment_label"]
AUDIT_COLUMNS = ["topic_id_raw", "category_id_agent"]
LABEL_COLUMNS = ["period_bin", "topic_id", "name", "purpose", "emotion_tone", "tone_category", "description",
                 "inclusion_cues", "round_iterations", "agreed", "agreement_mode", "needs_review", "analyst_a_final",
                 "analyst_b_final", "category_id", "category_name", "category_assignment", *ASSIGN_COLS, "n_cluster", "n_total",
                 "share", "keywords", "llm_model", "llm_revision", "prompt_version", "decoding", "name_agent", "merged_from"]
CATEGORY_COLUMNS = ["period_bin", "category_id", "name", "purpose", "emotion_tone", "tone_category", "description",
                    "inclusion_cues", "topic_ids", "n_topics", "n_posts", "share", "round_iterations", "agreed",
                    "agreement_mode", "needs_review", "name_agent", "merged_from"]


def _read(p: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(p, dtype=str, keep_default_na=False)
    except pd.errors.EmptyDataError:
        return pd.DataFrame()


def _int_str(x) -> str:
    """Integer columns are written as plain integers or empty (never "3.0")."""
    if x is None or x == "" or (isinstance(x, float) and np.isnan(x)):
        return ""
    return str(int(float(x)))


def bin_state(bin_dir: Path) -> str:
    if (bin_dir / "categories.csv").exists() and (bin_dir / "topics_labelled.csv").exists():
        if len(_read(bin_dir / "categories.csv")):
            return "complete"
        return "labelled"          # the taxonomy stage ran but produced no category (needs a re-run / review)
    if (bin_dir / "topics_labelled.csv").exists():
        return "labelled"
    if (bin_dir / "documents.csv").exists():
        return "topics"
    return "missing"


def load_bin_outputs(bin_dir: Path, b: int) -> dict:
    """Everything S5 needs from one bin, with overrides applied. Returns dict(posts, labels, categories, state, merges)."""
    state = bin_state(bin_dir)
    if state == "missing":
        return {"state": state}
    docs = _read(bin_dir / "documents.csv")
    topics = _read(bin_dir / "topics.csv")
    out = {"state": state, "topic_merges": {}, "category_merges": {}}
    labels = cats = tmap = None
    if state in ("labelled", "complete"):
        labelled = _read(bin_dir / "topics_labelled.csv")
        labels, merges = apply_topic_overrides(labelled, read_overrides(bin_dir / "topic_overrides.csv"))
        out["topic_merges"] = merges
        # topic_id after overrides (raw ids and outliers untouched)
        docs["topic_id"] = ["" if x == "" else str(merges.get(int(x), int(x))) for x in docs["topic_id"]]
        sizes = topics.set_index(topics["topic_id"].astype(int))
        n_cluster = sizes["n_cluster"].astype(int).to_dict(); n_total = sizes["n_total"].astype(int).to_dict()
        kw = sizes["keywords"].to_dict()
        for old, new in merges.items():
            n_cluster[new] = n_cluster.get(new, 0) + n_cluster.pop(old, 0)
            n_total[new] = n_total.get(new, 0) + n_total.pop(old, 0)
        merged_from = {}
        for old, new in merges.items():
            merged_from.setdefault(new, []).append(str(old))
        n_mod = max(1, sum(n_total.values()))
        labels["n_cluster"] = labels["topic_id"].map(n_cluster).fillna(0).astype(int)
        labels["n_total"] = labels["topic_id"].map(n_total).fillna(0).astype(int)
        labels["share"] = (labels["n_total"] / n_mod).round(5)
        labels["keywords"] = labels["topic_id"].map(kw).fillna(labels.get("keywords", ""))
        labels["merged_from"] = labels["topic_id"].map(lambda t: "|".join(merged_from.get(t, [])))
    if state == "complete":
        cats_raw = _read(bin_dir / "categories.csv")
        tmap = _read(bin_dir / "topic_categories.csv")
        cats, cmerges = apply_category_overrides(cats_raw, read_overrides(bin_dir / "category_overrides.csv"))
        out["category_merges"] = cmerges
        tmap["category_id_agent"] = tmap["category_id"]
        # F28: topic-level moves (topic_category_overrides.csv) on the surviving topics, before the category merges
        moves = read_final_categories(bin_dir / "topic_category_overrides.csv")
        applied = {}
        for t, cid in moves.items():
            if t in merges:
                log.warning("bin %d: topic_category_overrides for topic %d ignored — the topic was merged into %d", b, t, merges[t]); continue
            sel = tmap["topic_id"].astype(int) == t
            if not sel.any():
                log.warning("bin %d: topic_category_overrides names topic %d, which has no category row", b, t); continue
            tmap.loc[sel, "category_id"] = cid
            applied[t] = cid
        out["category_moves"] = applied
        tmap["category_id"] = tmap["category_id"].map(lambda c: cmerges.get(c, c))
        cmerged_from = {}
        for old, new in cmerges.items():
            cmerged_from.setdefault(new, []).append(old)
        cats["merged_from"] = cats["category_id"].map(lambda c: "|".join(cmerged_from.get(c, [])))
        # recompute topic lists / sizes after merges
        grp = tmap.groupby("category_id")["topic_id"].apply(lambda s: "|".join(sorted(s, key=int)))
        cats["topic_ids"] = cats["category_id"].map(grp).fillna("")
        cats["n_topics"] = cats["topic_ids"].map(lambda s: len([x for x in s.split("|") if x]))
        n_total_map = dict(zip(labels["topic_id"].astype(str), labels["n_total"])) if labels is not None else {}
        cats["n_posts"] = cats["topic_ids"].map(lambda s: sum(n_total_map.get(x, 0) for x in s.split("|") if x))
        cats["share"] = (cats["n_posts"] / max(1, sum(n_total_map.values()))).round(5)
        # per topic: category after overrides
        cat_of = dict(zip(tmap["topic_id"].astype(str), tmap["category_id"]))
        cat_agent_of = dict(zip(tmap["topic_id"].astype(str), tmap["category_id_agent"]))
        assign_of = dict(zip(tmap["topic_id"].astype(str), tmap["category_assignment"]))
        name_of = dict(zip(cats["category_id"], cats["name"]))
        labels["category_id"] = labels["topic_id"].astype(str).map(cat_of).fillna("")
        labels["category_name"] = labels["category_id"].map(name_of).fillna("")
        labels["category_assignment"] = labels["topic_id"].astype(str).map(assign_of).fillna("")
        for k in ASSIGN_COLS:   # per-topic assignment debate (two-step protocol); empty for v1.1 outputs
            m = dict(zip(tmap["topic_id"].astype(str), tmap[k])) if k in tmap.columns else {}
            labels[k] = labels["topic_id"].astype(str).map(m).fillna("")
        docs["category_id"] = docs["topic_id"].map(lambda t: cat_of.get(t, "") if t not in ("", "-1") else "")
        docs["category_id_agent"] = docs["topic_id"].map(lambda t: cat_agent_of.get(t, "") if t not in ("", "-1") else "")
    else:
        docs["category_id"] = ""; docs["category_id_agent"] = ""
        if labels is not None:
            labels["category_id"] = ""; labels["category_name"] = ""; labels["category_assignment"] = ""
            for k in ASSIGN_COLS:
                labels[k] = ""
    out.update(posts=docs, labels=labels, categories=cats, topics=topics)
    return out


def _check(cond: bool, msg: str, errors: list):
    if not cond:
        errors.append(msg)
        log.error("S5 check failed: %s", msg)


def collect(analysis_dir: Path, analysis_data_dir: Path, allow_partial: bool, run_info: dict) -> dict:
    pb = BINS.read_period_bins(analysis_data_dir)
    pb["period_bin"] = pb["period_bin"].astype(int)
    errors, states, label_frames, cat_frames, post_frames = [], {}, [], [], []
    for b in range(1, 10):
        bdir = analysis_dir / f"bin-{b}"
        o = load_bin_outputs(bdir, b)
        states[b] = o["state"]
        if o["state"] == "missing":
            continue
        posts = o["posts"].copy()
        posts["period_bin_run"] = b
        expected = set(pb.loc[pb["period_bin"] == b, "id"])
        present = set(posts["id"])
        if present - expected:
            _check(False, f"bin {b}: {len(present - expected)} posts in documents.csv do not belong to the bin", errors)
        if expected - present:
            msg = f"bin {b}: {len(expected - present)} posts of the bin are missing from documents.csv (debug subsample?)"
            if allow_partial:
                log.warning(msg)
            else:
                _check(False, msg, errors)
        valid_topics = set(o["topics"]["topic_id"])
        if o["topic_merges"]:
            valid_topics = {str(o["topic_merges"].get(int(t), int(t))) for t in valid_topics}
        tid = posts["topic_id"]
        _check(set(tid[tid != ""]) <= valid_topics, f"bin {b}: topic ids not in topics.csv", errors)
        ass = posts["topic_assignment"]
        _check(bool(((ass == "outlier") == (tid == "-1")).all()), f"bin {b}: topic_assignment 'outlier' inconsistent with topic_id -1", errors)
        _check(bool(((ass == "not_modelled") == (tid == "")).all()), f"bin {b}: topic_assignment 'not_modelled' inconsistent with empty topic_id", errors)
        if o["state"] == "complete":
            valid_cats = set(o["categories"]["category_id"])
            cid = posts["category_id"]
            _check(set(cid[cid != ""]) <= valid_cats, f"bin {b}: category ids not in categories.csv", errors)
            cats = o["categories"].copy(); cats["period_bin"] = b
            cat_frames.append(cats)
        if o["labels"] is not None:
            lab = o["labels"].copy(); lab["period_bin"] = b
            label_frames.append(lab)
        post_frames.append(posts[["id", "topic_id_raw", "topic_id", "topic_assignment", "category_id", "category_id_agent"]])

    pv = pb.copy()
    if post_frames:
        allp = pd.concat(post_frames, ignore_index=True)
        _check(not allp["id"].duplicated().any(), "a post appears in more than one bin run", errors)
        pv = pv.merge(allp, on="id", how="left")
    else:
        for c in ["topic_id_raw", "topic_id", "topic_assignment", "category_id", "category_id_agent"]:
            pv[c] = ""
    sent_path = analysis_data_dir / "sentiment.csv"
    if sent_path.exists():
        sent = pd.read_csv(sent_path, dtype={"id": str, "post_sentiment_label": str}, keep_default_na=False)
        pv = pv.merge(sent, on="id", how="left")
        s = pd.to_numeric(pv["post_sentiment"], errors="coerce")
        _check(bool(s.dropna().between(-1, 1).all()), "post_sentiment outside [-1, 1]", errors)
        lab_expected = s.map(lambda v: "" if np.isnan(v) else ("positive" if v > C.SENTIMENT_POS_THRESHOLD else "negative" if v < C.SENTIMENT_NEG_THRESHOLD else "neutral"))
        got = pv["post_sentiment_label"].fillna("")
        _check(bool((lab_expected == got).all()), "post_sentiment_label inconsistent with the thresholds", errors)
        n_sent = int(s.notna().sum())
    else:
        for c in ["post_sentiment", "post_sentiment_full", "post_sentiment_label"]:
            pv[c] = ""
        n_sent = 0
        msg = "sentiment.csv missing"
        if allow_partial: log.warning(msg)
        else: _check(False, msg, errors)
    _check(len(pv) == len(pb) and not pv["id"].duplicated().any(), "post_variables must hold every corpus id exactly once", errors)

    bins_complete = [b for b, s in states.items() if s == "complete"]
    complete = len(bins_complete) == 9 and n_sent == len(pb) and not errors
    if errors and not allow_partial:
        raise SystemExit("S5 collect: checks failed:\n  " + "\n  ".join(errors))

    for c in ["topic_id", "topic_id_raw"]:
        pv[c] = pv[c].map(_int_str)
    for c in ["topic_assignment", "category_id", "category_id_agent", "post_sentiment_label"]:
        pv[c] = pv[c].fillna("")
    pv = pv[POST_COLUMNS + AUDIT_COLUMNS]
    analysis_data_dir.mkdir(parents=True, exist_ok=True)
    pv.to_csv(analysis_data_dir / "post_variables.csv", index=False, float_format=f"%.{C.SENTIMENT_DECIMALS}f")

    # aggregated tables
    analysis_dir.mkdir(parents=True, exist_ok=True)
    if label_frames:
        lab = pd.concat(label_frames, ignore_index=True)
        for c in LABEL_COLUMNS:
            if c not in lab.columns: lab[c] = ""
        lab = lab[LABEL_COLUMNS].sort_values(["period_bin", "topic_id"])
        lab.to_csv(analysis_dir / "topic_labels_all_bins.csv", index=False)
    if cat_frames:
        cat = pd.concat(cat_frames, ignore_index=True)
        for c in CATEGORY_COLUMNS:
            if c not in cat.columns: cat[c] = ""
        cat = cat[CATEGORY_COLUMNS].sort_values(["period_bin", "category_id"])
        cat.to_csv(analysis_dir / "categories_all_bins.csv", index=False)
        xw = analysis_dir / "category_crosswalk.csv"
        if not xw.exists():
            pd.DataFrame({"period_bin": cat["period_bin"], "category_id": cat["category_id"], "category_name": cat["name"],
                          "chapter_theme": "", "notes": ""}).to_csv(xw, index=False)
    taxonomy_version = hashlib.sha256("".join(
        (analysis_dir / f"bin-{b}" / "categories.csv").read_text(encoding="utf-8") for b in bins_complete).encode()).hexdigest()[:12] if bins_complete else None
    sidecar = {"written": pd.Timestamp.utcnow().isoformat(timespec="seconds"), "n_posts": int(len(pv)),
               "bins_state": states, "bins_complete": bins_complete, "complete": complete, "n_with_sentiment": n_sent,
               "n_with_topic": int((pv["topic_id"] != "").sum()), "n_with_category": int((pv["category_id"] != "").sum()),
               "taxonomy_version": taxonomy_version, "columns": POST_COLUMNS, "audit_columns": AUDIT_COLUMNS,
               "errors": errors, **run_info}
    (analysis_data_dir / "post_variables.json").write_text(json.dumps(sidecar, indent=1, default=str), encoding="utf-8")
    log.info("S5: post_variables.csv %d rows; bins complete %s; sentiment %d; complete=%s%s", len(pv), bins_complete, n_sent, complete,
             f"; {len(errors)} check(s) failed (allowed by --allow-partial)" if errors else "")
    return sidecar
