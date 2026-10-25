"""
Stage S5 — collect every post's analysis variables and aggregate the topic labels (design §9.1, §7.7).
======================================================================================================
Inputs (all join tables by post id):  data/analysis/period_bins.csv, data/analysis/sentiment.csv and, per bin,
results/analysis/bin-N/{documents.csv, topics.csv, topics_labelled.csv, topic_overrides.csv}.  The co-authors' topic
overrides (renames, merges) are applied here (never in place), so the agents' values stay auditable (`topic_id_raw`,
`name_agent`, `merged_from`).  Since decision D7 (2026-10-25) the unit of analysis is the individual BERTopic topic:
there is no category layer.
Outputs:
  data/analysis/post_variables.csv (+ .json sidecar)  one row per clean post, the ten codebook §4.9 columns — incl. the
                                                       corpus-wide topic key `topic_uid` ("<bin>-<topic_id>"), the topic's
                                                       agreed label (`topic_name`) and controlled tone (`topic_tone`) after the
                                                       overrides — plus the audit column topic_id_raw; the input of 06-tag-corpus.R
  results/analysis/topic_labels_all_bins.csv           one row per topic (the brief's "separate CSV"), with the mean VADER
                                                       score of the topic's posts (`mean_vader_sentiment`, author 2026-10-25)
  results/analysis/topic_crosswalk.csv                 template for the co-authors' manual cross-bin crosswalk of topics
                                                       to chapter themes (F4; kept once it has entries)
S5 may run before all bins exist (refinement N6): columns stay empty for bins without outputs, the sidecar lists
`bins_complete`, and 06-tag-corpus.R refuses a partial write-back outside a dry run.
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
from . import sentiment as S
from .agents import apply_topic_overrides, read_overrides

log = logging.getLogger("topicmodel")

POST_COLUMNS = ["id", "period_bin", "period_type", "topic_id", "topic_uid", "topic_name", "topic_tone", "topic_assignment",
                "post_sentiment", "post_sentiment_full", "post_sentiment_label"]   # topic_uid / topic_name / topic_tone: author, 2026-10-25
AUDIT_COLUMNS = ["topic_id_raw"]
LABEL_COLUMNS = ["period_bin", "topic_id", "topic_uid", "name", "purpose", "emotion_tone", "tone_category", "description",
                 "inclusion_cues", "round_iterations", "agreed", "agreement_mode", "needs_review", "analyst_a_final",
                 "analyst_b_final", "n_cluster", "n_total", "share", "mean_vader_sentiment", "n_posts_scored", "keywords",
                 "llm_model", "llm_revision", "prompt_version", "decoding", "name_agent", "merged_from"]
CROSSWALK_COLUMNS = ["period_bin", "period_type", "topic_id", "topic_uid", "name", "n_total", "chapter_theme", "notes"]


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


CALIBRATION_NAME_COLUMN = "nearest_topic_name"


def topic_names(bin_dir: Path) -> dict[int, str]:
    """{topic_id: name} of a labelled bin as the posts will carry it: the analysts' agreed names with the co-authors'
    renames applied, a merged-away topic carrying its target's name (the rule of collect). Empty while unlabelled."""
    p = bin_dir / "topics_labelled.csv"
    if not p.exists():
        return {}
    labelled = _read(p)
    if labelled.empty or "topic_id" not in labelled.columns:
        return {}
    labels, merges = apply_topic_overrides(labelled, read_overrides(bin_dir / "topic_overrides.csv"))
    names = dict(zip(labels["topic_id"].astype(int), labels["name"]))
    for old, new in merges.items():
        names[int(old)] = names.get(int(new), "")
    return names


def annotate_calibration_samples(bin_dir: Path) -> int:
    """Write the nearest topic's name into every outlier_calibration/sample_T*.csv (column `nearest_topic_name`, right
    after `nearest_topic`) so the co-authors judge against a name, not only an id (author's request, 2026-10-25). Rows,
    their order and anything already filled in (`belongs`, `notes`) are kept; re-running refreshes the names (renames and
    merges from topic_overrides.csv included). Returns the number of files written: 0 while the bin is unlabelled."""
    names = topic_names(bin_dir)
    cdir = bin_dir / "outlier_calibration"
    if not names or not cdir.is_dir():
        return 0
    n = 0
    for p in sorted(cdir.glob("sample_T*.csv")):
        s = _read(p)
        if s.empty or "nearest_topic" not in s.columns:
            continue
        s[CALIBRATION_NAME_COLUMN] = [names.get(int(t), "") if str(t).strip() else "" for t in s["nearest_topic"]]
        cols = [c for c in s.columns if c != CALIBRATION_NAME_COLUMN]
        i = cols.index("nearest_topic") + 1
        s[cols[:i] + [CALIBRATION_NAME_COLUMN] + cols[i:]].to_csv(p, index=False)
        n += 1
    return n


def bin_state(bin_dir: Path) -> str:
    """missing -> topics (modelled, not labelled) -> labelled (= complete for a bin since D7)."""
    if (bin_dir / "topics_labelled.csv").exists():
        return "labelled"
    if (bin_dir / "documents.csv").exists():
        return "topics"
    return "missing"


def bin_status(bin_dir: Path) -> dict:
    """Where one bin stands (for `--stage status`, the notebook and the REVIEW_READY flag): state, topics, labelled topics,
    whether the review can start (every topic labelled), whether the outlier threshold was applied."""
    st = {"bin": int(str(bin_dir.name).split("-")[-1]) if "-" in bin_dir.name else None, "state": bin_state(bin_dir),
          "n_modelled": None, "n_topics": None, "n_labelled": 0, "n_needs_review": None, "review_ready": False,
          "reassigned": False, "outlier_threshold": None, "topics_at": None, "labelled_at": None}
    if st["state"] == "missing":
        return st
    cfg = {}
    if (bin_dir / "config.json").exists():
        try:
            cfg = json.loads((bin_dir / "config.json").read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            cfg = {}
    t = cfg.get("topics", {})
    st["n_modelled"] = t.get("n_modelled"); st["topics_at"] = t.get("run_at")
    topics = _read(bin_dir / "topics.csv")
    if len(topics):
        st["n_topics"] = int((topics["topic_id"].astype(int) != -1).sum())
    if st["state"] == "labelled":
        lab = _read(bin_dir / "topics_labelled.csv")
        st["n_labelled"] = int(len(lab))
        st["n_needs_review"] = int(lab["needs_review"].astype(str).str.lower().eq("true").sum()) if "needs_review" in lab.columns else None
        st["labelled_at"] = cfg.get("label", {}).get("run_at")
        st["review_ready"] = st["n_topics"] is not None and st["n_labelled"] >= st["n_topics"]
    if "reassign" in cfg:
        st["reassigned"] = True; st["outlier_threshold"] = cfg["reassign"].get("outlier_threshold")
    return st


def write_status(bin_dir: Path) -> dict:
    """status.json next to the bin's outputs after every per-bin stage; REVIEW_READY.txt appears when every topic of the bin
    is labelled (the flag the co-authors watch on Drive) and disappears again if the bin is re-clustered."""
    st = bin_status(bin_dir)
    st["written"] = pd.Timestamp.utcnow().isoformat(timespec="seconds")
    (bin_dir / "status.json").write_text(json.dumps(st, indent=1, default=str), encoding="utf-8")
    flag = bin_dir / "REVIEW_READY.txt"
    if st["review_ready"]:
        flag.write_text(f"bin {st['bin']}: {st['n_labelled']} topics labelled ({st['n_needs_review']} flagged needs_review), "
                        f"{st['n_modelled']} posts modelled — ready for the co-author review (topic_overrides.csv). {st['written']}\n", encoding="utf-8")
    elif flag.exists():
        flag.unlink()
    return st


def status_table(analysis_dir: Path) -> pd.DataFrame:
    """One row per bin 1–9 for `--stage status`."""
    rows = [bin_status(analysis_dir / f"bin-{b}") | {"bin": b} for b in range(1, 10)]
    return pd.DataFrame(rows)[["bin", "state", "n_modelled", "n_topics", "n_labelled", "n_needs_review", "review_ready", "reassigned", "outlier_threshold", "labelled_at"]]


def load_bin_outputs(bin_dir: Path, b: int) -> dict:
    """Everything S5 needs from one bin, with the topic overrides applied. Returns dict(posts, labels, topics, state, merges)."""
    state = bin_state(bin_dir)
    if state == "missing":
        return {"state": state}
    docs = _read(bin_dir / "documents.csv")
    topics = _read(bin_dir / "topics.csv")
    out = {"state": state, "topic_merges": {}}
    labels = None
    if state == "labelled":
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
        # the topic's label on every post of the topic (after renames and merges); empty for outliers and unmodelled posts
        name_of = dict(zip(labels["topic_id"].astype(str), labels["name"]))
        tone_of = dict(zip(labels["topic_id"].astype(str), labels["tone_category"])) if "tone_category" in labels.columns else {}
        docs["topic_name"] = docs["topic_id"].map(lambda t: name_of.get(t, "") if t not in ("", "-1") else "")
        docs["topic_tone"] = docs["topic_id"].map(lambda t: tone_of.get(t, "") if t not in ("", "-1") else "")
    else:
        docs["topic_name"] = ""; docs["topic_tone"] = ""
    # the corpus-wide topic key: "<period_bin>-<topic_id>" for posts with a topic (ids are bin-local; author, 2026-10-25)
    docs["topic_uid"] = docs["topic_id"].map(lambda t: f"{b}-{t}" if t not in ("", "-1") else "")
    out.update(posts=docs, labels=labels, topics=topics)
    return out


def _check(cond: bool, msg: str, errors: list):
    if not cond:
        errors.append(msg)
        log.error("S5 check failed: %s", msg)


def _template_is_filled(path: Path, columns: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    old = pd.read_csv(path, dtype=str, keep_default_na=False)
    return any(str(x).strip() for col in columns if col in old.columns for x in old[col])


def collect(analysis_dir: Path, analysis_data_dir: Path, allow_partial: bool, run_info: dict) -> dict:
    pb = BINS.read_period_bins(analysis_data_dir)
    pb["period_bin"] = pb["period_bin"].astype(int)
    errors, states, label_frames, post_frames = [], {}, [], []
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
        if o["labels"] is not None:
            lab = o["labels"].copy(); lab["period_bin"] = b
            label_frames.append(lab)
        _check(bool(((posts["topic_name"] != "") == (~posts["topic_id"].isin(["", "-1"]))).all()) or o["labels"] is None,
               f"bin {b}: topic_name missing for posts with a topic", errors)
        post_frames.append(posts[["id", "topic_id_raw", "topic_id", "topic_uid", "topic_name", "topic_tone", "topic_assignment", "period_bin_run"]])

    pv = pb.copy()
    if post_frames:
        allp = pd.concat(post_frames, ignore_index=True)
        _check(not allp["id"].duplicated().any(), "a post appears in more than one bin run", errors)
        pv = pv.merge(allp, on="id", how="left")
    else:
        for c in ["topic_id_raw", "topic_id", "topic_uid", "topic_name", "topic_tone", "topic_assignment", "period_bin_run"]:
            pv[c] = ""
    sent_path = analysis_data_dir / "sentiment.csv"
    if sent_path.exists():
        sent = pd.read_csv(sent_path, dtype={"id": str, "post_sentiment_label": str}, keep_default_na=False)
        pv = pv.merge(sent, on="id", how="left")
        s = pd.to_numeric(pv["post_sentiment"], errors="coerce")
        _check(bool(s.dropna().between(-1, 1).all()), "post_sentiment outside [-1, 1]", errors)
        lab_expected = s.map(lambda v: "" if np.isnan(v) else S.label_of(v))
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

    bins_complete = [b for b, st in states.items() if st == "labelled"]
    complete = len(bins_complete) == 9 and n_sent == len(pb) and not errors
    if errors and not allow_partial:
        raise SystemExit("S5 collect: checks failed:\n  " + "\n  ".join(errors))

    # the mean VADER score of each topic's posts (title score; after the topic overrides) — author's request, 2026-10-25
    topic_sent = {}
    if n_sent and post_frames:
        sc = pd.to_numeric(pv["post_sentiment"], errors="coerce")
        has = sc.notna() & (pv["topic_id"].fillna("") != "") & (pv["topic_id"].astype(str) != "-1")
        g = pd.DataFrame({"b": pv.loc[has, "period_bin_run"].astype(int), "t": pv.loc[has, "topic_id"].astype(str), "s": sc[has]}).groupby(["b", "t"])["s"]
        topic_sent = {k: (round(float(v), C.SENTIMENT_DECIMALS), int(n)) for k, v, n in zip(g.mean().index, g.mean().to_numpy(), g.size().to_numpy())}

    for c in ["topic_id", "topic_id_raw"]:
        pv[c] = pv[c].map(_int_str)
    for c in ["topic_uid", "topic_name", "topic_tone", "topic_assignment", "post_sentiment_label"]:
        pv[c] = pv[c].fillna("")
    pv = pv[POST_COLUMNS + AUDIT_COLUMNS]
    analysis_data_dir.mkdir(parents=True, exist_ok=True)
    pv.to_csv(analysis_data_dir / "post_variables.csv", index=False, float_format=f"%.{C.SENTIMENT_DECIMALS}f")

    # aggregated tables
    analysis_dir.mkdir(parents=True, exist_ok=True)
    n_topics = 0
    if label_frames:
        lab = pd.concat(label_frames, ignore_index=True)
        keys = list(zip(lab["period_bin"].astype(int), lab["topic_id"].astype(str)))
        lab["topic_uid"] = [f"{b_}-{t_}" for b_, t_ in keys]
        lab["mean_vader_sentiment"] = [f"{topic_sent[k][0]:.{C.SENTIMENT_DECIMALS}f}" if k in topic_sent else "" for k in keys]
        lab["n_posts_scored"] = [topic_sent[k][1] if k in topic_sent else "" for k in keys]
        for c in LABEL_COLUMNS:
            if c not in lab.columns: lab[c] = ""
        lab = lab[LABEL_COLUMNS].sort_values(["period_bin", "topic_id"])
        lab.to_csv(analysis_dir / "topic_labels_all_bins.csv", index=False)
        n_topics = int(len(lab))
        # the manual cross-bin crosswalk of topics (F4): a template, kept once a co-author has filled it in
        xw = analysis_dir / "topic_crosswalk.csv"
        if _template_is_filled(xw, ("chapter_theme", "notes")):
            log.warning("%s has entries and is kept; add rows for new bins by hand", xw.name)
        else:
            ptype = dict(zip(pb["period_bin"], pb["period_type"])) if "period_type" in pb.columns else {}
            pd.DataFrame({"period_bin": lab["period_bin"], "period_type": lab["period_bin"].map(ptype).fillna(""),
                          "topic_id": lab["topic_id"], "topic_uid": lab["topic_uid"], "name": lab["name"], "n_total": lab["n_total"],
                          "chapter_theme": "", "notes": ""})[CROSSWALK_COLUMNS].to_csv(xw, index=False)
    labels_version = hashlib.sha256("".join(
        (analysis_dir / f"bin-{b}" / "topics_labelled.csv").read_text(encoding="utf-8") for b in bins_complete).encode()).hexdigest()[:12] if bins_complete else None
    sidecar = {"written": pd.Timestamp.utcnow().isoformat(timespec="seconds"), "n_posts": int(len(pv)),
               "bins_state": states, "bins_complete": bins_complete, "complete": complete, "n_with_sentiment": n_sent,
               "n_with_topic": int((pv["topic_id"] != "").sum()), "n_topics_labelled": n_topics,
               "labels_version": labels_version, "columns": POST_COLUMNS, "audit_columns": AUDIT_COLUMNS,
               "errors": errors, **run_info}
    (analysis_data_dir / "post_variables.json").write_text(json.dumps(sidecar, indent=1, default=str), encoding="utf-8")
    log.info("S5: post_variables.csv %d rows; bins complete %s; %d topics; sentiment %d; complete=%s%s", len(pv), bins_complete, n_topics,
             n_sent, complete, f"; {len(errors)} check(s) failed (allowed by --allow-partial)" if errors else "")
    return sidecar
