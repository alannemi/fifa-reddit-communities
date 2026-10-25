#!/usr/bin/env python
"""
FIFA COMMUNITIES CHAPTER
5. ANALYTICAL PIPELINE — period bins → BERTopic per bin → two-agent topic labels → VADER → tagging
=====================================================================================================

PURPOSE
-------
Run the chapter's analytical pipeline stage by stage.  Design: docs/session-04_2026-10-22/01_analytical-pipeline-design.md
(v1.5, decisions D1–D8; since D7 of 2026-10-25 the unit of analysis is the individual BERTopic topic — no category layer).
Approvals: docs/session-05_2026-10-23/00_approvals.md (N1–N12, R1–R13), docs/session-07_2026-10-25/00_approvals.md
(one pipeline, PIPELINE_ROOT), docs/session-08_2026-10-25/00_colab-run-log.md (D7, D8).  Every stage is idempotent (it skips when its output exists; --force re-runs) and
resumable; every bin writes config.json (effective settings and paths, guard trials, backend, versions, prompts) and
log.txt next to its outputs.  The Colab notebook (code/05b-analysis-pipeline-colab.ipynb) only calls this program.

STAGES (design §3)                                            reads -> writes   (<tables> = config.ANALYSIS_DATA_DIR, <results> = config.ANALYSIS_DIR)
------------------                                            ---------------
  bins        S0 period_bin for every post                    yearly CSVs -> <tables>/period_bins.csv (+ summary)
  embed       S1 sentence embeddings, cached per year         yearly CSVs -> EMBED_CACHE_DIR/<model>/<year>.npz
  topics      S2 BERTopic for one bin (frozen UMAP, D8 guard)    S0, S1 -> <results>/bin-N/ (umap_5d.npy, hdbscan_labels_raw.npy,
                                                              documents.csv, topics.csv, topics_over_time.csv, model/, outlier_nearest.csv,
                                                              outlier_calibration/, config.json, report.html)
  grid        diagnostic of a failed large-bin gate (§12.2):  bin-N/umap_5d.npy -> bin-N/hdbscan_grid.csv (one row per HDBSCAN
              HDBSCAN over config.HDBSCAN_GRID on the cached   setting: topics, outlier share, largest cluster); the author chooses
              UMAP array (GPU under cuml.accel)               the setting, then `topics --force --keep-umap`
  reassign    D3 thresholded outlier reassignment             bin-N/ (+ --outlier-threshold T or config.OUTLIER_THRESHOLD) -> rewrites
                                                              documents.csv, topics.csv, topics_over_time.csv, report.html
  label       S3a two-agent cluster labelling (resumable)     bin-N/ -> labels/topic_<id>.json, labels.jsonl, topics_labelled.csv,
                                                              topic_overrides.csv (template)   -> co-author review
  sentiment   S4 VADER per post (title; title + body)         yearly CSVs -> <tables>/sentiment.csv (+ validation sample)
  collect     S5 post_variables.csv + aggregated tables       S0–S4 (+ topic overrides) -> <tables>/post_variables.csv (+ .json),
                                                              <results>/topic_labels_all_bins.csv (with mean_vader_sentiment),
                                                              topic_crosswalk.csv (template)
  report      per-bin report.html (re-render)                 bin-N/ -> report.html
  status      where every bin stands (no --bin)               bin-N/status.json (written after topics/reassign/label) -> a table;
                                                              bin-N/REVIEW_READY.txt appears when every topic of a bin is labelled
S6 (append the eight columns to the master and yearly files) is code/06-tag-corpus.R + code/07-validate-analysis.R.

HOW TO RUN (repository root; local .venv, or Colab through the notebook)
------------------------------------------------------------------------
  export TOKENIZERS_PARALLELISM=false
  python code/05-analysis-pipeline.py --stage bins
  python code/05-analysis-pipeline.py --stage embed --bin 1
  python -m cuml.accel code/05-analysis-pipeline.py --stage topics --bin 1      # GPU UMAP/HDBSCAN on Colab; without the prefix on CPU
  python code/05-analysis-pipeline.py --stage label --bin 1
  python code/05-analysis-pipeline.py --stage sentiment
  python code/05-analysis-pipeline.py --stage collect
PATHS: by default the repository layout (data/yearly datasets, data/analysis, results/analysis, results/embeddings,
  .hf-cache).  PIPELINE_ROOT=<folder> re-roots everything that persists on the flat layout of the shared Drive folder
  MyDrive/fifa-communities/ (yearly datasets/, analysis/, results/, embeddings/, hf-cache/) — the notebook sets it;
  PIPELINE_YEARLY_DIR overrides the input folder alone.  HF_HOME defaults to config.LLM_CACHE_DIR.
OPTIONS: --bin N | --bins 1-9   --device auto|cuda|mps|cpu   --embedding-model <hf id> (default config.EMBEDDING_MODEL)
  --llm <hf id>   --sampling greedy|seeded   --max-docs N (mechanics test on a subsample; guard off)   --max-topics N
  --outlier-threshold T   --years 2010 2011   --out-root PATH (dry runs: writes <PATH>/results/analysis AND
  <PATH>/data/analysis; input and caches unchanged)   --allow-partial (collect)   --force
  --keep-umap (topics with --force: keep umap_5d.npy, refit HDBSCAN only — after a clustering decision)   --grid '<json>' (grid stage:
  override config.HDBSCAN_GRID, e.g. '{"min_samples": [5]}')
Local dry run on the full bin 1 (memory/workflows.md §7): --embedding-model sentence-transformers/all-MiniLM-L6-v2
  (= config.EMBEDDING_MODEL_DRY_RUN)  --out-root results/dry-runs/<name>
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))   # make `topicmodel` importable
from topicmodel import config as C
from topicmodel import data, embed, bins as BINS, model as M, sentiment, report, tagging, prompts as P
from topicmodel import agents as G

log = logging.getLogger("topicmodel")
STAGES = ["bins", "embed", "topics", "grid", "reassign", "label", "sentiment", "collect", "report", "status"]


# ---------------------------------------------------------------------------
# Paths, logging, run metadata
# ---------------------------------------------------------------------------
class Paths:
    """The two output folders of a run: config.ANALYSIS_DIR / ANALYSIS_DATA_DIR (repository layout, or the flat Drive
    layout under PIPELINE_ROOT), or both under --out-root for dry runs (repository-shaped, so the R scripts read them)."""

    def __init__(self, out_root: str | None):
        if out_root:
            root = Path(out_root).resolve()
            self.analysis = root / "results" / "analysis"
            self.analysis_data = root / "data" / "analysis"
            self.out_root = str(root)
        else:
            self.analysis, self.analysis_data, self.out_root = C.ANALYSIS_DIR, C.ANALYSIS_DATA_DIR, None
        self.analysis.mkdir(parents=True, exist_ok=True)
        self.analysis_data.mkdir(parents=True, exist_ok=True)

    def describe(self) -> dict:
        """Every path a run reads or writes (logged at the start of each stage; stored in post_variables.json)."""
        return {"pipeline_root": C.PIPELINE_ROOT, "yearly_dir": str(C.YEARLY_DIR), "analysis_dir": str(self.analysis),
                "analysis_data_dir": str(self.analysis_data), "embed_cache_dir": str(C.EMBED_CACHE_DIR),
                "llm_cache_dir": str(C.LLM_CACHE_DIR), "out_root": self.out_root}

    def bin_dir(self, b: int) -> Path:
        d = self.analysis / f"bin-{b}"
        d.mkdir(parents=True, exist_ok=True)
        return d


def setup_logging(log_file: Path | None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, encoding="utf-8"))
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s", handlers=handlers, force=True)


def versions() -> dict:
    from importlib.metadata import version, distributions, PackageNotFoundError
    out = {"python": sys.version.split()[0]}
    for pkg in ("bertopic", "sentence-transformers", "torch", "transformers", "umap-learn", "hdbscan", "scikit-learn",
                "vaderSentiment", "pandas", "numpy", "cuml", "cuml-cu12", "cuml-cu13"):
        try:
            out[pkg] = version(pkg)
        except PackageNotFoundError:
            pass
    try:
        import torch
        out["torch_cuda"] = torch.version.cuda
        out["gpu"] = torch.cuda.get_device_name(0) if torch.cuda.is_available() else None
    except Exception:
        pass
    out["pip_freeze"] = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions() if d.metadata and d.metadata.get("Name"))
    return out


def read_config(bin_dir: Path) -> dict:
    p = bin_dir / "config.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else {}


def update_config(bin_dir: Path, section: str, payload: dict) -> None:
    cfg = read_config(bin_dir)
    cfg[section] = payload
    cfg["updated"] = dt.datetime.now().isoformat(timespec="seconds")
    (bin_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")


def bin_documents(args, paths: Paths, b: int):
    """(all docs of the bin, modelled docs, full-year docs) with the debug subsample applied if requested."""
    docs, full = BINS.load_bin(b, paths.analysis_data)
    if args.max_docs and len(docs) > args.max_docs:
        docs = docs.sample(args.max_docs, random_state=C.SEED).sort_values("date").reset_index(drop=True)
        log.info("DEBUG: subsampled bin %d to %d documents (guard disabled)", b, len(docs))
    modelled = docs[~docs["too_short"]].reset_index(drop=True)
    return docs, modelled, full


def parse_bins(args) -> list[int]:
    if args.bins:
        a, _, z = args.bins.partition("-")
        return list(range(int(a), int(z or a) + 1))
    if args.bin:
        return [args.bin]
    raise SystemExit("--bin N or --bins A-B is required for this stage")


# ---------------------------------------------------------------------------
# S0 / S1
# ---------------------------------------------------------------------------
def stage_bins(args, paths: Paths):
    if (paths.analysis_data / "period_bins.csv").exists() and not args.force:
        log.info("bins: %s exists (use --force to rebuild)", paths.analysis_data / "period_bins.csv"); return
    summ = BINS.build_period_bins(paths.analysis_data)
    print(summ.to_string(index=False))


def stage_embed(args, paths: Paths, b: int):
    device = embed.get_device(args.device)
    _, _, full = bin_documents(args, paths, b)
    st_model = None
    for y, ydocs in full.items():   # the cache holds ALL documents of a year (too-short ones included)
        if args.force or not embed.cached(args.embedding_model, y):
            st_model = st_model or embed.load_embedding_model(args.embedding_model, device)
        embed.embed_year(y, ydocs, args.embedding_model, device, st_model, force=args.force)


# ---------------------------------------------------------------------------
# S2: topics
# ---------------------------------------------------------------------------
def stage_topics(args, paths: Paths, b: int):
    bdir = paths.bin_dir(b)
    if (bdir / "documents.csv").exists() and not args.force:
        log.info("topics: bin %d already modelled (%s); use --force", b, bdir / "documents.csv"); return
    t0 = time.time()
    device = embed.get_device(args.device)
    backend = M.detect_backend()
    log.info("topics: bin %d on backend %s (%s)", b, backend["backend"], backend)
    docs, modelled, full = bin_documents(args, paths, b)
    info = BINS.bin_info(b)
    emb = embed.embeddings_for(modelled, full, args.embedding_model, device)
    ids = modelled["id"].to_numpy().astype(str)

    # UMAP once per bin (replayed from umap_5d.npy on re-runs with the same documents)
    umap_path = bdir / "umap_5d.npy"
    if umap_path.exists() and (not args.force or args.keep_umap):     # --keep-umap: refit HDBSCAN only (a clustering decision)
        umap5d = np.load(umap_path)
        if len(umap5d) != len(modelled):
            umap5d = None
    else:
        umap5d = None
    if umap5d is None:
        umap5d = M.fit_umap(emb)
        np.save(umap_path, umap5d)

    # HDBSCAN with the topic-count guard; raw labels persisted, then renumbered by size (R1)
    guard = M.guard_min_cluster_size(umap5d, len(modelled), enabled=not args.max_docs, period_type=info["period_type"])
    np.save(bdir / "hdbscan_labels_raw.npy", guard["labels"])
    labels_sorted, size_map = M.renumber_by_size(guard["labels"])

    bt = M.build_topic_model(umap5d, labels_sorted, args.embedding_model, device)
    topics_raw, bt_map = M.fit_model(bt, modelled["document"].tolist(), emb)
    hdb_to_topic = {int(h): int(bt_map.get(int(s), int(s))) for h, s in size_map.items()}
    (bdir / "hdbscan_to_topic_map.json").write_text(json.dumps({"hdbscan_label_to_topic_id_raw": hdb_to_topic,
                                                                 "size_renumbering": size_map, "bertopic_mapping": bt_map}, indent=1), encoding="utf-8")
    np.save(bdir / "topic_embeddings.npy", np.asarray(bt.topic_embeddings_)[bt._outliers:])   # row t = topic t (N3)

    rep = M.representative_ids(bt, modelled["document"].tolist(), ids, topics_raw, emb)
    nearest = M.outlier_nearest(bt, emb, topics_raw, ids)
    nearest.to_csv(bdir / "outlier_nearest.csv", index=False)
    calib = M.calibration_table(nearest, len(modelled))
    cdir = bdir / "outlier_calibration"; cdir.mkdir(exist_ok=True)
    calib.to_csv(cdir / "calibration_table.csv", index=False)
    kw_of = {t: ", ".join(w for w, _ in (bt.get_topic(t) or [])[:C.SAMPLE_KEYWORDS]) for t in set(topics_raw.tolist()) - {-1}}
    titles = dict(zip(modelled["id"].astype(str), modelled["title"]))
    for T, samp in M.calibration_samples(nearest, titles, kw_of, calib).items():
        samp.to_csv(cdir / f"sample_T{T:.4f}.csv", index=False)

    T = M.resolve_threshold(b, args.outlier_threshold)
    topic_id, assignment, sim = M.apply_outlier_threshold(topics_raw, ids, nearest, T)
    write_bin_tables(bdir, bt, docs, modelled, topics_raw, topic_id, assignment, sim, rep, info["period_type"])
    M.save_model(bt, bdir / "model", args.embedding_model)

    meta = {"period_bin": b, "period_type": info["period_type"], "start": str(info["start"]), "end": str(info["end"]),
            "n_documents": int(len(docs)), "n_modelled": int(len(modelled)), "n_too_short": int(docs["too_short"].sum()),
            "debug_max_docs": args.max_docs or None, "embedding_model": args.embedding_model, "device": device, "backend": backend,
            "hdbscan": M.hdbscan_settings(info["period_type"]),
            "guard": {k: v for k, v in guard.items() if k != "labels"}, "n_topics_raw": int(M.n_topics_of(topics_raw)),
            "outlier_share_raw": round(float((topics_raw == -1).mean()), 4), "outlier_threshold": T,
            "outlier_share_final": round(float((topic_id == -1).mean()), 4), "n_reassigned": int((assignment == "reassigned").sum()),
            "calibration": calib.to_dict(orient="records"), "elapsed_min": round((time.time() - t0) / 60, 1),
            "run_at": dt.datetime.now().isoformat(timespec="seconds"), "versions": versions(), "prompts": P.all_prompts(),
            "config": C.effective_settings()}
    (bdir / "config.json").write_text(json.dumps({"topics": meta}, indent=1, default=str), encoding="utf-8")
    render_report(args, paths, b)
    tagging.write_status(bdir)
    log.info("topics: bin %d done in %.1f min — %d topics, %.1f %% raw outliers, T=%s -> %s", b, (time.time() - t0) / 60,
             meta["n_topics_raw"], 100 * meta["outlier_share_raw"], T, bdir)


def write_bin_tables(bdir, bt, docs, modelled, topics_raw, topic_id, assignment, sim, rep, period_type):
    """documents.csv (every post of the bin), topics.csv, topics_over_time.csv — the files that depend on T (R4)."""
    out = docs[["id", "created_datetime", "period_bin", "post_type", "lang", "num_comments", "too_short"]].copy()
    m = pd.DataFrame({"id": modelled["id"].to_numpy(), "topic_id_raw": topics_raw, "topic_id": topic_id,
                      "topic_assignment": assignment, "similarity": np.round(sim, 4)})
    out = out.merge(m, on="id", how="left")
    out["topic_assignment"] = out["topic_assignment"].fillna("not_modelled")
    for c in ("topic_id_raw", "topic_id"):
        out[c] = out[c].map(lambda x: "" if pd.isna(x) else str(int(x)))
    out["similarity"] = out["similarity"].map(lambda x: "" if pd.isna(x) else f"{x:.4f}")
    out.drop(columns=["too_short"]).to_csv(bdir / "documents.csv", index=False)
    M.topic_table(bt, modelled, topics_raw, topic_id, rep).to_csv(bdir / "topics.csv", index=False)
    M.topics_over_time(bt, modelled, topic_id, period_type).to_csv(bdir / "topics_over_time.csv", index=False)


# ---------------------------------------------------------------------------
# reassign: apply (a new) T from the persisted arrays, no refit
# ---------------------------------------------------------------------------
def stage_grid(args, paths: Paths, b: int):
    """Diagnostic of a failed large-bin gate (design §12.2): HDBSCAN over config.HDBSCAN_GRID on the bin's cached UMAP
    array; writes bin-N/hdbscan_grid.csv and the grid section of config.json. Nothing else in the bin changes."""
    bdir = paths.bin_dir(b)
    if not (bdir / "umap_5d.npy").exists():
        raise SystemExit(f"bin {b}: no umap_5d.npy — run --stage topics first")
    t0 = time.time()
    umap5d = np.load(bdir / "umap_5d.npy")
    grid = json.loads(args.grid) if args.grid else None
    ptype = BINS.bin_info(b)["period_type"]
    lo, hi = C.target_topic_range(len(umap5d))
    log.info("grid: bin %d (%s), %d points, density target %d–%d topics, backend %s", b, ptype, len(umap5d), lo, hi, M.detect_backend()["backend"])
    table = M.hdbscan_grid(umap5d, grid, period_type=ptype)
    table.to_csv(bdir / "hdbscan_grid.csv", index=False)
    log.info("grid: %d fits in %.1f min -> %s\n%s", len(table), (time.time() - t0) / 60, bdir / "hdbscan_grid.csv", table.to_string(index=False))
    update_config(bdir, "grid", {"target_topic_range": [lo, hi], "grid": {**C.HDBSCAN_GRID, **(grid or {})}, "n_fits": int(len(table)),
                                 "settings_in_force": M.hdbscan_settings(ptype),
                                 "backend": M.detect_backend(), "elapsed_min": round((time.time() - t0) / 60, 1),
                                 "run_at": dt.datetime.now().isoformat(timespec="seconds")})


def stage_reassign(args, paths: Paths, b: int):
    bdir = paths.bin_dir(b)
    cfg = read_config(bdir)
    if not (bdir / "documents.csv").exists():
        raise SystemExit(f"bin {b}: run --stage topics first")
    T = M.resolve_threshold(b, args.outlier_threshold)
    if T is None:
        raise SystemExit("reassign needs --outlier-threshold T or config.OUTLIER_THRESHOLD")
    device = embed.get_device(args.device)
    # the document set is whatever --stage topics modelled (documents.csv), so no --max-docs is needed here
    prev = pd.read_csv(bdir / "documents.csv", dtype=str, keep_default_na=False)
    docs_all, full = BINS.load_bin(b, paths.analysis_data)
    docs = docs_all[docs_all["id"].isin(set(prev["id"]))].reset_index(drop=True)
    modelled = docs[~docs["too_short"]].reset_index(drop=True)
    prev_mod = prev[prev["topic_id_raw"] != ""]
    if len(prev_mod) != len(modelled) or set(prev_mod["id"]) != set(modelled["id"]):
        raise SystemExit("reassign: documents.csv does not match the bin's documents (corpus changed since --stage topics?)")
    raw = prev_mod.set_index("id").loc[modelled["id"], "topic_id_raw"].astype(int).to_numpy()
    ids = modelled["id"].to_numpy().astype(str)
    nearest = pd.read_csv(bdir / "outlier_nearest.csv", dtype={"id": str})
    topic_id, assignment, sim = M.apply_outlier_threshold(raw, ids, nearest, T)
    bt = M.load_model(bdir / "model")
    bt.embedding_model = embed.load_embedding_model(args.embedding_model, device)
    old_topics = pd.read_csv(bdir / "topics.csv", dtype=str, keep_default_na=False)
    rep = {int(r["topic_id"]): [x for x in r["representative_ids"].split("|") if x] for _, r in old_topics.iterrows() if r["topic_id"] != "-1"}
    write_bin_tables(bdir, bt, docs, modelled, raw, topic_id, assignment, sim, rep, BINS.bin_info(b)["period_type"])
    update_config(bdir, "reassign", {"outlier_threshold": T, "n_reassigned": int((assignment == "reassigned").sum()),
                                     "outlier_share_final": round(float((topic_id == -1).mean()), 4),
                                     "run_at": dt.datetime.now().isoformat(timespec="seconds")})
    render_report(args, paths, b)
    tagging.write_status(bdir)
    log.info("reassign: bin %d at T=%.4f -> %d reassigned, %.1f %% outliers remain", b, T,
             int((assignment == "reassigned").sum()), 100 * float((topic_id == -1).mean()))


# ---------------------------------------------------------------------------
# S3a
# ---------------------------------------------------------------------------
_LLM = None


def get_llm(args):
    global _LLM
    if _LLM is None:
        _LLM = G.LLM(args.llm, embed.get_device(args.device), sampling=args.sampling)
    return _LLM


def stage_label(args, paths: Paths, b: int):
    bdir = paths.bin_dir(b)
    if not (bdir / "documents.csv").exists():
        raise SystemExit(f"bin {b}: run --stage topics first")
    t0 = time.time()
    docs, modelled, _ = bin_documents(args, paths, b)
    dd = pd.read_csv(bdir / "documents.csv", dtype=str, keep_default_na=False)
    dd = dd[dd["topic_id_raw"] != ""][["id", "topic_id_raw"]]
    dd["topic_id_raw"] = dd["topic_id_raw"].astype(int)
    md = modelled[["id", "document"]].merge(dd, on="id", how="inner")
    topics = pd.read_csv(bdir / "topics.csv", dtype={"topic_id": int, "n_cluster": int, "representative_ids": str,
                                                      "keywords": str}, keep_default_na=False)
    llm = get_llm(args)
    labelled = G.run_labelling(bdir, b, md, topics, llm, max_topics=args.max_topics)
    update_config(bdir, "label", {**llm.describe(), "prompt_version": P.PROMPT_VERSION, "n_topics_labelled": int(len(labelled)),
                                  "n_needs_review": int(labelled["needs_review"].sum()) if len(labelled) else 0,
                                  "agreement_modes": labelled["agreement_mode"].value_counts().to_dict() if len(labelled) else {},
                                  "round_iterations": labelled["round_iterations"].value_counts().sort_index().to_dict() if len(labelled) else {},
                                  "llm_stats": llm.stats, "max_topics": args.max_topics, "elapsed_min": round((time.time() - t0) / 60, 1),
                                  "run_at": dt.datetime.now().isoformat(timespec="seconds"), "prompts": P.stage_prompts("label")})
    render_report(args, paths, b)
    tagging.write_status(bdir)


# ---------------------------------------------------------------------------
# S4 / S5 / report
# ---------------------------------------------------------------------------
def stage_sentiment(args, paths: Paths):
    if (paths.analysis_data / "sentiment.csv").exists() and not args.force and not args.years:
        log.info("sentiment: %s exists (use --force)", paths.analysis_data / "sentiment.csv"); return
    years = args.years or data.available_years()
    pb = BINS.read_period_bins(paths.analysis_data) if (paths.analysis_data / "period_bins.csv").exists() else None
    sentiment.run_sentiment(years, paths.analysis_data, pb)


def stage_collect(args, paths: Paths):
    tagging.collect(paths.analysis, paths.analysis_data, args.allow_partial,
                    {"paths": paths.describe(), "versions": {k: v for k, v in versions().items() if k != "pip_freeze"}})


def render_report(args, paths: Paths, b: int):
    """report.html / report.md of a bin from its tables (topics.csv + the labels, when present); re-rendered after every
    stage that changes them."""
    bdir = paths.bin_dir(b)
    try:
        t = pd.read_csv(bdir / "topics.csv", keep_default_na=False)
        if (bdir / "topics_labelled.csv").exists():
            lab = pd.read_csv(bdir / "topics_labelled.csv", keep_default_na=False)
            t = t.merge(lab[["topic_id", "name", "emotion_tone", "description", "needs_review"]], on="topic_id", how="left")
            t[["name", "emotion_tone", "description"]] = t[["name", "emotion_tone", "description"]].fillna("")
        tot = pd.read_csv(bdir / "topics_over_time.csv", parse_dates=["period"])
        docs = pd.read_csv(bdir / "documents.csv", dtype=str, keep_default_na=False)
        sent_path = paths.analysis_data / "sentiment.csv"
        if sent_path.exists():
            sd = pd.read_csv(sent_path, dtype={"id": str}, keep_default_na=False).merge(docs[["id", "topic_id"]], on="id")
            sd = sd[sd["topic_id"] != ""]; sd["topic_id"] = sd["topic_id"].astype(int)
            st = sd.groupby("topic_id").agg(n_docs=("id", "size"), mean_compound=("post_sentiment", "mean")).reset_index()
        else:
            st = pd.DataFrame({"topic_id": t["topic_id"], "n_docs": t["n_total"], "mean_compound": 0.0})
        cfg = read_config(bdir).get("topics", {})
        meta = {k: cfg.get(k) for k in ("period_bin", "period_type", "start", "end", "n_documents", "n_modelled", "n_topics_raw",
                                         "outlier_share_raw", "outlier_threshold", "outlier_share_final", "embedding_model", "backend")}
        meta["run_name"] = f"bin-{b}"
        info = BINS.bin_info(b)
        markers = {"kick-off": C.WORLD_CUPS[info["world_cup"]][0], "final": C.WORLD_CUPS[info["world_cup"]][1]} if info["world_cup"] else None
        report.write_report(bdir, meta, t, tot, st, markers=markers)
    except Exception as e:
        log.warning("report for bin %d skipped: %s", b, e)


# ---------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True, choices=STAGES)
    p.add_argument("--bin", type=int, choices=range(1, 10)); p.add_argument("--bins", help="range, e.g. 1-9")
    p.add_argument("--device", default="auto"); p.add_argument("--embedding-model", default=C.EMBEDDING_MODEL)
    p.add_argument("--llm", default=C.LLM_MODEL); p.add_argument("--sampling", default="seeded" if C.LLM_SAMPLING_SEEDED else "greedy", choices=["greedy", "seeded"])
    p.add_argument("--max-docs", type=int, default=0, help="mechanics test on a subsample of the bin (disables the topic-count guard)")
    p.add_argument("--max-topics", type=int, default=0, help="label only the N largest topics (smoke tests)")
    p.add_argument("--outlier-threshold", type=float, default=None)
    p.add_argument("--years", type=int, nargs="*", help="sentiment: restrict to these years")
    p.add_argument("--out-root", default=None, help="dry runs: write <PATH>/results/analysis and <PATH>/data/analysis (input and caches unchanged)")
    p.add_argument("--allow-partial", action="store_true"); p.add_argument("--force", action="store_true")
    p.add_argument("--keep-umap", action="store_true")
    p.add_argument("--grid", default=None, help="grid stage: JSON overriding config.HDBSCAN_GRID keys")
    args = p.parse_args()
    os.environ.setdefault("HF_HOME", str(C.LLM_CACHE_DIR))   # every loader passes cache_dir explicitly; this keeps hub metadata there too
    paths = Paths(args.out_root)
    per_bin = args.stage in ("embed", "topics", "grid", "reassign", "label", "report")
    if per_bin:
        for b in parse_bins(args):
            setup_logging(paths.bin_dir(b) / "log.txt")
            log.info("=== stage %s, bin %d === args: %s | paths: %s", args.stage, b, vars(args), paths.describe())
            {"embed": stage_embed, "topics": stage_topics, "grid": stage_grid, "reassign": stage_reassign, "label": stage_label,
             "report": render_report}[args.stage](args, paths, b)
    elif args.stage == "status":
        print(tagging.status_table(paths.analysis).to_string(index=False))
    else:
        setup_logging(paths.analysis_data / "pipeline-log.txt")
        log.info("=== stage %s === args: %s | paths: %s", args.stage, vars(args), paths.describe())
        {"bins": stage_bins, "sentiment": stage_sentiment, "collect": stage_collect}[args.stage](args, paths)


if __name__ == "__main__":
    main()
