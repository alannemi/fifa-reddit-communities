#!/usr/bin/env python
"""
FIFA COMMUNITIES CHAPTER
5. TOPIC MODELLING (BERTopic) — command-line entry point
=========================================================

PURPOSE
-------
Fit a BERTopic topic model on r/soccer posts (one post = one document) for
  * one calendar year                     --run year   --year 2010
  * one World Cup tournament window       --run window --worldcup 2010   (kickoff - 2 months .. final + 2 months)
  * one World Cup era (aligned year models) --run era  --era 2010        (needs the year runs first)
and write tables, charts and a report to results/topic-models/<run-name>/.
Optionally label topics with a local open LLM (--label) and/or apply the
co-authors' manual merge table (--apply-merges).

Method and settings: code/topicmodel/config.py and
docs/session-02_2026-10-13/02_topic-modelling-design.md.

HOW TO RUN (from the repository root, inside the virtual environment)
--------------------------------------------------------------------
  .venv/bin/python code/05-topic-model.py --run window --worldcup 2010 --label
  .venv/bin/python code/05-topic-model.py --run year --year 2010
  .venv/bin/python code/05-topic-model.py --run era --era 2010
  .venv/bin/python code/05-topic-model.py --label-only results/topic-models/window-wc2010
  .venv/bin/python code/05-topic-model.py --run year --year 2010 --apply-merges   # after editing topic_merge_table.csv
Options: --device auto|cuda|mps|cpu   --embedding-model <hf id>   --llm <hf id>   --max-docs N (debug subsample)

OUTPUTS (per run directory)
---------------------------
documents.csv, topics.csv, topics_over_time.csv, sentiment_by_topic.csv,
sentiment_by_period.csv, topic_merge_table.csv, labels.jsonl (resumable),
report.html, report.md, model/ (saved BERTopic model), config.json, log.txt
"""
from __future__ import annotations
import argparse
import datetime as dt
import json
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))   # make `topicmodel` importable
from topicmodel import config as C
from topicmodel import data, embed, model as M, sentiment, report, label as L

log = logging.getLogger("topicmodel")


def setup_logging(run_dir: Path):
    run_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(run_dir / "log.txt", encoding="utf-8")])


def run_name_for(args) -> str:
    if args.run == "year":
        return f"year-{args.year}"
    if args.run == "window":
        return f"window-wc{args.worldcup}"
    return f"era-{args.era}"


# ---------------------------------------------------------------------------
# Document selection for each run type
# ---------------------------------------------------------------------------
def select_documents(args):
    """Return (docs for this run, {year: full-year docs} for the embedding cache, markers for charts)."""
    markers = {}
    if args.run == "year":
        years = [args.year]
        full = {y: data.build_documents(data.load_year(y)) for y in years}
        docs = full[args.year]
    elif args.run == "window":
        start, end = C.world_cup_window(args.worldcup)
        years = sorted({start.year, end.year})
        full = {y: data.build_documents(data.load_year(y)) for y in years}
        import pandas as pd
        docs = data.select_window(pd.concat(full.values(), ignore_index=True), start, end)
        kick, final = C.WORLD_CUPS[args.worldcup]
        markers = {"kickoff": kick, "final": final}
        log.info("window %s..%s: %d documents", start, end, len(docs))
    else:
        raise ValueError("era runs use run_era()")
    if args.max_docs and len(docs) > args.max_docs:
        docs = docs.sample(args.max_docs, random_state=C.SEED).reset_index(drop=True)
        log.info("DEBUG: subsampled to %d documents", len(docs))
    return docs, full, markers


# ---------------------------------------------------------------------------
# Fit + outputs for year / window runs
# ---------------------------------------------------------------------------
def run_fit(args, run_dir: Path):
    t0 = time.time()
    device = embed.get_device(args.device)
    docs, full, markers = select_documents(args)
    modelled = docs[~docs["too_short"]].reset_index(drop=True)
    log.info("documents: %d total, %d modelled, %d too short", len(docs), len(modelled), int(docs["too_short"].sum()))

    emb = embed.embeddings_for(modelled, full, args.embedding_model, device)
    bt = M.build_topic_model(len(modelled), args.embedding_model, device)
    topics, stats = M.fit_model(bt, modelled["document"].tolist(), emb)

    # Optional reduced model when there are very many topics (kept separately; main outputs use the unreduced model)
    if stats["n_topics_final"] > C.MAX_TOPICS_BEFORE_REDUCE:
        log.info("%d topics > %d: also saving a reduced model (nr_topics='auto')", stats["n_topics_final"], C.MAX_TOPICS_BEFORE_REDUCE)
        try:
            import copy
            red = copy.deepcopy(bt); red.reduce_topics(modelled["document"].tolist(), nr_topics="auto")
            M.save_model(red, run_dir / "model_reduced", args.embedding_model)
            stats["n_topics_reduced_model"] = int((red.get_topic_info()["Topic"] != -1).sum())
        except Exception as e:
            log.warning("reduced model failed: %s", e)

    # Assemble document table (too-short docs get topic -2 = not modelled)
    docs = docs.copy(); docs["topic"] = -2
    docs.loc[~docs["too_short"], "topic"] = topics
    sent = sentiment.score_documents(docs["document"].tolist())
    docs = docs.join(sent)

    topic_tbl = M.topic_table(bt, modelled, topics)
    tot = M.topics_over_time(bt, modelled, topics, args.run)
    sent_topic = sentiment.by_topic(docs[docs["topic"] != -2])
    sent_period = sentiment.by_period(docs[docs["topic"] != -2], args.run)

    if args.label:
        topic_tbl = do_label(args, run_dir, topic_tbl, modelled, topics, device)

    M.save_model(bt, run_dir / "model", args.embedding_model)
    report.write_tables(run_dir, docs, topic_tbl, tot, sent_topic, sent_period)
    L.write_merge_table(topic_tbl, run_dir / "topic_merge_table.csv")
    meta = {"run_name": run_dir.name, "run_type": args.run, "n_documents": len(docs), "n_modelled": len(modelled),
            "n_too_short": int(docs["too_short"].sum()), **stats, "embedding_model": args.embedding_model,
            "device": device, "llm_model": args.llm if args.label else None,
            "min_cluster_size": M.min_cluster_size(len(modelled)), "elapsed_min": round((time.time() - t0) / 60, 1),
            "versions": versions()}
    (run_dir / "config.json").write_text(json.dumps({**meta, "config": C.effective_settings()}, indent=1, default=str), encoding="utf-8")
    report.write_report(run_dir, meta, docs, topic_tbl, tot, sent_topic, bt_model=bt, markers=markers)
    log.info("done in %.1f min -> %s", (time.time() - t0) / 60, run_dir)


def do_label(args, run_dir: Path, topic_tbl, modelled, topics, device):
    labels = L.label_all(topic_tbl, modelled, topics, args.llm, device, progress_path=run_dir / "labels.jsonl")
    topic_tbl = topic_tbl.merge(labels, on="topic", how="left")
    topic_tbl.to_csv(run_dir / "topics.csv", index=False)
    return topic_tbl


def label_only(args, run_dir: Path):
    """Label (or resume labelling) an existing run without refitting."""
    import pandas as pd
    device = embed.get_device(args.device)
    topic_tbl = pd.read_csv(run_dir / "topics.csv", dtype={"representative_ids": str}, keep_default_na=False)
    topic_tbl = topic_tbl[[c for c in topic_tbl.columns if not c.startswith(("label_", "analyst_", "debate_", "analysts_", "needs_", "llm_"))]]
    docs = pd.read_csv(run_dir / "documents.csv", dtype=str, keep_default_na=False)
    docs = docs[~docs["topic"].isin(["-2"])].copy(); docs["topic"] = docs["topic"].astype(int)
    # rebuild document text from the corpus (documents.csv does not duplicate text)
    years = sorted(docs["year"].astype(int).unique())
    full = data.build_documents(data.load_years(years))
    modelled = full.set_index("id").loc[docs["id"]].reset_index()
    topic_tbl = do_label(args, run_dir, topic_tbl, modelled, docs["topic"].tolist(), device)
    L.write_merge_table(topic_tbl, run_dir / "topic_merge_table.csv")
    cfg = json.loads((run_dir / "config.json").read_text(encoding="utf-8")); cfg["llm_model"] = args.llm
    (run_dir / "config.json").write_text(json.dumps(cfg, indent=1, default=str), encoding="utf-8")
    import pandas as pd  # noqa
    tot = pd.read_csv(run_dir / "topics_over_time.csv", parse_dates=["period"])
    st = pd.read_csv(run_dir / "sentiment_by_topic.csv")
    meta = {k: v for k, v in cfg.items() if k != "config"}
    report.write_report(run_dir, meta, docs, topic_tbl, tot, st, bt_model=M.load_model(run_dir / "model"))


def apply_merges(args, run_dir: Path):
    """Collapse topics according to topic_merge_table.csv and recompute tables/report."""
    import pandas as pd
    mt = pd.read_csv(run_dir / "topic_merge_table.csv", dtype=str, keep_default_na=False)
    docs = pd.read_csv(run_dir / "documents.csv", dtype=str, keep_default_na=False)
    docs["topic"] = docs["topic"].astype(int)
    docs["sentiment_compound"] = docs["sentiment_compound"].astype(float)
    merged = L.apply_merges(docs["topic"].tolist(), mt)
    docs["topic_original"] = docs["topic"]; docs["topic"] = merged
    docs["date"] = pd.to_datetime(docs["created_datetime"], utc=True)
    topics = pd.read_csv(run_dir / "topics.csv", keep_default_na=False)
    names = {int(r["topic"]): r.get("final_name") or r.get("label_name", "") for _, r in mt.iterrows()}
    out_dir = run_dir / "merged"; out_dir.mkdir(exist_ok=True)
    sizes = docs[docs["topic"] >= -1].groupby("topic").size()
    tbl = topics[topics["topic"].isin(sizes.index)].copy()
    tbl["size"] = tbl["topic"].map(sizes); tbl["share"] = (tbl["size"] / sizes.sum()).round(5)
    tbl["label_name"] = tbl["topic"].map(names).fillna(tbl.get("label_name", ""))
    tbl.to_csv(out_dir / "topics.csv", index=False)
    docs.to_csv(out_dir / "documents.csv", index=False)
    sentiment.by_topic(docs[docs["topic"] != -2]).to_csv(out_dir / "sentiment_by_topic.csv", index=False)
    log.info("merges applied: %d -> %d topics; outputs in %s", topics["topic"].nunique(), tbl["topic"].nunique(), out_dir)


def run_era(args, run_dir: Path):
    """Align the year models of an era with BERTopic.merge_models and report per-year topic shares."""
    import pandas as pd
    years = C.ERAS[args.era]
    models, per_year = [], {}
    for y in years:
        ydir = C.RESULTS_DIR / f"year-{y}" / "model"
        if not ydir.exists():
            log.warning("year model missing for %d (%s); run `--run year --year %d` first", y, ydir, y); continue
        models.append(M.load_model(ydir)); per_year[y] = y
    if len(models) < 2:
        raise SystemExit("era run needs at least two fitted year models")
    merged = M.merge_year_models(models)
    M.save_model(merged, run_dir / "model", args.embedding_model)
    device = embed.get_device(args.device)
    rows = []
    for y in per_year:
        docs = data.build_documents(data.load_year(y)); mod = docs[~docs["too_short"]].reset_index(drop=True)
        emb = embed.embeddings_for(mod, {y: docs}, args.embedding_model, device)
        t, _ = merged.transform(mod["document"].tolist(), emb)
        s = pd.Series(t).value_counts()
        for topic, n in s.items():
            rows.append({"era": args.era, "year": y, "topic": int(topic), "n_docs": int(n), "share": round(n / len(mod), 5)})
    shares = pd.DataFrame(rows)
    info = merged.get_topic_info()[["Topic", "Count", "Name"]].rename(columns={"Topic": "topic", "Count": "size", "Name": "name"})
    shares.to_csv(run_dir / "topic_shares_by_year.csv", index=False)
    info.to_csv(run_dir / "topics.csv", index=False)
    (run_dir / "config.json").write_text(json.dumps({"run_name": run_dir.name, "run_type": "era", "years": years,
                                                     "n_topics": int((info["topic"] != -1).sum()), "config": C.effective_settings()}, indent=1, default=str))
    log.info("era %d: %d aligned topics across %s", args.era, int((info["topic"] != -1).sum()), years)


def versions() -> dict:
    from importlib.metadata import version
    return {"python": sys.version.split()[0], **{pkg: version(pkg) for pkg in
            ("bertopic", "sentence-transformers", "torch", "transformers", "umap-learn", "hdbscan", "scikit-learn", "vaderSentiment")}}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", choices=["year", "window", "era"])
    p.add_argument("--year", type=int); p.add_argument("--worldcup", type=int, choices=sorted(C.WORLD_CUPS)); p.add_argument("--era", type=int, choices=sorted(C.ERAS))
    p.add_argument("--label", action="store_true", help="label topics with the local LLM after fitting")
    p.add_argument("--label-only", metavar="RUN_DIR", help="label an existing run (resumable)")
    p.add_argument("--apply-merges", action="store_true", help="apply topic_merge_table.csv of the run and recompute")
    p.add_argument("--device", default="auto"); p.add_argument("--embedding-model", default=C.EMBEDDING_MODEL)
    p.add_argument("--llm", default=C.LLM_MODEL); p.add_argument("--max-docs", type=int, default=0)
    args = p.parse_args()
    if args.label_only:
        run_dir = Path(args.label_only); setup_logging(run_dir); label_only(args, run_dir); return
    if not args.run:
        p.error("--run is required (or --label-only)")
    run_dir = C.RESULTS_DIR / run_name_for(args); setup_logging(run_dir)
    log.info("=== %s === args: %s", run_dir.name, vars(args))
    if args.apply_merges:
        apply_merges(args, run_dir)
    elif args.run == "era":
        run_era(args, run_dir)
    else:
        run_fit(args, run_dir)


if __name__ == "__main__":
    main()
