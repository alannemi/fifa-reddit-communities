#!/usr/bin/env python
"""
Unit checks for the analytical pipeline — no GPU, no language model (session 05, review R6; extended in sessions 06–08).
Run:  .venv/bin/python code/tests/test_pipeline_units.py        (exit code 1 if any check fails)
Covers: the path root (PIPELINE_ROOT) and the tone vocabulary; period-bin boundaries; the topic-density guard (D8), the
HDBSCAN grid diagnostic and the per-period-type settings; the frozen UMAP/cluster components inside BERTopic and the size
renumbering; JSON extraction, label validation and normalisation; the topic overrides (incl. merge chains); outlier
thresholds; the full D2 debate engine with a scripted StubLLM (every agreement_mode, the repair and no-answer paths,
resume); the partial S5 collect with the per-topic VADER mean and the topic crosswalk template; JSON salvage (N11);
sentiment labels. The taxonomy stage and its tests were removed by decision D7 (2026-10-25; git history keeps them).
"""
from __future__ import annotations
import json
import re
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from topicmodel import config as C, bins as BINS, model as M, agents as G, prompts as P, sentiment as S, tagging as T

RESULTS = []


def check(name, fn):
    try:
        fn()
        RESULTS.append((name, True, "")); print(f"[PASS] {name}")
    except Exception as e:
        RESULTS.append((name, False, repr(e))); print(f"[FAIL] {name}: {e!r}"); traceback.print_exc()


# ---------------------------------------------------------------------------
def test_paths_and_tones():
    # repository layout (PIPELINE_ROOT unset)
    p = C.resolve_paths(None)
    assert p["YEARLY_DIR"] == C.ROOT / "data" / "yearly datasets" and p["ANALYSIS_DATA_DIR"] == C.ROOT / "data" / "analysis"
    assert p["ANALYSIS_DIR"] == C.ROOT / "results" / "analysis" and p["EMBED_CACHE_DIR"] == C.ROOT / "results" / "embeddings"
    assert p["LLM_CACHE_DIR"] == C.ROOT / ".hf-cache"
    # the flat Drive layout (decision H2), with and without the local copy of the yearly files (H3)
    r = (Path(tempfile.gettempdir()) / "fifa-communities").resolve()
    q = C.resolve_paths(str(r))
    assert q["YEARLY_DIR"] == r / "yearly datasets" and q["ANALYSIS_DATA_DIR"] == r / "analysis" and q["ANALYSIS_DIR"] == r / "results"
    assert q["EMBED_CACHE_DIR"] == r / "embeddings" and q["LLM_CACHE_DIR"] == r / "hf-cache"
    q = C.resolve_paths(str(r), str(Path(tempfile.gettempdir()) / "content" / "yearly datasets"))
    assert q["YEARLY_DIR"].name == "yearly datasets" and q["YEARLY_DIR"] != r / "yearly datasets" and q["ANALYSIS_DIR"] == r / "results"
    q = C.resolve_paths(str(r), None, str(Path(tempfile.gettempdir()) / "hf-cache"))
    assert q["LLM_CACHE_DIR"].name == "hf-cache" and q["LLM_CACHE_DIR"] != r / "hf-cache" and q["EMBED_CACHE_DIR"] == r / "embeddings"
    s = C.effective_settings()
    assert s["PIPELINE_ROOT"] == C.PIPELINE_ROOT and s["ANALYSIS_DIR"] == str(C.ANALYSIS_DIR) and "EMBEDDING_MODEL_HQ" not in s
    # tone vocabulary: every stem maps into TONE_SET; the mapper is unchanged
    assert {cat for _, cat in C.TONE_STEMS} <= set(C.TONE_SET)
    assert [G.tone_category(x) for x in ("Frustration and anger", "anxious and desperate", "Satirical and humorous", "meh")] == ["frustrated", "anxious", "humorous", "other"]


def test_bin_boundaries():
    s = pd.Series(["2010-01-01 00:00:00", "2010-09-30 23:59:59", "2010-10-01 00:00:00", "2014-01-31 23:59:59",
                   "2014-02-01 00:00:00", "2018-09-30 12:00:00", "2018-10-01 00:00:00", "2022-06-30 23:59:59",
                   "2022-07-01 00:00:00", "2023-02-28 23:59:59", "2023-03-01 00:00:00", "2026-01-31 23:59:59",
                   "2026-02-01 00:00:00", "2026-07-31 23:59:59"])
    assert BINS.bin_of(s).tolist() == [1, 1, 2, 2, 3, 5, 6, 6, 7, 7, 8, 8, 9, 9]
    assert BINS.bin_years(7) == [2022, 2023] and BINS.bin_years(1) == [2010] and BINS.bin_years(8) == [2023, 2024, 2025, 2026]
    try:
        BINS.bin_of(pd.Series(["2009-12-31 23:59:59"])); raise AssertionError("should raise")
    except ValueError:
        pass
    assert sum(C.PERIOD_BIN_EXPECTED_N.values()) == 1302021


def test_guard():
    # the density target (decision D8): 50–75 raw topics per 10,000 modelled posts
    assert C.target_topic_range(10000) == (50, 75) and C.target_topic_range(9809) == (49, 74) and C.target_topic_range(331216) == (1656, 2484)
    X = np.zeros((10000, 5))
    def stub(X, m):                               # n_topics = 2000 / m: floor 25 -> 80 (too many) -> 38 -> 52, in range
        k = int(2000 / m); n = len(X)
        return np.array([(i % (k + 1)) - 1 for i in range(n)])
    r = M.guard_min_cluster_size(X, 10000, True, fit=stub)
    assert r["converged"] and [t["min_cluster_size"] for t in r["trials"]] == [25, 38] and r["n_topics"] == 52 and r["min_cluster_size"] == 38
    X4 = np.zeros((4000, 5))                      # target 20–30
    def stub2(X, m):
        k = int(6000 / m); n = len(X)             # 25 -> 240 topics (too many) -> 38 -> 157 -> 57 -> 105 -> 86 -> 69 -> ...
        return np.array([(i % (k + 1)) - 1 for i in range(n)])
    r = M.guard_min_cluster_size(X4, 4000, True, fit=stub2)
    assert len(r["trials"]) <= C.GUARD_MAX_TRIALS, r["trials"]
    lo, hi = C.target_topic_range(4000); ks = [t["n_topics"] for t in r["trials"]]
    assert r["n_topics"] == min(ks, key=lambda k: 0 if lo <= k <= hi else min(abs(k - lo), abs(k - hi))), (r, ks)
    def stub3(X, m):                              # never enough topics: stays at the floor, flagged
        return np.array([(i % 11) - 1 for i in range(len(X))])
    r = M.guard_min_cluster_size(X4, 4000, True, fit=stub3)
    assert not r["converged"] and r["min_cluster_size"] == C.MIN_CLUSTER_FLOOR
    r = M.guard_min_cluster_size(X4, 4000, False, fit=stub2)   # disabled: one fit
    assert len(r["trials"]) == 1 and r["enabled"] is False
    # the gate diagnostic: HDBSCAN over the grid on the cached array (stub: topics = 6000 / mcs, fewer outliers with lower min_samples)
    def stub4(X, mcs, ms, method, eps):
        k = max(1, int(6000 / mcs)); n = len(X); n_out = int(n * ms / 20)
        lab = np.array([i % k for i in range(n)]); lab[:n_out] = -1
        return lab
    g = M.hdbscan_grid(X, {"min_cluster_size": [100, 50], "min_samples": [10, 2], "method": ["eom", "leaf"], "epsilon": [0.0]}, fit=stub4)
    assert len(g) == 8 and list(g.columns) == ["min_cluster_size", "min_samples", "method", "epsilon", "n_topics", "outlier_share", "largest_share", "median_size", "in_range", "seconds"]
    row = g[(g.min_cluster_size == 100) & (g.min_samples == 10) & (g.method == "eom")].iloc[0]
    assert row.n_topics == 60 and abs(row.outlier_share - 0.5) < 1e-6 and bool(row.in_range) is True
    row = g[(g.min_cluster_size == 50) & (g.min_samples == 2)].iloc[0]
    assert row.n_topics == 120 and abs(row.outlier_share - 0.1) < 1e-6 and bool(row.in_range) is False
    assert M.cluster_stats(np.array([-1, -1, 0, 0, 0, 1])) == {"n_topics": 2, "outlier_share": 0.3333, "largest_share": 0.5, "median_size": 2}
    # settings by period type (the decision space of a failed gate)
    assert C.setting(10) == 10 and C.setting({"tournament": 10, "interim": 5}, "interim") == 5 and C.setting({"default": 7, "interim": 5}, "tournament") == 7
    saved = C.HDBSCAN_MIN_SAMPLES
    try:
        C.HDBSCAN_MIN_SAMPLES = {"tournament": 10, "interim": 5}
        assert M.hdbscan_settings("interim")["min_samples"] == 5 and M.hdbscan_settings("tournament")["min_samples"] == 10
        assert M.hdbscan_settings("interim")["topics_per_10k"] == [50, 75]
    finally:
        C.HDBSCAN_MIN_SAMPLES = saved


def test_frozen_components_in_bertopic():
    from bertopic import BERTopic
    from sklearn.feature_extraction.text import CountVectorizer
    rng = np.random.RandomState(0)
    sizes = [120, 80, 40]
    emb, labels, docs = [], [], []
    vocab = [["vuvuzela", "noise", "horn", "stadium"], ["referee", "penalty", "offside", "var"], ["transfer", "fee", "contract", "signing"]]
    for k, n in enumerate(sizes):
        centre = np.zeros(16); centre[k] = 5
        emb.append(centre + rng.normal(0, 0.3, (n, 16)))
        labels += [k] * n
        docs += [" ".join(rng.choice(vocab[k], 3)) + " post" for _ in range(n)]
    emb = np.vstack(emb).astype(np.float32); emb /= np.linalg.norm(emb, axis=1, keepdims=True)
    labels = np.array(labels); labels[:10] = -1                 # some outliers in the biggest cluster
    hdb = np.array([{0: 2, 1: 0, 2: 1, -1: -1}[l] for l in labels])   # HDBSCAN-style arbitrary numbering
    renum, mp = M.renumber_by_size(hdb)
    assert mp == {2: 0, 0: 1, 1: 2, -1: -1} and (renum == labels).all()
    umap5d = rng.normal(size=(len(docs), 5)).astype(np.float32)
    bt = BERTopic(embedding_model=None, umap_model=M.FrozenUMAP(umap5d), hdbscan_model=M.FrozenCluster(renum),
                  vectorizer_model=CountVectorizer(min_df=1), calculate_probabilities=False, verbose=False)
    topics, mapping = M.fit_model(bt, docs, emb)
    assert (topics == renum).all(), "BERTopic changed the injected labels"
    assert all(k == v for k, v in mapping.items()), mapping
    assert bt._outliers == 1 and np.asarray(bt.topic_embeddings_).shape == (4, 16)
    ids = np.array([f"p{i}" for i in range(len(docs))])
    rep = M.representative_ids(bt, docs, ids, topics, emb, n=5)
    assert set(rep) == {0, 1, 2} and all(1 <= len(v) <= 5 for v in rep.values())
    near = M.outlier_nearest(bt, emb, topics, ids)
    assert len(near) == 10 and (near["nearest_topic"] == 0).all()
    tid, ass, sim = M.apply_outlier_threshold(topics, ids, near, T=0.0)
    assert (ass[:10] == "reassigned").all() and (tid[:10] == 0).all()
    tid, ass, sim = M.apply_outlier_threshold(topics, ids, near, T=2.0)
    assert (ass[:10] == "outlier").all() and (tid[:10] == -1).all() and (ass[10:] == "cluster").all()
    tid, ass, sim = M.apply_outlier_threshold(topics, ids, near, T=None)
    assert (ass[:10] == "outlier").all() and not np.isnan(sim[:10]).any()
    tbl = M.calibration_table(near, len(docs))
    assert list(tbl["quantile"]) == list(C.OUTLIER_CALIBRATION_QUANTILES)
    docs_df = pd.DataFrame({"id": ids, "title": docs, "post_type": "regular", "lang": "en", "num_comments": "3"})
    tt = M.topic_table(bt, docs_df, topics, tid, rep)
    assert set(tt["topic_id"]) == {-1, 0, 1, 2} and tt.loc[tt["topic_id"] == 0, "n_cluster"].iloc[0] == 110


def test_json_and_validation():
    obj, err = G.extract_json('Here you go:\n```json\n{"decision": "revise", "categories": [{"name": "A"}]}\n```')
    assert err is None and obj["categories"][0]["name"] == "A"
    assert G.extract_json("no json here")[0] is None
    assert G.extract_json('{"decision": "revise", "categories": [')[0] is None
    full = {"decision": "revise", "categories": [{"name": "Referee criticism", "purpose": "complain", "emotion_tone": "Frustrated",
                                                  "description": "Posts attack referees.", "inclusion_cues": "ref; var; penalty"}]}
    p, e = G.validate_label(full, False)
    assert e is None and p["categories"][0]["inclusion_cues"] == ["ref", "var", "penalty"]
    assert G.validate_label({"decision": "accept"}, False)[0] is None and G.validate_label({"decision": "accept"}, True)[0] == {"decision": "accept"}
    assert G.validate_label({"categories": []}, False)[0] is None
    many = {"categories": [{**full["categories"][0], "inclusion_cues": [f"cue {i}" for i in range(15)]}]}
    p, e = G.validate_label(many, False)                       # over-long cue lists are kept (first 10), not rejected (N12)
    assert e is None and len(p["categories"][0]["inclusion_cues"]) == 10 and p["categories"][0]["n_cues_given"] == 15
    assert G.validate_label({"categories": [{**full["categories"][0], "inclusion_cues": []}]}, False)[0] is None
    assert G.validate_label({"categories": [full["categories"][0], full["categories"][0]]}, False)[0] is None


def test_label_identity():
    a = {"decision": "revise", "categories": [{"name": "Referee Criticism!", "purpose": "To complain", "emotion_tone": "frustrated",
                                               "description": "Posts attack referees.", "inclusion_cues": ["VAR", "penalty"]}]}
    b = json.loads(json.dumps(a)); b["categories"][0].update(name="referee criticism", inclusion_cues=["penalty", "var"])
    assert G.labels_identical(a, b)
    b["categories"][0]["name"] = "Referee praise"
    assert not G.labels_identical(a, b) and not G.labels_identical(a, None)


def test_overrides():
    lab = pd.DataFrame({"topic_id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "keywords": ["k"] * 4})
    ov = pd.DataFrame({"topic_id": ["3", "2", "4"], "final_name": ["", "", "D final"], "merge_into": ["2", "1", ""], "notes": ["", "", ""]})
    out, merges = G.apply_topic_overrides(lab, ov)
    assert merges == {3: 1, 2: 1} and out["topic_id"].tolist() == [1, 4]
    assert out.loc[out["topic_id"] == 4, "name"].iloc[0] == "D final" and out.loc[out["topic_id"] == 4, "name_agent"].iloc[0] == "d"


def label_json(name, tone="excited", cues=("a", "b", "c")):
    return json.dumps({"decision": "revise", "categories": [{"name": name, "purpose": "purpose", "emotion_tone": tone,
                                                            "description": "desc.", "inclusion_cues": list(cues)}]})


class StubLLM:
    def __init__(self, script: dict, default=None, taxonomy_fn=None):
        self.script, self.default, self.taxonomy_fn = script, default, taxonomy_fn
        self.calls, self.users = [], []
        self.stats = {"calls": 0, "prompts": 0, "seconds": 0.0, "tokens_out": 0, "truncated": 0}

    def describe(self):
        return {"llm_model": "stub", "llm_revision": None, "device": "cpu", "dtype": "none", "decoding": "greedy", "batch_size": 1}

    def generate(self, requests, max_new_tokens, seeds=None, meta=None):
        out = []
        for (sysm, user), m in zip(requests, meta):
            self.calls.append(m); self.users.append((m, user))
            key = (m["key"], m["analyst"], m["round"], m["repair"])
            if key in self.script:
                text = self.script[key]
            elif self.taxonomy_fn is not None:
                text = self.taxonomy_fn(user, m)
            elif self.default is not None:
                text = self.default(m)
            else:
                text = label_json(f"default {m['key']} {m['analyst']} r{m['round']}")
            if callable(text):
                text = text(user)
            trunc = text == "TRUNC" or text.startswith("TRUNC:")          # "TRUNC:<text>" = a cut-off answer with that text
            out.append({"text": text[6:] if text.startswith("TRUNC:") else text, "truncated": trunc, "n_tokens": 10, "seed": None, "seconds": 0.0})
        self.stats["prompts"] += len(out)
        return out


def test_debate_modes():
    A, B = P.AGENTS
    s = {}
    s[("t1", A, 1, False)] = label_json("Vuvuzela noise"); s[("t1", B, 1, False)] = label_json("vuvuzela noise!", cues=("c", "b", "a"))
    s[("t2", A, 1, False)] = label_json("A2"); s[("t2", B, 1, False)] = label_json("B2"); s[("t2", A, 2, False)] = '{"decision": "accept"}'
    s[("t3", A, 1, False)] = label_json("A3"); s[("t3", B, 1, False)] = label_json("B3"); s[("t3", A, 2, False)] = label_json("A3 new"); s[("t3", B, 2, False)] = ' {"decision":"ACCEPT"} '
    s[("t4", A, 1, False)] = label_json("A4"); s[("t4", B, 1, False)] = label_json("B4"); s[("t4", A, 2, False)] = label_json("A4")
    s[("t5", A, 1, False)] = "garbage"; s[("t5", A, 1, True)] = "still garbage"; s[("t5", B, 1, False)] = label_json("B5")
    for r in range(1, 6):
        s[("t6", A, r, False)] = label_json(f"A6 r{r}"); s[("t6", B, r, False)] = label_json(f"B6 r{r}")
    s[("t7", A, 1, False)] = '{"decision": "accept"}'; s[("t7", A, 1, True)] = label_json("Seven"); s[("t7", B, 1, False)] = label_json("seven")
    s[("t8", A, 1, False)] = label_json("A8"); s[("t8", B, 1, False)] = label_json("B8"); s[("t8", A, 2, False)] = label_json("B8")
    s[("t9", A, 1, False)] = "TRUNC"; s[("t9", A, 1, True)] = label_json("Nine"); s[("t9", B, 1, False)] = label_json("nine")
    llm = StubLLM(s)
    items = [{"key": k, "items_block": f"ITEMS {k}"} for k in ["t1", "t2", "t3", "t4", "t5", "t6", "t7", "t8", "t9"]]
    res = G.Debate("label", llm, lambda o, it, acc: G.validate_label(o, acc), G.labels_identical).run(items)
    def chk(k, mode, rnd, agreed, name=None):
        r = res[k]; assert (r["agreement_mode"], r["round_iterations"], r["agreed"]) == (mode, rnd, agreed), (k, r["agreement_mode"], r["round_iterations"], r["agreed"])
        if name is not None:
            assert r["final"]["categories"][0]["name"] == name, (k, r["final"])
        assert r["needs_review"] == (not agreed)
    chk("t1", "identical", 1, True, "Vuvuzela noise")
    chk("t2", "accepted_by_A", 2, True, "B2")
    chk("t3", "accepted_by_B", 2, True, "A3 new")
    chk("t4", "stalemate", 2, False, "A4")
    chk("t5", "no_answer", 1, False, "B5")
    chk("t6", "cap", 5, False, "A6 r5")
    chk("t7", "identical", 1, True, "Seven")
    chk("t8", "identical", 2, True, "B8")
    chk("t9", "identical", 1, True, "Nine")
    tr = res["t7"]["transcript"]
    assert [t["repair"] for t in tr] == [False, True, False] and "only valid in a revision turn" in tr[0]["error"]
    assert "cut off" in res["t9"]["transcript"][0]["error"]
    assert res["t6"]["analyst_b_final"]["categories"][0]["name"] == "B6 r5"
    # batch composition: round 1 asks all nine items for A, then all nine for B, in key order
    first = [c for c in llm.calls if c["round"] == 1 and not c["repair"]]
    assert [c["key"] for c in first[:9]] == [f"t{i}" for i in range(1, 10)] and first[0]["analyst"] == A and first[9]["analyst"] == B


def test_labelling_resume_and_outputs():
    tmp = Path(tempfile.mkdtemp())
    try:
        docs = pd.DataFrame({"id": [f"p{i}" for i in range(12)], "document": [f"post {i} text" for i in range(12)],
                             "topic_id_raw": [0] * 5 + [1] * 4 + [2] * 3})
        topics = pd.DataFrame({"topic_id": [-1, 0, 1, 2], "n_cluster": [0, 5, 4, 3], "keywords": ["", "k0", "k1", "k2"],
                               "representative_ids": ["", "p0|p1", "p5", "p9"]})
        A, B = P.AGENTS
        s = {(0, A, 1, False): label_json("Zero"), (0, B, 1, False): label_json("zero"),
             (1, A, 1, False): label_json("One A"), (1, B, 1, False): label_json("One B"), (1, A, 2, False): '{"decision": "accept"}'}
        llm = StubLLM(s)
        out = G.run_labelling(tmp, 1, docs, topics, llm, max_topics=2)
        assert out["topic_id"].tolist() == [0, 1] and out["name"].tolist() == ["Zero", "One B"]
        assert (tmp / "labels" / "topic_0.json").exists() and (tmp / "topic_overrides.csv").exists()
        n_calls = len(llm.calls)
        out2 = G.run_labelling(tmp, 1, docs, topics, llm, max_topics=2)       # resume: nothing to do
        assert len(llm.calls) == n_calls and out2["name"].tolist() == ["Zero", "One B"]
        llm.script.update({(2, A, 1, False): label_json("Two"), (2, B, 1, False): label_json("two")})
        out3 = G.run_labelling(tmp, 1, docs, topics, llm)                     # the remaining topic (2) only
        assert out3["topic_id"].tolist() == [0, 1, 2] and len(llm.calls) == n_calls + 2, (out3["topic_id"].tolist(), len(llm.calls) - n_calls)
        rec = json.loads((tmp / "labels" / "topic_1.json").read_text())
        assert rec["agreement_mode"] == "accepted_by_A" and rec["round_iterations"] == 2 and rec["n_sample_posts"] == 4
        assert rec["sample_ids"][0] == "p5" and "ITEMS" not in rec["transcript"][0]["user"] and "Keywords: k1" in rec["transcript"][0]["user"]
        cols = pd.read_csv(tmp / "topics_labelled.csv").columns
        assert "n_total" not in cols and "name" in cols and "round_iterations" in cols
    finally:
        shutil.rmtree(tmp)


def test_collect_partial():
    tmp = Path(tempfile.mkdtemp())
    try:
        an, ad = tmp / "results" / "analysis", tmp / "data" / "analysis"
        (an / "bin-1").mkdir(parents=True); ad.mkdir(parents=True)
        ids = [f"p{i}" for i in range(10)]
        pd.DataFrame({"id": ids, "period_bin": [1] * 6 + [2] * 4, "period_type": ["tournament"] * 6 + ["interim"] * 4}).to_csv(ad / "period_bins.csv", index=False)
        pd.DataFrame({"id": ids, "post_sentiment": [0.5, -0.5, 0.0, 0.1, -0.1, 0.0, 0.3, 0.0, 0.0, 0.9], "post_sentiment_full": [0.0] * 10,
                      "post_sentiment_label": ["positive", "negative", "neutral", "positive", "negative", "neutral", "positive", "neutral", "neutral", "positive"]}).to_csv(ad / "sentiment.csv", index=False)
        b = an / "bin-1"
        pd.DataFrame({"id": ids[:6], "created_datetime": ["2010-06-01 00:00:00"] * 6, "period_bin": [1] * 6, "post_type": ["regular"] * 6, "lang": ["en"] * 6,
                      "num_comments": ["1"] * 6, "topic_id_raw": ["0", "0", "1", "-1", "-1", ""], "topic_id": ["0", "0", "1", "1", "-1", ""],
                      "topic_assignment": ["cluster", "cluster", "cluster", "reassigned", "outlier", "not_modelled"], "similarity": ["", "", "", "0.6", "0.2", ""]}).to_csv(b / "documents.csv", index=False)
        pd.DataFrame({"topic_id": [-1, 0, 1], "n_cluster": [2, 2, 1], "n_total": [1, 2, 2], "share": [0.2, 0.4, 0.4], "keywords": ["", "a, b", "c, d"], "representative_ids": ["", "p0", "p2"]}).to_csv(b / "topics.csv", index=False)
        pd.DataFrame({"topic_id": [0, 1], "keywords": ["a, b", "c, d"], "name": ["Zero", "One"], "purpose": ["p", "p"], "emotion_tone": ["excited", "sad"], "tone_category": ["excited", "sad"],
                      "description": ["d", "d"], "inclusion_cues": ["x", "y"], "round_iterations": [1, 2], "agreed": [True, False], "agreement_mode": ["identical", "cap"],
                      "needs_review": [False, True], "analyst_a_final": ["{}", "{}"], "analyst_b_final": ["{}", "{}"], "llm_model": ["stub"] * 2, "llm_revision": ["", ""],
                      "prompt_version": ["1.4"] * 2, "decoding": ["greedy"] * 2}).to_csv(b / "topics_labelled.csv", index=False)
        pd.DataFrame({"topic_id": [1], "name": ["One"], "needs_review": [True], "final_name": ["One (final)"], "merge_into": [""], "notes": [""]}).to_csv(b / "topic_overrides.csv", index=False)
        side = T.collect(an, ad, allow_partial=True, run_info={})
        pv = pd.read_csv(ad / "post_variables.csv", dtype=str, keep_default_na=False)
        assert list(pv.columns) == T.POST_COLUMNS + T.AUDIT_COLUMNS and len(pv) == 10 and "category_id" not in pv.columns
        assert pv["topic_id"].tolist()[:6] == ["0", "0", "1", "1", "-1", ""] and pv["topic_id"].tolist()[6:] == [""] * 4
        assert pv["post_sentiment"].tolist()[0] == "0.5000" and pv["topic_id_raw"].tolist()[3] == "-1"
        assert pv["topic_name"].tolist()[:6] == ["Zero", "Zero", "One (final)", "One (final)", "", ""] and pv["topic_tone"].tolist()[:4] == ["excited", "excited", "sad", "sad"]
        assert pv["topic_uid"].tolist()[:6] == ["1-0", "1-0", "1-1", "1-1", "", ""]
        assert side["bins_complete"] == [1] and side["complete"] is False and side["errors"] == [] and side["n_topics_labelled"] == 2
        lab = pd.read_csv(an / "topic_labels_all_bins.csv", dtype=str, keep_default_na=False)
        assert list(lab.columns) == T.LABEL_COLUMNS and lab.loc[lab["topic_id"] == "1", "name"].iloc[0] == "One (final)" and lab["n_total"].tolist() == ["2", "2"]
        assert lab["topic_uid"].tolist() == ["1-0", "1-1"]
        # the mean VADER score per topic (title score of the topic's posts after reassignment): topic 0 = p0, p1; topic 1 = p2, p3
        assert lab["mean_vader_sentiment"].tolist() == ["0.0000", "0.0500"] and lab["n_posts_scored"].tolist() == ["2", "2"]
        st = T.write_status(b)                                                            # the per-bin flag for the review
        assert st["state"] == "labelled" and st["n_topics"] == 2 and st["n_labelled"] == 2 and st["review_ready"] is True and st["n_needs_review"] == 1
        assert (b / "REVIEW_READY.txt").exists() and json.loads((b / "status.json").read_text())["review_ready"] is True
        assert T.status_table(an).set_index("bin").loc[1, "state"] == "labelled" and T.status_table(an).set_index("bin").loc[2, "state"] == "missing"
        xw = pd.read_csv(an / "topic_crosswalk.csv", keep_default_na=False)
        assert list(xw.columns) == T.CROSSWALK_COLUMNS and xw["topic_id"].tolist() == [0, 1] and (xw["chapter_theme"] == "").all() and xw["period_type"].tolist() == ["tournament"] * 2
        assert xw["topic_uid"].tolist() == ["1-0", "1-1"]
        xw.loc[0, "chapter_theme"] = "Match talk"; xw.to_csv(an / "topic_crosswalk.csv", index=False)     # a filled crosswalk is kept
        # a topic merge: 1 into 0 -> the four posts carry topic 0, the label row of 1 disappears, the mean covers four posts
        pd.DataFrame({"topic_id": [1], "name": ["One"], "needs_review": [True], "final_name": [""], "merge_into": ["0"], "notes": [""]}).to_csv(b / "topic_overrides.csv", index=False)
        side = T.collect(an, ad, allow_partial=True, run_info={})
        pv = pd.read_csv(ad / "post_variables.csv", dtype=str, keep_default_na=False)
        assert pv["topic_id"].tolist()[:4] == ["0"] * 4 and pv["topic_id_raw"].tolist()[2] == "1" and side["errors"] == []
        assert pv["topic_name"].tolist()[:4] == ["Zero"] * 4 and pv["topic_tone"].tolist()[:4] == ["excited"] * 4
        lab = pd.read_csv(an / "topic_labels_all_bins.csv", dtype=str, keep_default_na=False)
        assert lab["topic_id"].tolist() == ["0"] and lab["merged_from"].tolist() == ["1"] and lab["n_total"].tolist() == ["4"]
        assert lab["mean_vader_sentiment"].tolist() == ["0.0250"] and lab["n_posts_scored"].tolist() == ["4"]
        xw = pd.read_csv(an / "topic_crosswalk.csv", keep_default_na=False)
        assert len(xw) == 2 and xw.loc[xw["topic_id"] == 0, "chapter_theme"].iloc[0] == "Match talk"
        # a post whose topic is not in topics.csv fails the S5 check (tolerated only with --allow-partial)
        d = pd.read_csv(b / "documents.csv", dtype=str, keep_default_na=False); d.loc[0, "topic_id"] = "7"; d.to_csv(b / "documents.csv", index=False)
        side = T.collect(an, ad, allow_partial=True, run_info={})
        assert any("topic ids not in topics.csv" in e for e in side["errors"])
    finally:
        shutil.rmtree(tmp)


def test_sentiment_labels():
    assert [S.label_of(x) for x in (0.05, 0.0501, -0.05, -0.0501, 0.0)] == ["neutral", "positive", "neutral", "negative", "neutral"]


def test_salvage_json():
    broken = ('{"decision": "revise", "categories": [{"id": "cat_1", "name": "World Cup Predictions", "purpose": "Predicting", '
              '"emotion_tone": "enthusiastic", "description": "Posts about predictions.", "inclusion_cues": ["a", "b"]}], '
              '[{"id": "cat_2", "name": "Streaming", "purpose": "Streams", "emotion_tone": "frustrated", "description": "Where to watch.", '
              '"inclusion_cues": ["espn"]]]}]')
    assert G.extract_json(broken)[0] is None
    obj = G.salvage_json(broken)
    assert obj and [c["name"] for c in obj["categories"]] == ["World Cup Predictions", "Streaming"] and obj["categories"][1]["inclusion_cues"] == ["espn"]
    assert G.salvage_json("no categories here") is None
    # S3a: a label answer with the same defect is recovered, validated and agreed by the engine
    lab = '{"decision": "revise", "categories": [{"name": "Refs", "purpose": "p", "emotion_tone": "angry", "description": "d", "inclusion_cues": ["var"]]]}]'
    obj = G.salvage_json(lab); assert obj and G.validate_label(obj, False)[1] is None
    A, B = P.AGENTS
    s = {("t", A, 1, False): lab, ("t", B, 1, False): lab}
    res = G.Debate("label", StubLLM(s), lambda o, it, acc: G.validate_label(o, acc), G.labels_identical).run([{"key": "t", "items_block": "ITEMS"}])["t"]
    assert res["agreement_mode"] == "identical" and res["transcript"][0]["parsed_via"] == "salvage" and res["final"]["categories"][0]["name"] == "Refs"
    assert G._default_salvage(lab, True) is None and G._default_salvage(lab, False) is not None      # never on a truncated answer (N11)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
