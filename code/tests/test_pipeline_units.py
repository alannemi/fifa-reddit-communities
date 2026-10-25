#!/usr/bin/env python
"""
Unit checks for the analytical pipeline — no GPU, no language model (session 05, review R6; extended in sessions 06–07).
Run:  .venv/bin/python code/tests/test_pipeline_units.py        (exit code 1 if any check fails)
Covers: the path root (PIPELINE_ROOT) and the tone vocabulary; period-bin boundaries; the topic-count guard; the
frozen UMAP/cluster components inside BERTopic and the size renumbering; JSON extraction, validation, normalisation
and partition comparison; the deterministic repair; override application (incl. merge chains and the topic-level
category moves, F28); outlier thresholds; the full D2 debate engine with a scripted StubLLM (every agreement_mode,
the repair and no-answer paths, resume); the S3b two-step protocol (definitions, assignment, salvage, checkpoint
resume, batching, cap) and the superseded partition protocol with its chunking fallback; a partial S5 collect;
sentiment labels.
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
    # n_topics = 2000 / m  (too many at the start -> grow -> bisect)
    def stub(X, m):
        k = int(2000 / m); n = len(X)
        return np.array([(i % (k + 1)) - 1 for i in range(n)])
    X = np.zeros((10000, 5))                      # start = max(25, 25) = 25 -> 80 topics: inside the range at once
    r = M.guard_min_cluster_size(X, 10000, True, fit=stub)
    assert r["converged"] and len(r["trials"]) == 1 and r["min_cluster_size"] == 25
    X = np.zeros((4000, 5))                       # start 25 -> 80 ... also fine; force too many with another stub
    def stub2(X, m):
        k = int(6000 / m); n = len(X)             # 25 -> 240 topics (too many) -> 38 -> 157 -> 57 -> 105 -> bisect 47 -> 127 ...
        return np.array([(i % (k + 1)) - 1 for i in range(n)])
    r = M.guard_min_cluster_size(X, 4000, True, fit=stub2)
    assert len(r["trials"]) <= C.GUARD_MAX_TRIALS, r["trials"]
    ks = [t["n_topics"] for t in r["trials"]]
    assert r["n_topics"] == min(ks, key=lambda k: 0 if 50 <= k <= 100 else min(abs(k - 50), abs(k - 100))), (r, ks)
    def stub3(X, m):                              # never enough topics: stays at the floor, flagged
        return np.array([(i % 11) - 1 for i in range(len(X))])
    r = M.guard_min_cluster_size(X, 4000, True, fit=stub3)
    assert not r["converged"] and r["min_cluster_size"] == C.MIN_CLUSTER_FLOOR
    r = M.guard_min_cluster_size(X, 4000, False, fit=stub2)   # disabled: one fit
    assert len(r["trials"]) == 1 and r["enabled"] is False


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
    tax = {"categories": [{"id": "cat_1", "name": "A", "purpose": "p", "emotion_tone": "excited", "description": "d", "inclusion_cues": ["x"], "topics": [0, "1", "topic_2"]},
                          {"id": "cat_2", "name": "B", "purpose": "p", "emotion_tone": "critical", "description": "d", "inclusion_cues": [], "topics": [3]}]}
    p, e = G.validate_taxonomy(tax, [0, 1, 2, 3], False)
    assert e is None and p["categories"][0]["topics"] == [0, 1, 2]
    p, e = G.validate_taxonomy(tax, [0, 1, 2, 3, 4], False)
    assert p is not None and "missing" in e
    assert G.validate_taxonomy({"categories": [{"name": "A", "topics": [1]}]}, [1], False)[0] is None   # missing fields


def test_identity_and_partition():
    a = {"decision": "revise", "categories": [{"name": "Referee Criticism!", "purpose": "To complain", "emotion_tone": "frustrated",
                                               "description": "Posts attack referees.", "inclusion_cues": ["VAR", "penalty"]}]}
    b = json.loads(json.dumps(a)); b["categories"][0].update(name="referee criticism", inclusion_cues=["penalty", "var"])
    assert G.labels_identical(a, b)
    b["categories"][0]["name"] = "Referee praise"
    assert not G.labels_identical(a, b) and not G.labels_identical(a, None)
    ta = {"categories": [{"id": "cat_1", "name": "Match talk", "emotion_tone": "excited", "topics": [0, 1]}, {"id": "cat_2", "name": "Refs", "emotion_tone": "angry", "topics": [2]}]}
    tb = {"categories": [{"id": "cat_9", "name": "Refs", "emotion_tone": "furious and angry", "topics": [2]}, {"id": "cat_3", "name": "match talk", "emotion_tone": "Excited", "topics": [1, 0]}]}
    assert G.taxonomies_identical(ta, tb)
    tb["categories"][0]["emotion_tone"] = "humorous"
    assert not G.taxonomies_identical(ta, tb)


def test_repair_partition():
    emb = {t: np.eye(6)[t] for t in range(6)}
    cats = [{"id": "cat_1", "name": "A", "purpose": "p", "emotion_tone": "t", "description": "d", "inclusion_cues": [], "topics": [0, 1, 1, 99]},
            {"id": "x", "name": "B", "purpose": "p", "emotion_tone": "t", "description": "d", "inclusion_cues": [], "topics": [2, 0]},
            {"id": "cat_3", "name": "C", "purpose": "p", "emotion_tone": "t", "description": "d", "inclusion_cues": [], "topics": [77]}]
    out, assign, rlog = G.repair_partition(cats, [0, 1, 2, 3, 4, 5], emb.get)
    assert [c["id"] for c in out] == ["cat_1", "cat_2"], out           # 'x' invalid -> renumbered; empty C dropped
    assert rlog["unknown"] == [99, 77] and rlog["duplicated"] == [1, 0]
    assert {t for c in out for t in c["topics"]} == {0, 1, 2, 3, 4, 5}
    assert assign[0] == "agent" and assign[3] == "repaired" and assign[5] == "repaired"
    out, assign, _ = G.repair_partition([], [0, 1], emb.get)
    assert out == [] and set(assign.values()) == {"unplaced"}


def test_overrides():
    lab = pd.DataFrame({"topic_id": [1, 2, 3, 4], "name": ["a", "b", "c", "d"], "keywords": ["k"] * 4})
    ov = pd.DataFrame({"topic_id": ["3", "2", "4"], "final_name": ["", "", "D final"], "merge_into": ["2", "1", ""], "notes": ["", "", ""]})
    out, merges = G.apply_topic_overrides(lab, ov)
    assert merges == {3: 1, 2: 1} and out["topic_id"].tolist() == [1, 4]
    assert out.loc[out["topic_id"] == 4, "name"].iloc[0] == "D final" and out.loc[out["topic_id"] == 4, "name_agent"].iloc[0] == "d"
    cats = pd.DataFrame({"category_id": ["cat_1", "cat_2", "cat_3"], "name": ["x", "y", "z"]})
    cov = pd.DataFrame({"category_id": ["cat_3", "cat_1"], "final_name": ["", "X!"], "merge_into": ["cat_2", ""]})
    out, cm = G.apply_category_overrides(cats, cov)
    assert cm == {"cat_3": "cat_2"} and out["category_id"].tolist() == ["cat_1", "cat_2"] and out["name"].tolist() == ["X!", "y"]


# ---------------------------------------------------------------------------
# A scripted stand-in for the language model: answers keyed by (item key, analyst, round, repair flag).
# ---------------------------------------------------------------------------
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


def _taxonomy_answer(user: str, meta: dict, n_groups: int = 2) -> str:
    """Builds a valid partition from the item lines of the prompt (used for chunks and the second pass alike)."""
    ids = [int(m.group(1)) for m in re.finditer(r"^(\d+) \| ", user, flags=re.M)]
    cats = []
    for g in range(n_groups):
        part = ids[g::n_groups]
        if part:
            cats.append({"id": f"cat_{g + 1}", "name": f"Group {g + 1}", "purpose": "p", "emotion_tone": "excited",
                         "description": "d", "inclusion_cues": ["x"], "topics": part})
    return json.dumps({"decision": "revise", "categories": cats})


def _taxonomy_fixture(tmp: Path, n_topics: int):
    pd.DataFrame({"topic_id": list(range(n_topics)), "n_cluster": [10] * n_topics, "n_total": [12] * n_topics,
                  "keywords": [f"kw{t}a, kw{t}b, kw{t}c" for t in range(n_topics)]}).to_csv(tmp / "topics.csv", index=False)
    pd.DataFrame({"topic_id": list(range(n_topics)), "keywords": [f"kw{t}a, kw{t}b" for t in range(n_topics)],
                  "name": [f"Topic {t}" for t in range(n_topics)], "emotion_tone": ["excited"] * n_topics,
                  "description": [f"About {t}. More." for t in range(n_topics)]}).to_csv(tmp / "topics_labelled.csv", index=False)


def test_taxonomy_small_and_repair():
    tmp = Path(tempfile.mkdtemp())
    try:
        _taxonomy_fixture(tmp, 6)
        pd.DataFrame({"topic_id": ["5"], "final_name": [""], "merge_into": ["4"], "notes": [""]}).to_csv(tmp / "topic_overrides.csv", index=False)
        emb = np.eye(6)
        A, B = P.AGENTS
        # both analysts: same partition but one forgets topic 3 and duplicates 0 -> repair turn -> still invalid -> deterministic repair
        bad = json.dumps({"categories": [{"id": "cat_1", "name": "X", "purpose": "p", "emotion_tone": "sad", "description": "d", "inclusion_cues": [], "topics": [0, 1, 0]},
                                         {"id": "cat_2", "name": "Y", "purpose": "p", "emotion_tone": "sad", "description": "d", "inclusion_cues": [], "topics": [2, 4]}]})
        s = {("bin", A, 1, False): bad, ("bin", A, 1, True): bad, ("bin", B, 1, False): bad, ("bin", B, 1, True): bad}
        cats = G.run_taxonomy(tmp, 1, lambda t: emb[int(t)], StubLLM(s), "2010-01-01", "2010-09-30", protocol="partition")
        assert len(cats) == 2 and cats["agreement_mode"].iloc[0] == "identical"
        tc = pd.read_csv(tmp / "topic_categories.csv")
        assert sorted(tc["topic_id"]) == [0, 1, 2, 3, 4] and tc.loc[tc["topic_id"] == 3, "category_assignment"].iloc[0] == "repaired"
        assert cats["n_posts"].sum() == 12 * 6 and (tmp / "category_overrides.csv").exists() and (tmp / "taxonomy.json").exists()
        tj = json.loads((tmp / "taxonomy.json").read_text())
        assert tj["topic_overrides_merged"] == {"5": 4} or tj["topic_overrides_merged"] == {5: 4}
    finally:
        shutil.rmtree(tmp)


def test_taxonomy_chunking():
    tmp = Path(tempfile.mkdtemp())
    try:
        n = 130
        _taxonomy_fixture(tmp, n)
        rng = np.random.RandomState(1)
        emb = rng.normal(size=(n, 8))
        cats = G.run_taxonomy(tmp, 4, lambda t: emb[int(t)], StubLLM({}, taxonomy_fn=_taxonomy_answer), "2014-10-01", "2018-01-31", protocol="partition")
        tc = pd.read_csv(tmp / "topic_categories.csv")
        assert sorted(tc["topic_id"]) == list(range(n)) and set(tc["category_assignment"]) == {"agent"}
        tj = json.loads((tmp / "taxonomy.json").read_text())
        assert tj["chunking"]["n_chunks"] == 2 and len(cats) == 2 and set(tj["debates"]) == {"chunk_1", "chunk_2", "bin"}
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
                      "prompt_version": ["1.1"] * 2, "decoding": ["greedy"] * 2}).to_csv(b / "topics_labelled.csv", index=False)
        pd.DataFrame({"topic_id": [1], "name": ["One"], "needs_review": [True], "final_name": ["One (final)"], "merge_into": [""], "notes": [""]}).to_csv(b / "topic_overrides.csv", index=False)
        pd.DataFrame({"topic_id": [0, 1], "category_id": ["cat_1", "cat_2"], "category_assignment": ["agent", "agent"]}).to_csv(b / "topic_categories.csv", index=False)
        pd.DataFrame({"period_bin": [1, 1], "category_id": ["cat_1", "cat_2"], "name": ["C1", "C2"], "purpose": ["p", "p"], "emotion_tone": ["t", "t"], "tone_category": ["other", "other"],
                      "description": ["d", "d"], "inclusion_cues": ["", ""], "topic_ids": ["0", "1"], "n_topics": [1, 1], "n_posts": [2, 2], "share": [0.5, 0.5], "round_iterations": [1, 1],
                      "agreed": [True, True], "agreement_mode": ["identical", "identical"], "needs_review": [False, False], "n_repaired": [0, 0]}).to_csv(b / "categories.csv", index=False)
        pd.DataFrame({"category_id": ["cat_2"], "name": ["C2"], "n_topics": [1], "needs_review": [False], "final_name": [""], "merge_into": ["cat_1"], "notes": [""]}).to_csv(b / "category_overrides.csv", index=False)
        side = T.collect(an, ad, allow_partial=True, run_info={})
        pv = pd.read_csv(ad / "post_variables.csv", dtype=str, keep_default_na=False)
        assert list(pv.columns) == T.POST_COLUMNS + T.AUDIT_COLUMNS and len(pv) == 10
        assert pv["topic_id"].tolist()[:6] == ["0", "0", "1", "1", "-1", ""] and pv["topic_id"].tolist()[6:] == [""] * 4
        assert pv["category_id"].tolist()[:6] == ["cat_1", "cat_1", "cat_1", "cat_1", "", ""]       # cat_2 merged into cat_1
        assert pv["category_id_agent"].tolist()[2] == "cat_2" and pv["post_sentiment"].tolist()[0] == "0.5000"
        assert side["bins_complete"] == [1] and side["complete"] is False and side["errors"] == []
        lab = pd.read_csv(an / "topic_labels_all_bins.csv", keep_default_na=False)
        assert list(lab.columns) == T.LABEL_COLUMNS and lab.loc[lab["topic_id"] == 1, "name"].iloc[0] == "One (final)" and lab["n_total"].tolist() == [2, 2]
        cat = pd.read_csv(an / "categories_all_bins.csv", keep_default_na=False)
        assert cat["category_id"].tolist() == ["cat_1"] and cat["n_topics"].iloc[0] == 2 and (an / "category_crosswalk.csv").exists()
        assert all(c in lab.columns for c in G.ASSIGN_COLS) and (lab[G.ASSIGN_COLS] == "").all().all()   # v1.1-style outputs: empty
        # --- F28: topic-level move, no category merge: topic 0 -> cat_2; cat_1 stays with n_topics 0 ----------------
        pd.DataFrame({"category_id": ["cat_2"], "name": ["C2"], "n_topics": [1], "needs_review": [False], "final_name": [""], "merge_into": [""], "notes": [""]}).to_csv(b / "category_overrides.csv", index=False)
        pd.DataFrame({"topic_id": [0, 1, 7], "name": ["Zero", "One", "?"], "category_id_agent": ["cat_1", "cat_2", ""], "category_name": ["C1", "C2", ""],
                      "category_assignment": ["agent", "agent", ""], "final_category_id": ["cat_2", "", "cat_1"], "notes": ["", "", "unknown topic"]}).to_csv(b / "topic_category_overrides.csv", index=False)
        side = T.collect(an, ad, allow_partial=True, run_info={})
        pv = pd.read_csv(ad / "post_variables.csv", dtype=str, keep_default_na=False)
        assert pv["category_id"].tolist()[:4] == ["cat_2"] * 4 and pv["category_id_agent"].tolist()[:4] == ["cat_1", "cat_1", "cat_2", "cat_2"] and side["errors"] == []
        cat = pd.read_csv(an / "categories_all_bins.csv", keep_default_na=False)
        assert cat.set_index("category_id")["n_topics"].to_dict() == {"cat_1": 0, "cat_2": 2}
        lab = pd.read_csv(an / "topic_labels_all_bins.csv", keep_default_na=False)
        assert lab["category_id"].tolist() == ["cat_2", "cat_2"]
        # --- F28 + topic merge: topic 1 merged into 0; a move on the merged-away topic is ignored, one on the target moves all
        pd.DataFrame({"topic_id": [1], "name": ["One"], "needs_review": [True], "final_name": [""], "merge_into": ["0"], "notes": [""]}).to_csv(b / "topic_overrides.csv", index=False)
        pd.DataFrame({"topic_id": [0, 1], "final_category_id": ["cat_2", "cat_1"]}).to_csv(b / "topic_category_overrides.csv", index=False)
        side = T.collect(an, ad, allow_partial=True, run_info={})
        pv = pd.read_csv(ad / "post_variables.csv", dtype=str, keep_default_na=False)
        assert pv["topic_id"].tolist()[:4] == ["0"] * 4 and pv["category_id"].tolist()[:4] == ["cat_2"] * 4 and side["errors"] == []
        # --- F28 to an unknown category -> S5 check fails (tolerated only with --allow-partial) -------------------
        pd.DataFrame({"topic_id": [0], "final_category_id": ["cat_9"]}).to_csv(b / "topic_category_overrides.csv", index=False)
        side = T.collect(an, ad, allow_partial=True, run_info={})
        assert any("category ids not in categories.csv" in e for e in side["errors"])
    finally:
        shutil.rmtree(tmp)


def test_sentiment_labels():
    assert [S.label_of(x) for x in (0.05, 0.0501, -0.05, -0.0501, 0.0)] == ["neutral", "positive", "neutral", "negative", "neutral"]


def test_salvage_json():
    broken = ('{"decision": "revise", "categories": [{"id": "cat_1", "name": "World Cup Predictions", "purpose": "Predicting", '
              '"emotion_tone": "enthusiastic", "description": "Posts about predictions.", "inclusion_cues": ["a", "b"], "topics": [1, 3, 9]}], '
              '[{"id": "cat_2", "name": "Streaming", "purpose": "Streams", "emotion_tone": "frustrated", "description": "Where to watch.", '
              '"inclusion_cues": ["espn"], "topics": [4, 5]}], [{"id": "cat_3", "name": "Vuvuzela Complaints", "purpose": "Complaints", '
              '"emotion_tone": "negative", "description": "Noise.", "inclusion_cues": ["horns"], "topics": [6]]], [{"id": "cat_4", "name": "Rest", '
              '"purpose": "p", "emotion_tone": "t", "description": "d", "inclusion_cues": ["x"], "topics": [0, 2, 7, 8, 10, 11]]]}]')
    assert G.extract_json(broken)[0] is None
    obj = G.salvage_json(broken)
    assert obj and [c["id"] for c in obj["categories"]] == ["cat_1", "cat_2", "cat_3", "cat_4"] and obj["categories"][2]["topics"] == [6]
    parsed, err = G.validate_taxonomy(obj, list(range(12)), False)
    assert err is None and len(parsed["categories"]) == 4
    assert G.salvage_json("no categories here") is None
    # the engine uses it: a broken round-1 answer from both analysts is recovered, validated and agreed
    A, B = P.AGENTS
    s = {("bin", A, 1, False): broken, ("bin", B, 1, False): broken}
    res = G.Debate("taxonomy", StubLLM(s), lambda o, it, acc: G.validate_taxonomy(o, it["topic_ids"], acc), G.taxonomies_identical).run(
        [{"key": "bin", "topic_ids": list(range(12)), "items_block": "ITEMS"}])["bin"]
    assert res["agreement_mode"] == "identical" and res["transcript"][0]["parsed_via"] == "salvage" and len(res["final"]["categories"]) == 4
    # S3a: a label answer with the same defect
    lab = '{"decision": "revise", "categories": [{"name": "Refs", "purpose": "p", "emotion_tone": "angry", "description": "d", "inclusion_cues": ["var"]]]}]'
    obj = G.salvage_json(lab); assert obj and G.validate_label(obj, False)[1] is None


def test_taxonomy_empty_is_not_fatal():
    tmp = Path(tempfile.mkdtemp())
    try:
        _taxonomy_fixture(tmp, 4)
        A, B = P.AGENTS
        for protocol, key in (("partition", "bin"), ("two_step", "definitions")):
            s = {(key, a, 1, r): "garbage" for a in (A, B) for r in (False, True)}
            cats = G.run_taxonomy(tmp, 1, lambda t: np.eye(4)[int(t)], StubLLM(s), "2010-01-01", "2010-09-30", protocol=protocol)
            assert len(cats) == 0 and (tmp / "categories.csv").exists(), protocol
            tc = pd.read_csv(tmp / "topic_categories.csv", keep_default_na=False)
            assert sorted(tc["topic_id"]) == [0, 1, 2, 3] and set(tc["category_assignment"]) == {"unplaced"}, protocol
            assert T.bin_state(tmp) == "labelled"
            tj = json.loads((tmp / "taxonomy.json").read_text())
            assert tj["protocol"] == protocol and tj["agreement_mode"] == "no_answer"
            if protocol == "two_step":
                assert tj["assignment_items"] == [] and (tmp / "taxonomy_definitions.json").exists()
            (tmp / "categories.csv").unlink()
    finally:
        shutil.rmtree(tmp)


# ---------------------------------------------------------------------------
# S3b two-step protocol (decision D5, 2026-10-24)
# ---------------------------------------------------------------------------
def test_two_step_validators():
    # definitions: canonical ids, soft problems kept with the answer, hard failure without names
    obj = {"categories": [{"id": "1", "name": "A", "purpose": "p", "emotion_tone": "sad", "description": "d", "inclusion_cues": ["x"]},
                          {"id": "C2", "name": "a", "purpose": "", "emotion_tone": "sad", "description": "d", "inclusion_cues": [f"c{i}" for i in range(12)], "topics": [1]},
                          {"id": "cat_3", "name": "B", "purpose": "p", "emotion_tone": "angry", "description": "d", "inclusion_cues": "x; y"}]}
    d, err = G.validate_definitions(obj, False)
    assert d is not None and [c["id"] for c in d["categories"]] == ["cat_1", "cat_2", "cat_3"] and d["categories"][0]["id_given"] == "1"
    assert "duplicated category name" in err and "field 'purpose' is empty" in err
    assert d["categories"][1]["n_cues_given"] == 12 and len(d["categories"][1]["inclusion_cues"]) == 10 and d["categories"][1]["topics_ignored"]
    assert d["categories"][2]["inclusion_cues"] == ["x", "y"] and "id_given" not in d["categories"][2]
    clean, log_ = G.cleanup_definitions(d["categories"])
    assert [c["id"] for c in clean] == ["cat_1", "cat_2"] and clean[0]["name"] == "A" and clean[1]["name"] == "B"
    assert log_["merged_same_name"] == {"cat_2": "cat_1"} and log_["renumbered"] == {"cat_3": "cat_2"} and "x" in clean[0]["inclusion_cues"]
    assert G.validate_definitions({"categories": [{"name": "", "purpose": "p"}]}, False)[0] is None
    assert G.validate_definitions({"categories": []}, False)[0] is None
    assert G.validate_definitions({"decision": "accept"}, False)[0] is None and G.validate_definitions({"decision": "accept"}, True)[0] == {"decision": "accept"}
    good, err = G.validate_definitions({"categories": [{"id": "cat_1", "name": "A", "purpose": "p", "emotion_tone": "t", "description": "d", "inclusion_cues": ["x"]}]}, False)
    assert err is None and good["categories"][0]["id"] == "cat_1" and "id_given" not in good["categories"][0]
    # assignments: shapes, lenient references, coverage problems kept with the cleaned answer
    idx = G.build_category_index([{"id": "cat_1", "name": "World Cup Streaming & Access"}, {"id": "cat_2", "name": "Referee Criticism"}])
    assert G.resolve_category_ref("cat_2", idx) == ("cat_2", "id") and G.resolve_category_ref("2", idx) == ("cat_2", "number")
    assert G.resolve_category_ref("Category 1", idx) == ("cat_1", "number") and G.resolve_category_ref("referee criticism", idx) == ("cat_2", "name")
    assert G.resolve_category_ref(None, idx) == (None, "missing") and G.resolve_category_ref("none", idx) == (None, "missing") and G.resolve_category_ref("cat_9", idx) == (None, "unknown")
    p, e = G.validate_assignments({"assignments": [{"topic_id": 5, "category_id": "cat_2"}, {"topic_id": "7", "category_id": "Referee Criticism"}]}, [5, 7], idx, False)
    assert e is None and G._mapping(p) == {5: "cat_2", 7: "cat_2"} and p["resolved_from"] == {"7": "name"}
    p, e = G.validate_assignments({"assignments": {"5": "cat_1", "7": "cat_2"}}, [5, 7], idx, False)                 # mapping form
    assert e is None and G._mapping(p) == {5: "cat_1", 7: "cat_2"}
    p, e = G.validate_assignments({"categories": [{"id": "cat_1", "topics": [5, 7]}]}, [5, 7], idx, False)           # v1.1 habit
    assert e is None and G._mapping(p) == {5: "cat_1", 7: "cat_1"}
    p, e = G.validate_assignments({"assignments": [{"topic_id": 5, "category_id": None}, {"topic_id": 5, "category_id": "cat_1"}, {"topic_id": 99, "category_id": "cat_1"}]}, [5, 7], idx, False)
    assert p is not None and G._mapping(p) == {5: "cat_1"} and "missing topic_id(s): [7]" in e and "not among" not in e
    assert "not in this list" not in e and p["cleanup"]["foreign"] == [99]                                            # N13: extras recorded, no repair
    p, e = G.validate_assignments({"assignments": [{"topic_id": 5, "category_id": "cat_2"}, {"topic_id": 5, "category_id": "cat_1"}, {"topic_id": 7, "category_id": "none"}]}, [5, 7], idx, False)
    assert G._mapping(p) == {5: "cat_2"} and p["cleanup"]["duplicated"] == [5] and "more than once" not in e
    assert "topic_id(s) 7 ('none')" in e and "missing topic_id(s): [7]" in e
    p, e = G.validate_assignments({"assignments": [{"topic_id": 5, "category_id": "cat_2"}, {"topic_id": 5, "category_id": "cat_1"}, {"topic_id": 99, "category_id": "cat_1"}]}, [5], idx, False)
    assert e is None and G._mapping(p) == {5: "cat_2"} and p["cleanup"] == {"foreign": [99], "duplicated": [5]}     # complete -> no repair turn
    # D6: category_id + category_name; a conflict is resolved in favour of the name and reported (one repair turn)
    p, e = G.validate_assignments({"assignments": [{"topic_id": 5, "category_id": "cat_2", "category_name": "Referee Criticism"}]}, [5], idx, False)
    assert e is None and G._mapping(p) == {5: "cat_2"} and p["resolved_from"] == {}
    p, e = G.validate_assignments({"assignments": [{"topic_id": 5, "category_id": "cat_1", "category_name": "Referee Criticism"}]}, [5], idx, False)
    assert e is None and G._mapping(p) == {5: "cat_2"} and p["resolved_from"] == {"5": "name_over_id"} and p["cleanup"]["conflicts"] == [[5, "cat_1", "Referee Criticism"]]
    p, e = G.validate_assignments({"assignments": [{"topic_id": 5, "category_name": "Referee Criticism"}]}, [5], idx, False)
    assert e is None and G._mapping(p) == {5: "cat_2"} and p["resolved_from"] == {"5": "name"}
    p, e = G.validate_assignments({"assignments": [{"topic_id": 5, "category_id": "cat_1", "category_name": "No Such Category"}]}, [5], idx, False)
    assert e is None and G._mapping(p) == {5: "cat_1"}
    assert G.validate_assignments({"foo": 1}, [5], idx, False)[0] is None
    assert G.validate_assignments({"decision": "accept"}, [5], idx, True)[0] == {"decision": "accept"}
    assert G.assignments_identical({"assignments": [{"topic_id": 5, "category_id": "cat_1"}]}, {"assignments": [{"topic_id": 5, "category_id": "cat_1"}]})
    assert not G.assignments_identical({"assignments": []}, {"assignments": []}) and not G.assignments_identical(None, {"assignments": []})
    da = {"categories": [{"name": "Match Talk!", "emotion_tone": "excited"}, {"name": "Refs", "emotion_tone": "angry"}]}
    db = {"categories": [{"name": "refs", "emotion_tone": "furious and angry"}, {"name": "match talk", "emotion_tone": "Excited"}]}
    assert G.definitions_identical(da, db)
    db["categories"].append({"name": "Other", "emotion_tone": "neutral"})
    assert not G.definitions_identical(da, db) and not G.definitions_identical(da, None)


def test_salvage_assignments_and_truncated_definitions():
    broken = '{"decision":"revise","assignments":[{"topic_id": 5, "category_id": "cat_2"}], [{"topic_id":7,"category_id":"cat_1"}]]]'
    assert G.extract_json(broken)[0] is None
    obj = G.salvage_assignments(broken, False)
    assert obj and G._mapping(obj) == {5: "cat_2", 7: "cat_1"} and obj["decision"] == "revise"
    assert G._mapping(G.salvage_assignments('[{"category_id": "cat_3", "topic_id": "4"}', False)) == {4: "cat_3"}
    assert G._mapping(G.salvage_assignments('{"assignments": {"12": "cat_3", "13": "Refs"', False)) == {12: "cat_3", 13: "Refs"}
    assert G.salvage_assignments(broken, True) is None and G.salvage_assignments("nothing here", False) is None
    # definitions: a truncated answer keeps the complete category objects and drops the last (cut-off) one
    cut = ('{"decision": "revise", "categories": [{"id": "cat_1", "name": "A", "purpose": "p", "emotion_tone": "t", "description": "d", "inclusion_cues": ["x"]}, '
           '{"id": "cat_2", "name": "B", "purpose": "p", "emotion_tone": "t", "description": "d", "inclusion_cues": ["y"]}, {"id": "cat_3", "name": "C", "purpose": "p')
    obj = G.salvage_definitions(cut, True)
    assert obj and [c["name"] for c in obj["categories"]] == ["A", "B"]
    assert [c["name"] for c in G.salvage_definitions(cut, False)["categories"]] == ["A", "B", "C"]
    assert G._default_salvage(cut, True) is None and G._default_salvage(cut, False) is not None
    # the engine uses the stage's salvage: a truncated definitions answer is recovered
    A, B = P.AGENTS
    s = {("definitions", A, 1, False): "TRUNC:" + cut, ("definitions", B, 1, False): cut.replace('{"id": "cat_3", "name": "C", "purpose": "p', '') + "]}"}
    llm = StubLLM(s)
    res = G.Debate("define", llm, lambda o, it, acc: G.validate_definitions(o, acc), G.definitions_identical,
                   max_new_tokens=50, salvage=G.salvage_definitions).run([{"key": "definitions", "items_block": "ITEMS"}])["definitions"]
    assert res["transcript"][0]["truncated"] and res["transcript"][0]["parsed_via"] == "salvage" and res["agreement_mode"] == "identical"


def _two_step_answer(user: str, meta: dict) -> str:
    """Scripted stand-in for both steps: definitions X/Y/Z/W with non-canonical ids; assignment by topic_id modulo."""
    if "Topics to assign" in user:
        tids = [int(m.group(1)) for m in re.finditer(r"^(\d+) \| ", user.split("Topics to assign", 1)[1], flags=re.M)]
        cids = [m.group(1) for m in re.finditer(r"^(cat_\d+) \| ", user, flags=re.M)]
        return json.dumps({"decision": "revise", "assignments": [{"topic_id": t, "category_id": cids[t % min(3, len(cids))]} for t in tids]})
    cats = [{"id": str(i), "name": n, "purpose": "p", "emotion_tone": "excited", "description": f"About {n}. More.", "inclusion_cues": ["x"]}
            for i, n in enumerate(["X", "Y", "Z", "W"], 1)]
    return json.dumps({"decision": "revise", "categories": cats})


def _pairs(*pairs):
    return json.dumps({"decision": "revise", "assignments": [{"topic_id": t, "category_id": c} for t, c in pairs]})


def test_taxonomy_two_step():
    tmp = Path(tempfile.mkdtemp())
    try:
        n = 45
        _taxonomy_fixture(tmp, n)
        emb = np.random.RandomState(0).normal(size=(n, 8))
        A, B = P.AGENTS
        dup = json.dumps({"categories": [{"id": "1", "name": "X", "purpose": "p", "emotion_tone": "excited", "description": "d", "inclusion_cues": ["x"]},
                                         {"id": "1", "name": "x", "purpose": "p", "emotion_tone": "excited", "description": "d", "inclusion_cues": ["x"]},
                                         {"id": "2", "name": "Y", "purpose": "p", "emotion_tone": "excited", "description": "d", "inclusion_cues": ["x"]},
                                         {"id": "3", "name": "Z", "purpose": "p", "emotion_tone": "excited", "description": "d", "inclusion_cues": ["x"]}]})
        s = {("definitions", A, 1, False): dup, ("definitions", A, 1, True): dup,          # soft error kept; B's default X/Y/Z/W
             ("definitions", A, 2, False): '{"decision": "accept"}',                           # -> accepted_by_A, round 2
             ("assign_002", A, 1, False): _pairs((1, "cat_1")), ("assign_002", B, 1, False): _pairs((1, "Y")),
             ("assign_002", A, 2, False): '{"decision": "accept"}',                            # -> cat_2, contested, agreed
             ("assign_003", A, 1, False): '{"decision": "revise", "assignments": {"2": "cat_3"}}',
             ("assign_003", B, 1, False): _pairs((2, "Z")),                                   # mapping form + name -> identical
             ("assign_004", A, 1, False): "garbage", ("assign_004", A, 1, True): "garbage",
             ("assign_004", B, 1, False): _pairs((3, None)), ("assign_004", B, 1, True): _pairs((3, "none")),   # no_answer; topic 3 repaired
             ("assign_005", A, 1, False): _pairs((99, "cat_1"), (4, "cat_1")), ("assign_005", A, 1, True): _pairs((4, "cat_1")),
             ("assign_005", B, 1, False): _pairs((4, "cat_1")),                               # foreign id repaired away -> identical
             ("assign_006", A, 1, False): _pairs((5, "cat_2"), (5, "cat_3")), ("assign_006", A, 1, True): _pairs((5, "cat_2"), (5, "cat_3")),
             ("assign_006", B, 1, False): _pairs((5, "cat_2")),                               # duplicate kept-with-error, first wins -> identical
             ("assign_007", A, 1, False): '{"decision": "revise", "assignments": [{"topic_id": 6, "category_id": "cat_1"}], ["x"]}',
             ("assign_007", B, 1, False): _pairs((6, "cat_1")),                               # salvage -> identical
             }
        for r in (False, True):
            s[("assign_008", A, 1, r)] = "garbage"; s[("assign_008", B, 1, r)] = "garbage"      # no answer at all -> repaired
        llm = StubLLM(s, taxonomy_fn=_two_step_answer)
        cats = G.run_taxonomy(tmp, 1, lambda t: emb[int(t)], llm, "2010-01-01", "2010-09-30", protocol="two_step", assign_batch=1)
        assert cats["category_id"].tolist() == ["cat_1", "cat_2", "cat_3"] and cats["name"].tolist() == ["X", "Y", "Z"]
        assert set(cats["agreement_mode"]) == {"accepted_by_A"} and set(cats["round_iterations"]) == {2} and not cats["needs_review"].any()
        tc = pd.read_csv(tmp / "topic_categories.csv", keep_default_na=False).set_index("topic_id")
        assert sorted(tc.index) == list(range(n)) and list(tc.columns) == ["category_id", "category_assignment"] + G.ASSIGN_COLS
        assert tc.loc[1, "category_id"] == "cat_2" and tc.loc[1, "assignment_agreement_mode"] == "accepted_by_A" and tc.loc[1, "assignment_round_iterations"] == 2
        assert bool(tc.loc[1, "assignment_contested"]) is True and bool(tc.loc[1, "assignment_needs_review"]) is False
        assert tc.loc[0, "assignment_agreement_mode"] == "identical" and bool(tc.loc[0, "assignment_contested"]) is False
        assert tc.loc[2, "category_id"] == "cat_3" and tc.loc[2, "assignment_agreement_mode"] == "identical"
        assert tc.loc[3, "category_assignment"] == "repaired" and tc.loc[3, "assignment_agreement_mode"] == "no_answer" and bool(tc.loc[3, "assignment_needs_review"]) is True
        assert tc.loc[4, "category_id"] == "cat_1" and tc.loc[4, "assignment_agreement_mode"] == "identical"
        assert tc.loc[5, "category_id"] == "cat_2" and tc.loc[5, "category_assignment"] == "agent"
        assert tc.loc[6, "category_id"] == "cat_1" and tc.loc[7, "category_assignment"] == "repaired" and bool(tc.loc[7, "assignment_needs_review"]) is True
        assert (tc["category_assignment"] == "repaired").sum() == 2 and (tc["assignment_item"] == "assign_001").sum() == 1
        tj = json.loads((tmp / "taxonomy.json").read_text())
        assert tj["protocol"] == "two_step" and tj["assign_batch"] == 1 and tj["categories_dropped_empty"] == ["cat_4"]
        assert [c["id"] for c in tj["categories_defined"]] == ["cat_1", "cat_2", "cat_3", "cat_4"] and tj["categories_defined"][0]["id_given"] == "1"
        assert len(tj["assignment_items"]) == n and set(tj["debates"]) == {"definitions"} | {f"assign_{i:03d}" for i in range(1, n + 1)}
        assert tj["n_answers_salvaged"] >= 1 and tj["debates"]["assign_007"]["transcript"][0]["parsed_via"] == "salvage"
        assert tj["debates"]["assign_003"]["transcript"][1]["parsed"]["resolved_from"] == {"2": "name"}
        t5 = tj["debates"]["assign_005"]["transcript"]
        assert t5[0]["error"] is None and t5[0]["parsed"]["cleanup"]["foreign"] == [99] and len(t5) == 2 and tj["n_answers_cleaned"] >= 2   # N13
        assert (tmp / "taxonomy_definitions.json").exists() and (tmp / "topic_category_overrides.csv").exists() and (tmp / "category_overrides.csv").exists()
        tov = pd.read_csv(tmp / "topic_category_overrides.csv", keep_default_na=False)
        assert list(tov.columns) == ["topic_id", "name", "category_id_agent", "category_name", "category_assignment", "assignment_needs_review",
                                     "similarity_rank", "final_category_id", "notes"] and len(tov) == n
        assert tj["shortlist"]["text_embeddings"] is False and tj["shortlist"]["k"] == 0 and (tc["assignment_similarity_rank"] == "").all()
        # revision turns of the assign stage name "assignments", never "categories"; bookkeeping keys are not shown
        rev = [u for m, u in llm.users if m["key"] == "assign_002" and m["round"] == 2]
        assert rev and '"assignments": [...]' in rev[0] and '"categories": [...]' not in rev[0] and "resolved_from" not in rev[0]
        rev_def = [u for m, u in llm.users if m["key"] == "definitions" and m["round"] == 2]
        assert rev_def and '"categories": [...]' in rev_def[0] and "id_given" not in rev_def[0]
        # resume: categories.csv removed, the step-1 checkpoint is reused even if the model could no longer define anything
        (tmp / "categories.csv").unlink()
        s2 = {("definitions", a, 1, r): "garbage" for a in (A, B) for r in (False, True)}
        cats2 = G.run_taxonomy(tmp, 1, lambda t: emb[int(t)], StubLLM(s2, taxonomy_fn=_two_step_answer), "2010-01-01", "2010-09-30", protocol="two_step")
        assert cats2["name"].tolist() == ["X", "Y", "Z"] and cats2["agreement_mode"].iloc[0] == "accepted_by_A"
        # batching: 3 topics per item -> 15 items, item recorded per topic
        (tmp / "categories.csv").unlink(); (tmp / "taxonomy_definitions.json").unlink()
        cats3 = G.run_taxonomy(tmp, 1, lambda t: emb[int(t)], StubLLM({}, taxonomy_fn=_two_step_answer), "2010-01-01", "2010-09-30", protocol="two_step", assign_batch=3)
        tj3 = json.loads((tmp / "taxonomy.json").read_text()); tc3 = pd.read_csv(tmp / "topic_categories.csv", keep_default_na=False)
        assert len(tj3["assignment_items"]) == 15 and tc3["assignment_item"].nunique() == 15 and set(tc3["assignment_agreement_mode"]) == {"identical"}
        assert len(cats3) == 3 and tj3["agreement_mode"] == "identical" and tj3["round_iterations"] == 1
    finally:
        shutil.rmtree(tmp)


def test_taxonomy_shortlist():
    """D6: step 2 lists each topic's k most similar categories (text embeddings), nearest first; the answer carries
    category_id + category_name (the name wins a conflict); a topic whose agreed category ranks worse than
    TAXONOMY_REVIEW_RANK by similarity is flagged although the analysts agreed."""
    tmp = Path(tempfile.mkdtemp()); saved = (C.TAXONOMY_SHORTLIST, C.TAXONOMY_REVIEW_RANK)
    try:
        n = 12
        _taxonomy_fixture(tmp, n)
        C.TAXONOMY_SHORTLIST, C.TAXONOMY_REVIEW_RANK = 4, 3
        groups = [{"id": f"cat_{i + 1}", "name": f"Group {i}", "purpose": "p", "emotion_tone": "excited",
                   "description": f"Posts about {i}. More.", "inclusion_cues": ["x"]} for i in range(n)]
        defs = json.dumps({"decision": "revise", "categories": groups})

        def embed_texts(texts):          # positions on a line: topic t at t, category "Group c" at c + 0.3 -> nearest = same number
            out = []
            for s_ in texts:
                m = re.search(r"(Topic|Group) (\d+)", s_)
                out.append([1.0, (int(m.group(2)) + (0.3 if m.group(1) == "Group" else 0.0)) / 100.0])
            return np.asarray(out)

        def answer(user, meta):
            if "Topics to assign" not in user:
                return defs
            tid = int(re.search(r"^(\d+) \| Topic", user.split("Topics to assign", 1)[1], flags=re.M).group(1))
            cands = re.findall(r"^(cat_\d+) \| (Group \d+) \|", user, flags=re.M)
            assert len(cands) == 4 and "4 of the 12 categories" in user, user[:300]
            if tid == 5:
                cid, name = cands[3]                       # both analysts take the 4th candidate -> agreed, rank 4 -> flagged
            elif tid == 6:
                cid, name = cands[0][0], cands[1][1]       # id of the 1st candidate, name of the 2nd -> the name wins
            else:
                cid, name = cands[0]
            return json.dumps({"decision": "revise", "assignments": [{"topic_id": tid, "category_id": cid, "category_name": name}]})

        emb = np.eye(n)
        cats = G.run_taxonomy(tmp, 1, lambda t: emb[int(t)], StubLLM({}, taxonomy_fn=answer), "2010-01-01", "2010-09-30",
                              protocol="two_step", assign_batch=1, embed_texts=embed_texts)
        tc = pd.read_csv(tmp / "topic_categories.csv", keep_default_na=False).set_index("topic_id")
        tj = json.loads((tmp / "taxonomy.json").read_text())
        assert tc.loc[0, "category_id"] == "cat_1" and tc.loc[11, "category_id"] == "cat_12"
        assert tc.loc[3, "assignment_candidates"] == "cat_4|cat_3|cat_5|cat_2" and int(tc.loc[3, "assignment_similarity_rank"]) == 1
        assert tc.loc[5, "category_id"] == "cat_4" and int(tc.loc[5, "assignment_similarity_rank"]) == 4
        assert bool(tc.loc[5, "assignment_needs_review"]) is True and bool(tc.loc[5, "assignment_contested"]) is False
        assert tc.loc[5, "assignment_agreement_mode"] == "identical"
        assert tc.loc[6, "category_id"] == "cat_6" and int(tc.loc[6, "assignment_similarity_rank"]) == 2 and bool(tc.loc[6, "assignment_needs_review"]) is False
        tr = tj["debates"]["assign_007"]["transcript"]
        assert tr[0]["parsed"]["resolved_from"] == {"6": "name_over_id"} and tr[0]["error"] is None and len(tr) == 2        # N13: no repair turn
        assert tr[0]["parsed"]["cleanup"]["conflicts"] == [[6, "cat_7", "Group 5"]] and tj["n_answers_cleaned"] == 2
        assert tj["shortlist"] == {"k": 4, "review_rank": 3, "text_embeddings": True, "rank_distribution": {"1": 10, "2": 1, "4": 1}, "n_flagged_by_rank": 1}
        assert tj["categories_dropped_empty"] == ["cat_7"] and len(cats) == 11 and tj["assignment_items"][3]["candidates"] == ["cat_4", "cat_3", "cat_5", "cat_2"]
        tov = pd.read_csv(tmp / "topic_category_overrides.csv", keep_default_na=False).set_index("topic_id")
        assert bool(tov.loc[5, "assignment_needs_review"]) is True and int(tov.loc[5, "similarity_rank"]) == 4
    finally:
        C.TAXONOMY_SHORTLIST, C.TAXONOMY_REVIEW_RANK = saved
        shutil.rmtree(tmp)


def test_taxonomy_two_step_cap():
    tmp = Path(tempfile.mkdtemp())
    try:
        _taxonomy_fixture(tmp, 4)
        A, B = P.AGENTS
        def defs(name):
            return json.dumps({"categories": [{"id": "cat_1", "name": name, "purpose": "p", "emotion_tone": "t", "description": "d", "inclusion_cues": ["x"]}]})
        s = {}
        for r in range(1, 6):
            s[("definitions", A, r, False)] = defs(f"A{r}"); s[("definitions", B, r, False)] = defs(f"B{r}")
        cats = G.run_taxonomy(tmp, 1, lambda t: np.eye(4)[int(t)], StubLLM(s, taxonomy_fn=_two_step_answer), "2010-01-01", "2010-09-30", protocol="two_step")
        assert len(cats) == 1 and cats["name"].iloc[0] == "A5" and cats["agreement_mode"].iloc[0] == "cap" and cats["round_iterations"].iloc[0] == 5
        assert cats["needs_review"].all() and not cats["agreed"].any()
        tc = pd.read_csv(tmp / "topic_categories.csv", keep_default_na=False)
        assert set(tc["category_id"]) == {"cat_1"} and set(tc["category_assignment"]) == {"agent"}
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
