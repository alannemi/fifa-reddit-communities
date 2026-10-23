#!/usr/bin/env python
"""
Unit checks for the analytical pipeline (session 05, plan §5.1 / review R6) — no GPU, no language model.
Run:  .venv/bin/python code/tests/test_pipeline_units.py        (exit code 1 if any check fails)
Covers: period-bin boundaries; the topic-count guard; the frozen UMAP/cluster components inside BERTopic and
the size renumbering; JSON extraction, validation, normalisation and partition comparison; the deterministic
repair; override application (incl. merge chains); outlier thresholds; the full D2 debate engine with a
scripted StubLLM (every agreement_mode, the repair and no-answer paths, resume); the S3b chunking fallback;
a partial S5 collect; sentiment labels.
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
    assert not r["converged"] and r["min_cluster_size"] == C.BIN_MIN_CLUSTER_FLOOR
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
    topics, mapping = M.fit_bin_model(bt, docs, emb)
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
    tt = M.bin_topic_table(bt, docs_df, topics, tid, rep)
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
        self.calls = []
        self.stats = {"calls": 0, "prompts": 0, "seconds": 0.0, "tokens_out": 0, "truncated": 0}

    def describe(self):
        return {"llm_model": "stub", "llm_revision": None, "device": "cpu", "dtype": "none", "decoding": "greedy", "batch_size": 1}

    def generate(self, requests, max_new_tokens, seeds=None, meta=None):
        out = []
        for (sysm, user), m in zip(requests, meta):
            self.calls.append(m)
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
            out.append({"text": text, "truncated": text == "TRUNC", "n_tokens": 10, "seed": None, "seconds": 0.0})
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
        cats = G.run_taxonomy(tmp, 1, lambda t: emb[int(t)], StubLLM(s), "2010-01-01", "2010-09-30")
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
        cats = G.run_taxonomy(tmp, 4, lambda t: emb[int(t)], StubLLM({}, taxonomy_fn=_taxonomy_answer), "2014-10-01", "2018-01-31")
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
        s = {("bin", a, 1, r): "garbage" for a in (A, B) for r in (False, True)}
        cats = G.run_taxonomy(tmp, 1, lambda t: np.eye(4)[int(t)], StubLLM(s), "2010-01-01", "2010-09-30")
        assert len(cats) == 0 and (tmp / "categories.csv").exists()
        tc = pd.read_csv(tmp / "topic_categories.csv", keep_default_na=False)
        assert sorted(tc["topic_id"]) == [0, 1, 2, 3] and set(tc["category_assignment"]) == {"unplaced"}
        assert T.bin_state(tmp) == "labelled"
    finally:
        shutil.rmtree(tmp)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            check(name, fn)
    n_fail = sum(1 for _, ok, _ in RESULTS if not ok)
    print(f"\n{len(RESULTS) - n_fail}/{len(RESULTS)} checks passed")
    sys.exit(1 if n_fail else 0)
