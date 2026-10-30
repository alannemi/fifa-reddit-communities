#!/usr/bin/env python
"""
Unit checks for the analytical pipeline — no GPU, no language model (session 05, review R6; extended in sessions 06–08).
Run:  .venv/bin/python code/tests/test_pipeline_units.py        (exit code 1 if any check fails)
Covers: the path root (PIPELINE_ROOT) and the tone vocabulary; period-bin boundaries; the topic-density guard (D8), the
HDBSCAN grid diagnostic and the per-period-type settings; the frozen UMAP/cluster components inside BERTopic, the size
renumbering and the K1 topic counting (a dual post recorded, not counted); JSON extraction, label validation and normalisation; the topic overrides (incl. merge chains); outlier
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
def test_script_names_defined():
    """Every global name that a function of the stage script or the review-pack script reads must be defined at module
    level (import, def, assignment) or be a builtin. Added 2026-10-25 after a rewrite of stage_reassign had taken the
    LLM loader between two functions with it (NameError: get_llm, found only when Colab reached the label stage)."""
    import builtins, symtable
    for script in ("05-analysis-pipeline.py", "08-review-pack.py"):
        src = (Path(__file__).resolve().parents[1] / script).read_text(encoding="utf-8")
        top = symtable.symtable(src, script, "exec")
        defined = {sym.get_name() for sym in top.get_symbols() if sym.is_assigned() or sym.is_imported() or sym.is_namespace()}
        defined |= set(dir(builtins)) | {"__file__", "__name__", "__doc__"}
        missing = set()
        def walk(tbl):
            for child in tbl.get_children():
                for sym in child.get_symbols():
                    if sym.is_global() and sym.is_referenced() and sym.get_name() not in defined:
                        missing.add(f"{child.get_name()}: {sym.get_name()}")
                walk(child)
        walk(top)
        assert not missing, f"{script}: undefined global name(s) {sorted(missing)}"


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


def test_tone_vocabulary_e6():
    # E6 (2026-10-27): four categories from "other"; the earlier stems keep priority, among the new ones the first written wins
    assert {cat for _, cat in C.TONE_STEMS_ORDERED} == {"competitive", "informative", "analytical", "concerned"} <= set(C.TONE_SET)
    cases = {"analytical and competitive": "analytical", "informative and analytical": "informative",
             "confident and competitive": "competitive", "concern and urgency": "concerned", "Informed": "informative",
             "Frustration and anger": "frustrated", "critical and analytical": "critical", "excited, competitive": "excited",
             "nostalgic": "other"}
    assert {x: G.tone_category(x) for x in cases} == cases
    # retone_labels: rewrites only tone_category, only from "other", and refuses anything else
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        recs = [{"topic_id": i, "label": {"name": f"T{i}", "emotion_tone": tone}, "round_iterations": 1, "agreed": True,
                 "agreement_mode": "identical", "needs_review": False, "analyst_a_final": {}, "analyst_b_final": {}}
                for i, tone in enumerate(["excited", "competitive", "nostalgic"])]
        (d / "labels.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")
        old = G.labelled_table({r["topic_id"]: r for r in recs}); old.loc[1, "tone_category"] = "other"
        old.to_csv(d / "topics_labelled.csv", index=False)
        assert G.retone_labels(d, 0) == {("other", "competitive"): 1}
        assert pd.read_csv(d / "topics_labelled.csv", keep_default_na=False)["tone_category"].tolist() == ["excited", "competitive", "other"]
        assert G.retone_labels(d, 0) == {}                                            # idempotent
        old.loc[1, "tone_category"] = "critical"; old.to_csv(d / "topics_labelled.csv", index=False)
        try:
            G.retone_labels(d, 0); raise AssertionError("a move out of an existing category must be refused")
        except RuntimeError:
            pass


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
    assert r["floor_bound"] is True and len(r["trials"]) == 1, r          # decision D10: too few topics at the floor -> floor-bound, one fit
    r = M.guard_min_cluster_size(X4, 4000, True, fit=stub2)
    assert r["floor_bound"] is False                                    # unconverged with too MANY topics is not floor-bound
    r = M.guard_min_cluster_size(X4, 4000, False, fit=stub2)   # disabled: one fit
    assert len(r["trials"]) == 1 and r["enabled"] is False and r["floor_bound"] is False
    # the guard state as the gate and `--stage status` read it (tagging.guard_state), incl. a config written before the key existed (bin 7)
    from topicmodel import tagging as T
    assert T.guard_state({"guard": {"converged": True, "enabled": True}}) == "converged"
    assert T.guard_state({"guard": {"converged": False, "floor_bound": True, "enabled": True}}) == "floor-bound"
    bin7 = {"hdbscan": {"min_cluster_floor": 25}, "guard": {"min_cluster_size": 25, "n_topics": 172, "target": [207, 310], "converged": False, "enabled": True}}
    assert T.guard_state(bin7) == "floor-bound" and T.gate_ok(bin7 | {"outlier_share_raw": 0.356}, 0.50) is True
    assert T.guard_state({"hdbscan": {"min_cluster_floor": 25}, "guard": {"min_cluster_size": 547, "n_topics": 74, "target": [1656, 2484], "converged": False, "enabled": True}}) == "unconverged"
    assert T.guard_state({"guard": {"converged": False, "enabled": False}}) == "disabled" and T.guard_state({}) == "unknown"
    assert T.gate_ok({"outlier_share_raw": 0.562, "guard": {"converged": True, "enabled": True}}, 0.50) is False
    # the label record survives a re-run with nothing to do (bins 3 and 5 on Drive, 2026-10-25); a real run writes its own
    real = {"llm_stats": {"calls": 92, "prompts": 1415, "seconds": 947.9, "tokens_out": 182029, "truncated": 5}, "elapsed_min": 16.2,
            "run_at": "2026-10-25T20:26:18", "n_needs_review": 7}
    noop = {"llm_stats": {"calls": 0, "prompts": 0, "seconds": 0.0, "tokens_out": 0, "truncated": 0}, "elapsed_min": 0.2,
            "run_at": "2026-10-25T21:27:10", "n_needs_review": 7}
    m = T.merge_label_record(real, noop)
    assert m["llm_stats"]["calls"] == 92 and m["elapsed_min"] == 16.2 and m["run_at"] == "2026-10-25T20:26:18" and m["refreshed_at"] == "2026-10-25T21:27:10"
    assert T.merge_label_record(None, noop) == noop and T.merge_label_record(real, real | {"run_at": "later"})["run_at"] == "later"
    # a relabel of chosen topics keeps the full run's statistics and appends itself to `relabels`
    part = {"llm_stats": {"calls": 2, "prompts": 4, "seconds": 20.0, "tokens_out": 1500, "truncated": 0}, "elapsed_min": 0.9,
            "run_at": "2026-10-26T21:00:00", "n_needs_review": 6, "max_new_tokens": 900}
    m = T.merge_label_record(real, part, relabelled=[67])
    assert m["llm_stats"]["calls"] == 92 and m["run_at"] == "2026-10-25T20:26:18" and m["n_needs_review"] == 6
    assert m["relabels"] == [{"topics": [67], "max_new_tokens": 900, "llm_stats": part["llm_stats"], "elapsed_min": 0.9, "run_at": "2026-10-26T21:00:00"}]
    m2 = T.merge_label_record(m, noop)                       # a later no-op keeps the relabel entry too
    assert m2["relabels"] == m["relabels"] and m2["llm_stats"]["calls"] == 92
    m3 = T.merge_label_record(m2, {"n_needs_review": 5, "agreement_modes": {"x": 1}, "run_at": "2026-10-26T22:00:00"}, salvaged=[369])
    assert m3["llm_stats"]["calls"] == 92 and m3["n_needs_review"] == 5 and m3["salvages"] == [{"topics": [369], "run_at": "2026-10-26T22:00:00"}] and m3["relabels"] == m["relabels"]
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
    assert list(near.columns) == M.NEAREST_COLUMNS and len(near) == 10 and (near["nearest_topic"] == 0).all()
    assert (near["nearest_topic_2"] != near["nearest_topic"]).all() and (near["similarity_2"] <= near["similarity"]).all()
    tid, ass, sim, tid2, sim2 = M.apply_outlier_threshold(topics, ids, near, T=0.0)        # one tier (the D3 form)
    assert (ass[:10] == "reassigned").all() and (tid[:10] == 0).all() and (tid2 == -1).all() and np.isnan(sim2).all()
    tid, ass, sim, tid2, sim2 = M.apply_outlier_threshold(topics, ids, near, T=2.0)
    assert (ass[:10] == "outlier").all() and (tid[:10] == -1).all() and (ass[10:] == "cluster").all()
    tid, ass, sim, tid2, sim2 = M.apply_outlier_threshold(topics, ids, near, T=None)
    assert (ass[:10] == "outlier").all() and not np.isnan(sim[:10]).any()
    # D9: T and C from the bin's own similarity quantiles; between them a post is dual (nearest + second-nearest topic)
    T, Cc = M.tier_thresholds(near)
    s = near["similarity"].to_numpy(dtype=float)
    assert T == round(float(np.quantile(s, C.OUTLIER_REASSIGN_QUANTILE)), 4) and Cc == round(float(np.quantile(s, C.OUTLIER_SINGLE_QUANTILE)), 4)
    assert M.tier_thresholds(near, fixed_T=0.1)[0] == 0.1 and M.tier_thresholds(near.iloc[0:0]) == (None, None)
    tid, ass, sim, tid2, sim2 = M.apply_outlier_threshold(topics, ids, near, T, Cc)
    n_out, n_dual, n_single = int((ass[:10] == "outlier").sum()), int((ass[:10] == "dual").sum()), int((ass[:10] == "reassigned").sum())
    assert n_out + n_dual + n_single == 10 and n_single >= 2 and n_dual >= 1 and n_out >= 1, (n_out, n_dual, n_single)
    by_id = near.set_index("id")
    for k in range(10):
        if ass[k] == "dual":
            assert tid[k] == 0 and tid2[k] == by_id.loc[ids[k], "nearest_topic_2"] and sim2[k] == by_id.loc[ids[k], "similarity_2"]
        else:
            assert tid2[k] == -1 and np.isnan(sim2[k])
    tbl = M.calibration_table(near, len(docs))
    assert list(tbl["quantile"]) == list(C.OUTLIER_CALIBRATION_QUANTILES)
    assert tbl["role"].tolist()[0].startswith("T") and tbl["role"].tolist()[-1].startswith("C") and tbl["role"].tolist()[1] == ""
    # the judgement samples: one per candidate band; the nearest topic's name beside its id (empty until the bin is labelled)
    near2 = pd.DataFrame({"id": [f"o{i}" for i in range(20)], "nearest_topic": [0] * 10 + [1] * 10, "similarity": np.linspace(0.3, 0.9, 20).round(4)})
    tbl2 = M.calibration_table(near2, 100)
    samp = M.calibration_samples(near2, {f"o{i}": f"title {i}" for i in range(20)}, {0: "a, b", 1: "c, d"}, tbl2, n=5, names_of={1: "Topic one"})
    assert list(samp) == tbl2["threshold"].tolist() and all(len(x) > 0 for x in samp.values())
    for x in samp.values():
        assert list(x.columns)[:5] == ["id", "title", "nearest_topic", "nearest_topic_name", "nearest_topic_keywords"] and (x["belongs"] == "").all()
        assert all(nm == ("Topic one" if t == 1 else "") for t, nm in zip(x["nearest_topic"], x["nearest_topic_name"]))
    assert all((x["nearest_topic_name"] == "").all() for x in M.calibration_samples(near2, {}, {}, tbl2, n=5).values())
    docs_df = pd.DataFrame({"id": ids, "title": docs, "post_type": "regular", "lang": "en", "num_comments": "3"})
    tt = M.topic_table(bt, docs_df, topics, tid, rep, tid2, ass).set_index("topic_id")
    assert list(tt.reset_index().columns) == M.TOPIC_COLUMNS and set(tt.index) == {-1, 0, 1, 2}
    # K1 counting (2026-10-30): n_total = n_cluster + n_reassigned; a dual post is recorded (n_dual_first / n_dual_second), counted in neither topic
    assert tt.loc[0, "n_cluster"] == 110 and tt.loc[0, "n_reassigned"] == n_single and tt.loc[0, "n_dual_first"] == n_dual and tt.loc[0, "n_total"] == 110 + n_single
    assert tt["n_dual_second"].sum() == n_dual and tt.loc[0, "n_dual_second"] == 0
    assert (tt.drop(index=-1)["n_total"] == tt.drop(index=-1)["n_cluster"] + tt.drop(index=-1)["n_reassigned"]).all()
    assert tt.loc[-1, "n_cluster"] == 10 and tt.loc[-1, "n_total"] == n_out                    # the -1 row: raw outliers, posts left as outliers
    assert abs(tt.drop(index=-1)["share"].sum() - (230 + n_single) / 240) < 1e-4 and tt.drop(index=-1)["share"].sum() <= 1   # shares rounded to 5 decimals
    assert tt.loc[0, "share_lang_en"] == 1.0 and tt.loc[0, "mean_comments"] == 3.0
    tt2 = M.topic_table(bt, docs_df, topics, tid, rep, tid2).set_index("topic_id")               # the assignment derived from the ids
    assert tt2[M.TOPIC_COUNT_COLUMNS].equals(tt[M.TOPIC_COUNT_COLUMNS]) and (M.derive_assignment(topics, tid, tid2) == ass).all()


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
        # relabel one topic with a larger answer budget (decision of 2026-10-26, F45): its old transcript is superseded, not deleted;
        # the other topics keep their labels; labels.jsonl holds one line per topic; the budget is recorded per topic
        llm.script.update({(1, A, 1, False): label_json("One redone"), (1, B, 1, False): label_json("one redone")})
        n_calls = len(llm.calls)
        out4 = G.run_labelling(tmp, 1, docs, topics, llm, redo=[1], max_new_tokens=900)
        assert out4["topic_id"].tolist() == [0, 1, 2] and out4["name"].tolist() == ["Zero", "One redone", "Two"], out4[["topic_id", "name"]].values.tolist()
        assert len(llm.calls) == n_calls + 2 and len(G.read_jsonl(tmp / "labels.jsonl")) == 3
        sup = list((tmp / "labels" / "superseded").glob("topic_1_*.json")); assert len(sup) == 1 and json.loads(sup[0].read_text())["label"]["name"] == "One B"
        rec1 = json.loads((tmp / "labels" / "topic_1.json").read_text())
        assert rec1["max_new_tokens"] == 900 and rec1["relabelled"] is True and rec1["agreement_mode"] == "identical"
        assert json.loads((tmp / "labels" / "topic_0.json").read_text())["relabelled"] is False
        # decision (a) of 2026-10-26 (F45): a no_answer topic whose analysts were both cut off gets its label from the complete
        # fields of their last answers — nothing invented, the cut field left out, the transcript kept, the old record superseded
        cut_a = '{"decision": "revise", "categories": [{"name": "Rayo Vallecano vs. Clubs", "purpose": "Matches involving Rayo", "emotion_tone": "Excitement", "description": "Posts about Rayo Vallecano matc'
        cut_b = '{"decision": "revise", "categories": [{"name": "rayo vallecano vs. clubs", "purpose": "Matches between Rayo and other clubs", "emotion_tone": "Excitement", "description": "Match threads.", "inclusion_cues": ["Rayo Vallecano", "match"], "x": "y'
        f = salvage_truncated_fields = G.salvage_truncated_fields
        assert f(cut_a) == {"name": "Rayo Vallecano vs. Clubs", "purpose": "Matches involving Rayo", "emotion_tone": "Excitement"}
        assert f(cut_b)["inclusion_cues"] == ["Rayo Vallecano", "match"] and f(cut_b)["description"] == "Match threads."
        rec2 = json.loads((tmp / "labels" / "topic_2.json").read_text())
        rec2.update({"label": None, "analyst_a_final": None, "analyst_b_final": None, "agreed": False, "agreement_mode": "no_answer", "needs_review": True,
                     "transcript": [{"analyst": A, "round": 1, "repair": False, "answer": cut_a, "truncated": True},
                                    {"analyst": B, "round": 1, "repair": False, "answer": cut_b, "truncated": True}]})
        (tmp / "labels" / "topic_2.json").write_text(json.dumps(rec2), encoding="utf-8")
        lines = [json.loads(l) for l in (tmp / "labels.jsonl").read_text().splitlines()]
        for l in lines:
            if l["topic_id"] == 2: l.update({"label": None, "agreement_mode": "no_answer", "agreed": False, "needs_review": True})
        (tmp / "labels.jsonl").write_text("".join(json.dumps(l) + "\n" for l in lines), encoding="utf-8")
        out5 = G.salvage_labels(tmp, 1, [2, 0])                            # topic 0 has a label: skipped
        r2 = out5[out5.topic_id == 2].iloc[0]
        assert r2["name"] == "Rayo Vallecano vs. Clubs" and r2["emotion_tone"] == "Excitement" and r2["tone_category"] == "excited" and r2["description"] == ""
        assert r2["agreement_mode"] == "salvaged_truncated" and bool(r2["agreed"]) is True and bool(r2["needs_review"]) is True
        assert out5[out5.topic_id == 0].iloc[0]["name"] == "Zero" and len(list((tmp / "labels" / "superseded").glob("topic_2_*.json"))) == 1
        rec2b = json.loads((tmp / "labels" / "topic_2.json").read_text())
        assert rec2b["label"]["name"] == "Rayo Vallecano vs. Clubs" and len(rec2b["transcript"]) == 2 and rec2b["salvaged_fields"]["A"] == ["emotion_tone", "name", "purpose"]
        assert json.loads(rec2b["analyst_b_final"] if isinstance(rec2b["analyst_b_final"], str) else json.dumps(rec2b["analyst_b_final"]))["categories"][0]["description"] == "Match threads."
        assert [l["topic_id"] for l in (json.loads(x) for x in (tmp / "labels.jsonl").read_text().splitlines())] == [0, 1, 2]
        assert G.salvage_no_answer({"transcript": [{"analyst": A, "answer": '{"decision": "revise", "categories": [{"nam'}]}) is None
        # a refit retires the previous clustering's labels, review files and calibration samples (nothing deleted), so a
        # resume cannot attach old labels to renumbered topics; labelling then starts from scratch
        (tmp / "outlier_calibration").mkdir(); (tmp / "outlier_calibration" / "sample_T0.5000.csv").write_text("id\n", encoding="utf-8")
        (tmp / "REVIEW_READY.txt").write_text("x", encoding="utf-8")
        stale = Path(tempfile.mkdtemp())
        try:
            dest = T.retire_label_outputs(tmp, stale)
            assert dest is not None and dest.parent == stale and dest.name.startswith(tmp.name + "_")
            assert not any((tmp / n).exists() for n in T.STALE_ON_REFIT)
            assert (dest / "labels" / "topic_0.json").exists() and (dest / "labels.jsonl").exists() and (dest / "topics_labelled.csv").exists()
            assert (dest / "topic_overrides.csv").exists() and (dest / "outlier_calibration" / "sample_T0.5000.csv").exists() and (dest / "README.txt").exists()
            assert T.retire_label_outputs(tmp, stale) is None
            n_before = len(llm.calls)
            out4 = G.run_labelling(tmp, 1, docs, topics, llm)
            assert out4["topic_id"].tolist() == [0, 1, 2] and len(llm.calls) > n_before and (tmp / "labels.jsonl").exists()
        finally:
            shutil.rmtree(stale)
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
                      "topic_id_2": ["", "", "", "0", "", ""], "topic_assignment": ["cluster", "cluster", "cluster", "dual", "outlier", "not_modelled"],
                      "similarity": ["", "", "", "0.6", "0.2", ""], "similarity_2": ["", "", "", "0.55", "", ""]}).to_csv(b / "documents.csv", index=False)
        pd.DataFrame({"topic_id": [-1, 0, 1], "n_cluster": [2, 2, 1], "n_total": [1, 2, 2], "share": [0.2, 0.4, 0.4], "keywords": ["", "a, b", "c, d"], "representative_ids": ["", "p0", "p2"]}).to_csv(b / "topics.csv", index=False)
        pd.DataFrame({"topic_id": [0, 1], "keywords": ["a, b", "c, d"], "name": ["Zero", "One"], "purpose": ["p", "p"], "emotion_tone": ["excited", "sad"], "tone_category": ["excited", "sad"],
                      "description": ["d", "d"], "inclusion_cues": ["x", "y"], "round_iterations": [1, 2], "agreed": [True, False], "agreement_mode": ["identical", "cap"],
                      "needs_review": [False, True], "analyst_a_final": ["{}", "{}"], "analyst_b_final": ["{}", "{}"], "llm_model": ["stub"] * 2, "llm_revision": ["", ""],
                      "prompt_version": ["1.4"] * 2, "decoding": ["greedy"] * 2}).to_csv(b / "topics_labelled.csv", index=False)
        pd.DataFrame({"topic_id": [1], "name": ["One"], "needs_review": [True], "final_name": ["One (final)"], "merge_into": [""], "notes": [""]}).to_csv(b / "topic_overrides.csv", index=False)
        # the outlier judgement samples get the nearest topic's name (renames applied); filled cells survive; idempotent
        (b / "outlier_calibration").mkdir()
        pd.DataFrame({"id": ["p4", "p9"], "title": ["t4", "t9"], "nearest_topic": [1, 0], "nearest_topic_keywords": ["c, d", "a, b"], "similarity": [0.55, 0.57],
                      "candidate_threshold": [0.5, 0.5], "band_upper": ["", ""], "belongs": ["yes", ""], "notes": ["", ""]}).to_csv(b / "outlier_calibration" / "sample_T0.5000.csv", index=False)
        coded = b / "outlier_calibration" / "sample_T0.5000-coded-XX.csv"                     # a returned copy, any coder suffix: never rewritten
        coded_bytes = b"id,belongs,title,nearest_topic\r\np4,1,t4,1\r\n"; coded.write_bytes(coded_bytes)
        assert T.topic_names(b) == {0: "Zero", 1: "One (final)"} and T.topic_names(an / "bin-2") == {} and T.annotate_calibration_samples(an / "bin-2") == 0
        assert T.annotate_calibration_samples(b) == 1 and T.annotate_calibration_samples(b) == 1
        cs = pd.read_csv(b / "outlier_calibration" / "sample_T0.5000.csv", dtype=str, keep_default_na=False)
        assert list(cs.columns) == ["id", "title", "nearest_topic", "nearest_topic_name", "nearest_topic_keywords", "similarity", "candidate_threshold", "band_upper", "belongs", "notes"]
        assert cs["nearest_topic_name"].tolist() == ["One (final)", "Zero"] and cs["belongs"].tolist() == ["yes", ""] and cs["similarity"].tolist() == ["0.55", "0.57"]
        assert coded.read_bytes() == coded_bytes                                              # byte for byte, CRLF included
        side = T.collect(an, ad, allow_partial=True, run_info={})
        pv = pd.read_csv(ad / "post_variables.csv", dtype=str, keep_default_na=False)
        assert list(pv.columns) == T.POST_COLUMNS + T.AUDIT_COLUMNS and len(pv) == 10 and "category_id" not in pv.columns
        assert pv["topic_id"].tolist()[:6] == ["0", "0", "1", "1", "-1", ""] and pv["topic_id"].tolist()[6:] == [""] * 4
        assert pv["post_sentiment"].tolist()[0] == "0.5000" and pv["topic_id_raw"].tolist()[3] == "-1"
        assert pv["topic_name"].tolist()[:6] == ["Zero", "Zero", "One (final)", "One (final)", "", ""] and pv["topic_tone"].tolist()[:4] == ["excited", "excited", "sad", "sad"]
        assert pv["topic_uid"].tolist()[:6] == ["1-0", "1-0", "1-1", "1-1", "", ""]
        # the dual post (D9): second topic with uid, name, tone and both similarities; empty everywhere else
        assert pv["topic_assignment"].tolist()[:6] == ["cluster", "cluster", "cluster", "dual", "outlier", "not_modelled"]
        assert pv["topic_uid_2"].tolist()[:6] == ["", "", "", "1-0", "", ""] and pv["topic_name_2"].tolist()[3] == "Zero" and pv["topic_tone_2"].tolist()[3] == "excited"
        assert pv["topic_similarity"].tolist()[:6] == ["", "", "", "0.6", "0.2", ""] and pv["topic_similarity_2"].tolist()[:6] == ["", "", "", "0.55", "", ""]
        assert side["bins_complete"] == [1] and side["complete"] is False and side["errors"] == [] and side["n_topics_labelled"] == 2
        lab = pd.read_csv(an / "topic_labels_all_bins.csv", dtype=str, keep_default_na=False)
        assert list(lab.columns) == T.LABEL_COLUMNS and lab.loc[lab["topic_id"] == "1", "name"].iloc[0] == "One (final)"
        assert lab["topic_uid"].tolist() == ["1-0", "1-1"]
        # K1 counting (2026-10-30): the dual post p3 (first topic 1, second topic 0) is recorded in both topics and counted in neither
        assert lab["n_cluster"].tolist() == ["2", "1"] and lab["n_reassigned"].tolist() == ["0", "0"]
        assert lab["n_dual_first"].tolist() == ["0", "1"] and lab["n_dual_second"].tolist() == ["1", "0"] and lab["n_total"].tolist() == ["2", "1"]
        assert lab["share"].tolist() == ["0.4", "0.2"]                     # over the 5 modelled posts; p3 (dual) and p4 (outlier) count nowhere
        # the mean VADER score per topic (title score) over the counted posts: topic 0 = p0, p1; topic 1 = p2 (p3 is dual: in neither)
        assert lab["mean_vader_sentiment"].tolist() == ["0.0000", "0.0000"] and lab["n_posts_scored"].tolist() == ["2", "1"]
        assert side["topic_count_assignments"] == ["cluster", "reassigned"]
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
        assert T.topic_names(b) == {0: "Zero", 1: "Zero"} and T.annotate_calibration_samples(b) == 1     # a merged-away topic carries its target's name
        cs = pd.read_csv(b / "outlier_calibration" / "sample_T0.5000.csv", dtype=str, keep_default_na=False)
        assert cs["nearest_topic_name"].tolist() == ["Zero", "Zero"] and list(cs.columns).count("nearest_topic_name") == 1 and cs["belongs"].tolist() == ["yes", ""]
        side = T.collect(an, ad, allow_partial=True, run_info={})
        pv = pd.read_csv(ad / "post_variables.csv", dtype=str, keep_default_na=False)
        assert pv["topic_id"].tolist()[:4] == ["0"] * 4 and pv["topic_id_raw"].tolist()[2] == "1" and side["errors"] == []
        assert pv["topic_name"].tolist()[:4] == ["Zero"] * 4 and pv["topic_tone"].tolist()[:4] == ["excited"] * 4
        # the dual post's second topic merged into its first: it becomes a plain reassignment
        assert pv["topic_assignment"].tolist()[3] == "reassigned" and pv["topic_uid_2"].tolist()[3] == "" and pv["topic_similarity_2"].tolist()[3] == ""
        lab = pd.read_csv(an / "topic_labels_all_bins.csv", dtype=str, keep_default_na=False)
        assert lab["topic_id"].tolist() == ["0"] and lab["merged_from"].tolist() == ["1"] and lab["n_total"].tolist() == ["4"]
        assert lab["n_cluster"].tolist() == ["3"] and lab["n_reassigned"].tolist() == ["1"] and lab["n_dual_first"].tolist() == ["0"] and lab["n_dual_second"].tolist() == ["0"]
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
