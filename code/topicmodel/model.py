"""
Stage S2 — BERTopic per period bin (design §6) and the outlier stage (decision D3, §6.3).
=========================================================================================
Inside BERTopic (every setting in config.py): pre-computed sentence embeddings -> UMAP to 5 dimensions, fitted ONCE
per bin and replayed through `FrozenUMAP` -> HDBSCAN labels chosen by the topic-count guard and injected through
`FrozenCluster` (refinement N2) -> CountVectorizer + c-TF-IDF (distinctive words per topic) -> KeyBERTInspired +
MaximalMarginalRelevance (readable, diverse keywords).
Around it: the guard (§6.2), the renumbering of cluster labels by size (review R1), the representative posts (R3),
the outlier similarities and the calibration samples for the threshold T (N1), the thresholded reassignment, and the
per-bin tables topics.csv / topics_over_time.csv (§6.4).  `save_model` / `load_model` keep the fitted model in
safetensors format; UMAP and HDBSCAN are not part of it — the persisted .npy arrays are the replay record (§11).
"""
from __future__ import annotations
import logging
import time
from collections import Counter
import numpy as np
import pandas as pd
from . import config as C

log = logging.getLogger("topicmodel")


# ---------------------------------------------------------------------------
# Frozen components (N2): BERTopic receives the 5-d array and the labels instead of refitting them
# ---------------------------------------------------------------------------
class FrozenUMAP:
    """Stands in for UMAP inside BERTopic: the 5-d array was computed once per bin (`fit_umap`) and is replayed,
    so the topic-count guard and any re-run never refit UMAP (design §6.1)."""

    def __init__(self, reduced: np.ndarray):
        self.embedding_ = np.asarray(reduced, dtype=np.float32)

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if len(X) != len(self.embedding_):
            raise ValueError(f"FrozenUMAP holds {len(self.embedding_)} rows, got {len(X)} documents")
        return self.embedding_

    def fit_transform(self, X, y=None):
        return self.transform(X)


class FrozenCluster:
    """Stands in for HDBSCAN inside BERTopic: the labels chosen by the guard are injected unchanged (N2)."""

    def __init__(self, labels: np.ndarray):
        self.labels_ = np.asarray(labels, dtype=int)

    def fit(self, X, y=None):
        if len(X) != len(self.labels_):
            raise ValueError(f"FrozenCluster holds {len(self.labels_)} labels, got {len(X)} rows")
        return self


def detect_backend() -> dict:
    """Which implementation of UMAP/HDBSCAN will run: NVIDIA cuML (under `python -m cuml.accel`) or CPU (R8)."""
    info = {"backend": "cpu", "cuml_accel": False, "cuml_version": None, "umap_class": None, "hdbscan_class": None}
    try:
        import cuml
        info["cuml_version"] = getattr(cuml, "__version__", "unknown")
        try:
            from cuml.accel import enabled
            info["cuml_accel"] = bool(enabled())
        except Exception:
            pass
    except Exception:
        pass
    try:
        from umap import UMAP
        from hdbscan import HDBSCAN
        info["umap_class"] = f"{UMAP.__module__}.{UMAP.__name__}"
        info["hdbscan_class"] = f"{HDBSCAN.__module__}.{HDBSCAN.__name__}"
    except Exception as e:
        info["import_error"] = str(e)
    if info["cuml_accel"] or "cuml" in str(info["umap_class"]):
        info["backend"] = "cuml"
    return info


# ---------------------------------------------------------------------------
# UMAP once per bin, HDBSCAN with the topic-count guard
# ---------------------------------------------------------------------------
def fit_umap(emb: np.ndarray) -> np.ndarray:
    """One UMAP fit per bin on the 768-d embeddings (seeded; GPU under cuml.accel). Returns the 5-d array."""
    from umap import UMAP
    t0 = time.time()
    red = UMAP(**C.UMAP_PARAMS).fit_transform(emb)
    red = np.nan_to_num(np.asarray(red, dtype=np.float32))
    log.info("UMAP: %d x %d -> %d dims in %.1f min", emb.shape[0], emb.shape[1], red.shape[1], (time.time() - t0) / 60)
    return red


def hdbscan_settings(period_type: str | None = None) -> dict:
    """The HDBSCAN choices in force for a bin of this period type and the density target (recorded in config.json)."""
    return {"min_samples": int(C.setting(C.HDBSCAN_MIN_SAMPLES, period_type)), "method": C.setting(C.HDBSCAN_SELECTION_METHOD, period_type),
            "epsilon": float(C.setting(C.HDBSCAN_EPSILON, period_type)), "topics_per_10k": list(C.TOPICS_PER_10K),
            "min_cluster_floor": int(C.MIN_CLUSTER_FLOOR)}


def hdbscan_labels(umap5d: np.ndarray, min_cluster_size: int, min_samples: int | None = None, method: str | None = None,
                   epsilon: float | None = None, period_type: str | None = None) -> np.ndarray:
    """One HDBSCAN fit on the frozen 5-d array; the three choices default to the settings in force for the bin's period
    type (design §6.1; per-type values since 2026-10-25)."""
    from hdbscan import HDBSCAN
    d = hdbscan_settings(period_type)
    kw = dict(min_cluster_size=int(min_cluster_size), min_samples=int(d["min_samples"] if min_samples is None else min_samples),
              metric="euclidean", cluster_selection_method=method or d["method"])
    eps = d["epsilon"] if epsilon is None else epsilon
    if eps:
        kw["cluster_selection_epsilon"] = float(eps)
    return np.asarray(HDBSCAN(**kw).fit(umap5d).labels_, dtype=int)


def cluster_stats(labels: np.ndarray) -> dict:
    """Topic count, outlier share, share of the largest cluster and the median cluster size of one labelling."""
    lab = labels[labels != -1]
    sizes = np.bincount(lab) if len(lab) else np.array([0])
    sizes = sizes[sizes > 0]
    return {"n_topics": int(len(sizes)) if len(lab) else 0, "outlier_share": round(float((labels == -1).mean()), 4),
            "largest_share": round(float(sizes.max() / max(1, len(labels))), 4) if len(lab) else 0.0,
            "median_size": int(np.median(sizes)) if len(lab) else 0}


def hdbscan_grid(umap5d: np.ndarray, grid: dict | None = None, fit=hdbscan_labels, period_type: str | None = None) -> pd.DataFrame:
    """The diagnostic of a failed large-bin gate (design §12.2): HDBSCAN refitted on the cached UMAP array over
    config.HDBSCAN_GRID (min_cluster_size, min_samples, selection method, epsilon). One row per fit with the cluster
    statistics and whether the topic count lies in the bin's density target (config.target_topic_range).
    `fit(umap5d, mcs, min_samples, method, epsilon)` is injectable for tests."""
    import time as _time
    g = {**C.HDBSCAN_GRID, **(grid or {})}
    lo, hi = C.target_topic_range(len(umap5d))
    rows = []
    for mcs in g["min_cluster_size"]:
        mcs = int(mcs)
        for ms in g["min_samples"]:
            for method in g["method"]:
                for eps in g["epsilon"]:
                    t0 = _time.time()
                    st = cluster_stats(fit(umap5d, mcs, int(ms), method, float(eps)))
                    rows.append({"min_cluster_size": mcs, "min_samples": int(ms), "method": method, "epsilon": float(eps), **st,
                                 "in_range": lo <= st["n_topics"] <= hi, "seconds": round(_time.time() - t0, 1)})
                    log.info("grid: mcs=%d ms=%d %s eps=%.2f -> %d topics, %.1f %% outliers, largest %.1f %%", mcs, ms, method, eps,
                             st["n_topics"], 100 * st["outlier_share"], 100 * st["largest_share"])
    return pd.DataFrame(rows)


def n_topics_of(labels: np.ndarray) -> int:
    return int(len(set(labels.tolist()) - {-1}))


def guard_min_cluster_size(umap5d: np.ndarray, n_modelled: int, enabled: bool = True, fit=hdbscan_labels,
                           period_type: str | None = None) -> dict:
    """Topic-density guard (design §6.2, decision D8): start at MIN_CLUSTER_FLOOR, then move ×1.5 / ÷1.5 and bisect
    between the bracketing values until the raw topic count lies in the bin's target range — TOPICS_PER_10K scaled to
    its modelled posts (config.target_topic_range) — within GUARD_MAX_TRIALS fits, never below the floor. Keeps the
    trial nearest to the range; `converged` says whether it was reached; `floor_bound` (decision D10, 2026-10-25) says the floor
    itself gave too few topics, which the gate accepts. `fit` is injectable for tests."""
    lo, hi = C.target_topic_range(n_modelled)
    if fit is hdbscan_labels:
        fit = lambda X, m: hdbscan_labels(X, m, period_type=period_type)
    trials = []

    def run(m):
        lab = fit(umap5d, m)
        k = n_topics_of(lab)
        trials.append({"min_cluster_size": int(m), "n_topics": k, "outlier_share": round(float((lab == -1).mean()), 4)})
        log.info("guard: min_cluster_size=%d -> %d topics, %.1f %% outliers", m, k, 100 * (lab == -1).mean())
        return lab, k

    def dist(k):
        return 0 if lo <= k <= hi else (lo - k if k < lo else k - hi)

    mcs = int(C.MIN_CLUSTER_FLOOR)
    lab, k = run(mcs)
    best = (lab, mcs, k)
    if enabled and dist(k):
        too_many, too_few = (mcs, None) if k > hi else (None, mcs)   # sizes that gave too many / too few topics
        tried = {mcs}
        while len(trials) < C.GUARD_MAX_TRIALS:
            if too_many is not None and too_few is not None:
                m = (too_many + too_few) // 2
            elif too_many is not None:
                m = int(round(too_many * C.GUARD_STEP))
            else:
                m = max(C.MIN_CLUSTER_FLOOR, int(round(too_few / C.GUARD_STEP)))
            if m in tried:
                break
            tried.add(m)
            lab, k = run(m)
            if dist(k) < dist(best[2]):
                best = (lab, m, k)
            if dist(k) == 0:
                break
            if k > hi:
                too_many = m
            else:
                too_few = m
    lab, m, k = best
    conv = dist(k) == 0
    # Decision D10 (Alan Nemirovski, 2026-10-25, bin 7): too FEW topics at the floor itself — the guard cannot go lower (the floor is a
    # constraint of design §6.1, not a search bound), so the floor wins over the density target: the bin is "floor-bound", stands, and
    # passes the gate; `converged` stays False, honestly.
    floor_bound = bool(enabled and not conv and int(m) == int(C.MIN_CLUSTER_FLOOR) and k < lo)
    if floor_bound:
        log.warning("guard: %d topics at the floor min_cluster_size=%d, below the target %s — floor-bound (decision D10: the floor wins, the bin stands)", k, m, (lo, hi))
    elif enabled and not conv:
        log.warning("guard: topic count %d outside %s after %d trial(s); keeping min_cluster_size=%d (flagged)", k, (lo, hi), len(trials), m)
    return {"labels": lab, "min_cluster_size": int(m), "n_topics": k, "target": [int(lo), int(hi)], "trials": trials, "converged": conv,
            "floor_bound": floor_bound, "enabled": enabled}


def renumber_by_size(labels: np.ndarray) -> tuple[np.ndarray, dict]:
    """Renumber cluster labels by size (largest -> 0; ties by original label) so that BERTopic's own
    frequency sort is the identity up to ties (R1). Returns (new labels, {old: new})."""
    sizes = Counter(int(l) for l in labels if l != -1)
    order = sorted(sizes, key=lambda l: (-sizes[l], l))
    mp = {l: i for i, l in enumerate(order)}
    mp[-1] = -1
    return np.array([mp[int(l)] for l in labels], dtype=int), mp


# ---------------------------------------------------------------------------
# The BERTopic model of a bin
# ---------------------------------------------------------------------------
def build_topic_model(umap5d: np.ndarray, labels: np.ndarray, embedding_model_name: str, device: str):
    """BERTopic for one bin: frozen UMAP + frozen cluster labels, then the keyword chain of design §6.1
    (CountVectorizer with the stop-list -> c-TF-IDF -> KeyBERTInspired -> MaximalMarginalRelevance)."""
    from bertopic import BERTopic
    from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
    from bertopic.vectorizers import ClassTfidfTransformer
    from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS
    from .embed import load_embedding_model

    stop = list(ENGLISH_STOP_WORDS) + C.EXTRA_STOPWORDS
    vectorizer = CountVectorizer(ngram_range=C.NGRAM_RANGE, stop_words=stop, min_df=2)
    ctfidf = ClassTfidfTransformer(reduce_frequent_words=True)
    representation = [KeyBERTInspired(top_n_words=C.TOP_N_WORDS * 2),
                      MaximalMarginalRelevance(diversity=C.MMR_DIVERSITY, top_n_words=C.TOP_N_WORDS)]
    st_model = load_embedding_model(embedding_model_name, device)   # KeyBERTInspired embeds candidate words
    model = BERTopic(embedding_model=st_model, umap_model=FrozenUMAP(umap5d), hdbscan_model=FrozenCluster(labels),
                     vectorizer_model=vectorizer, ctfidf_model=ctfidf, representation_model=representation,
                     top_n_words=C.TOP_N_WORDS, calculate_probabilities=False, verbose=True)
    return model


def fit_model(model, docs_text: list[str], emb: np.ndarray) -> tuple[np.ndarray, dict]:
    """fit_transform with the frozen components; returns BERTopic's topic ids and the {injected: BERTopic} map."""
    topics, _ = model.fit_transform(docs_text, emb)
    mapping = {int(k): int(v) for k, v in model.topic_mapper_.get_mappings().items()}
    changed = {k: v for k, v in mapping.items() if k != v}
    if changed:
        log.info("BERTopic renumbered %d topic id(s) (ties in size): %s", len(changed), dict(list(changed.items())[:10]))
    return np.asarray(topics, dtype=int), mapping


def representative_ids(model, docs_text: list[str], ids: np.ndarray, topics: np.ndarray, emb: np.ndarray,
                       n: int = C.N_REPRESENTATIVE_DOCS) -> dict[int, list[str]]:
    """n representative posts per topic, nearest to the topic's c-TF-IDF representation (BERTopic computes
    only 3 by default, R3); fallback = members nearest the topic embedding."""
    out = {}
    tlist = sorted(set(topics.tolist()) - {-1})
    try:
        df = pd.DataFrame({"Document": docs_text, "ID": range(len(docs_text)), "Topic": topics})
        repr_map, _, _, _ = model._extract_representative_docs(model.c_tf_idf_, df, model.topic_representations_,
                                                                nr_samples=500, nr_repr_docs=n)
        for t in tlist:
            members = df[df["Topic"] == t]
            first_pos = {}
            for pos, text in zip(members["ID"], members["Document"]):
                first_pos.setdefault(text, pos)
            out[t] = [str(ids[first_pos[x]]) for x in repr_map.get(t, []) if x in first_pos][:n]
    except Exception as e:   # private API changed? fall back to the embedding-nearest members
        log.warning("representative docs via BERTopic failed (%s); using embedding-nearest members", e)
        out = {}
    if not out or any(len(v) == 0 for v in out.values()):
        te = np.asarray(model.topic_embeddings_)
        for t in tlist:
            if out.get(t):
                continue
            idx = np.where(topics == t)[0]
            sims = emb[idx] @ te[model._outliers + t] / (np.linalg.norm(te[model._outliers + t]) or 1.0)
            out[t] = [str(ids[i]) for i in idx[np.argsort(-sims)[:n]]]
    return out


# ---------------------------------------------------------------------------
# Outliers (D3 -> D9): the two nearest topics and similarities, calibration table and samples, tiered reassignment
# ---------------------------------------------------------------------------
NEAREST_COLUMNS = ["id", "nearest_topic", "similarity", "nearest_topic_2", "similarity_2"]


def outlier_nearest_from(topic_embeddings: np.ndarray, emb: np.ndarray, topics_raw: np.ndarray, ids: np.ndarray,
                         chunk: int = 50_000) -> pd.DataFrame:
    """For every raw outlier: the two most similar topics (cosine to the topic embeddings, row t = topic t) and the
    similarities (N1, D9). The second pair is empty when the bin has a single topic. Computed in row chunks of `chunk`
    outliers (2026-10-30, the reference run: an estimated 580k outliers × thousands of topics — the dense similarity matrix
    (≈ 14 GB) and its full argsort (≈ 28 GB) would not fit in Colab's RAM; a chunk of 50,000 × 6,000 is ≈ 1.2 GB)."""
    from sklearn.metrics.pairwise import cosine_similarity
    out_idx = np.where(topics_raw == -1)[0]
    if len(out_idx) == 0 or n_topics_of(topics_raw) == 0:
        return pd.DataFrame(columns=NEAREST_COLUMNS)
    te = np.asarray(topic_embeddings)
    k = te.shape[0]
    step = max(1, int(chunk))
    first, s1, second, s2 = [], [], [], []
    for a in range(0, len(out_idx), step):
        sims = cosine_similarity(emb[out_idx[a:a + step]], te)
        if k >= 2:
            top2 = np.argpartition(-sims, 1, axis=1)[:, :2]            # the two largest of each row, in either order
            v = np.take_along_axis(sims, top2, axis=1)
            swap = v[:, 1] > v[:, 0]
            top2[swap] = top2[swap][:, ::-1]
            v[swap] = v[swap][:, ::-1]
            first.append(top2[:, 0]); s1.append(v[:, 0]); second.append(top2[:, 1]); s2.append(v[:, 1])
        else:
            first.append(np.zeros(len(sims), dtype=int)); s1.append(sims[:, 0])
    df = pd.DataFrame({"id": ids[out_idx].astype(str), "nearest_topic": np.concatenate(first).astype(int),
                       "similarity": np.concatenate(s1).round(4)})
    if k >= 2:
        df["nearest_topic_2"] = np.concatenate(second).astype(int)
        df["similarity_2"] = np.concatenate(s2).round(4)
    else:
        df["nearest_topic_2"] = np.nan
        df["similarity_2"] = np.nan
    return df


def outlier_nearest(model, emb: np.ndarray, topics_raw: np.ndarray, ids: np.ndarray) -> pd.DataFrame:
    """outlier_nearest_from() on a fitted BERTopic model (stage S2); `reassign` uses the saved topic_embeddings.npy."""
    return outlier_nearest_from(np.asarray(model.topic_embeddings_)[model._outliers:], emb, topics_raw, ids)


def tier_thresholds(nearest: pd.DataFrame, fixed_T: float | None = None,
                    q_reassign: float = C.OUTLIER_REASSIGN_QUANTILE, q_single: float = C.OUTLIER_SINGLE_QUANTILE):
    """(T, C) of one bin (D9): T = the reassignment floor, C = the single-topic floor, both quantiles of the bin's
    outlier-to-nearest-topic similarity; `fixed_T` (config.OUTLIER_THRESHOLD / --outlier-threshold) replaces T.
    (None, None) when the bin has no outliers."""
    s = nearest["similarity"].to_numpy(dtype=float) if len(nearest) else np.array([])
    if len(s) == 0:
        return (None if fixed_T is None else float(fixed_T)), None
    T = float(fixed_T) if fixed_T is not None else round(float(np.quantile(s, q_reassign)), 4)
    Cc = round(float(np.quantile(s, q_single)), 4)
    return T, max(Cc, T)


def calibration_table(nearest: pd.DataFrame, n_modelled: int) -> pd.DataFrame:
    """The judgement bands = quantiles of the outlier-to-nearest-topic similarity (design §6.3): what each candidate
    floor would reassign, and which band is the floor T / the single-topic floor C in force."""
    rows = []
    s = nearest["similarity"].to_numpy(dtype=float) if len(nearest) else np.array([])
    for q in C.OUTLIER_CALIBRATION_QUANTILES:
        T = float(np.quantile(s, q)) if len(s) else float("nan")
        n_re = int((s >= T).sum()) if len(s) else 0
        role = ("T (reassignment floor)" if q == C.OUTLIER_REASSIGN_QUANTILE else
                "C (single-topic floor)" if q == C.OUTLIER_SINGLE_QUANTILE else "")
        rows.append({"quantile": q, "threshold": round(T, 4), "role": role, "n_outliers": len(s), "n_reassigned_at_T": n_re,
                     "share_outliers_remaining": round((len(s) - n_re) / max(1, n_modelled), 4)})
    return pd.DataFrame(rows)


def calibration_samples(nearest: pd.DataFrame, titles: dict, keywords_of: dict, table: pd.DataFrame,
                        n: int = C.OUTLIER_SAMPLE_PER_CANDIDATE, names_of: dict | None = None) -> dict[float, pd.DataFrame]:
    """One judgement sample per candidate threshold: posts whose similarity lies in the band between that
    candidate and the next higher one (so the sample shows what lowering the threshold to T would add).
    `nearest_topic_name` is the nearest topic's agreed name (`names_of`); at stage S2 no labels exist yet, so the column
    is empty and the label stage fills it in (tagging.annotate_calibration_samples; author's request, 2026-10-25).
    Column `belongs` is left blank for the co-author (yes / no)."""
    out = {}
    names_of = names_of or {}
    ths = table["threshold"].tolist()
    for i, T in enumerate(ths):
        hi = ths[i + 1] if i + 1 < len(ths) else 1.01
        band = nearest[(nearest["similarity"] >= T) & (nearest["similarity"] < hi)]
        samp = band.sample(min(n, len(band)), random_state=C.SEED) if len(band) else band
        out[T] = pd.DataFrame({"id": samp["id"], "title": [titles.get(i, "")[:200] for i in samp["id"]],
                               "nearest_topic": samp["nearest_topic"],
                               "nearest_topic_name": [names_of.get(int(t), "") for t in samp["nearest_topic"]],
                               "nearest_topic_keywords": [keywords_of.get(int(t), "") for t in samp["nearest_topic"]],
                               "similarity": samp["similarity"], "candidate_threshold": T, "band_upper": hi if hi <= 1 else "",
                               "belongs": "", "notes": ""})
    return out


def resolve_threshold(bin_id: int, override: float | None = None) -> float | None:
    if override is not None:
        return float(override)
    th = C.OUTLIER_THRESHOLD or {}
    v = th.get(int(bin_id), th.get(str(bin_id), th.get("default")))
    return None if v is None else float(v)


def apply_outlier_threshold(topics_raw: np.ndarray, ids: np.ndarray, nearest: pd.DataFrame, T: float | None,
                            C_single: float | None = None):
    """Tiered reassignment (D9): an outlier with similarity >= T joins its nearest topic — alone when the similarity
    reaches C_single ("reassigned"), with the second-nearest topic as a second label below it ("dual"); below T it stays
    an outlier. C_single=None gives D3's single tier. Returns (topic_id, topic_assignment, similarity, topic_id_2,
    similarity_2) aligned with `ids`; topic_id_2 is -1 and similarity_2 NaN except for dual posts."""
    topic_id = topics_raw.copy()
    topic_id_2 = np.full(len(ids), -1)
    assignment = np.where(topics_raw == -1, "outlier", "cluster").astype(object)
    sim = np.full(len(ids), np.nan)
    sim_2 = np.full(len(ids), np.nan)
    if len(nearest):
        pos = {str(i): k for k, i in enumerate(ids.astype(str))}
        has2 = "nearest_topic_2" in nearest.columns
        for r in nearest.itertuples(index=False):
            k = pos.get(str(r.id))
            if k is None:
                continue
            s = float(r.similarity)
            sim[k] = s
            if T is None or s < T:
                continue
            topic_id[k] = int(r.nearest_topic)
            t2 = r.nearest_topic_2 if has2 else None
            if C_single is not None and s < C_single and t2 is not None and not pd.isna(t2) and int(t2) != int(r.nearest_topic):
                assignment[k] = "dual"
                topic_id_2[k] = int(t2)
                sim_2[k] = float(r.similarity_2)
            else:
                assignment[k] = "reassigned"
    return topic_id, assignment, sim, topic_id_2, sim_2


# ---------------------------------------------------------------------------
# The per-bin tables (design §6.4) — counting rule K1 (author, 2026-10-30; docs/session-11_2026-10-30/): a topic's
# aggregates count the posts whose topic_assignment is in config.TOPIC_COUNT_ASSIGNMENTS (cluster + reassigned). A dual post
# keeps both labels on the post (the recorded hedge of D9) but is counted in neither topic; n_dual_first / n_dual_second
# record it. This replaced D9's counting (n_total = n_first + n_as_second, a dual post once in each of its two topics) after
# the coded judgement samples put the dual tier's precision at 43.6 % and the reassigned tier's at 53.4 %.
# ---------------------------------------------------------------------------
TOPIC_COUNT_COLUMNS = ["n_cluster", "n_reassigned", "n_dual_first", "n_dual_second", "n_total", "share"]
TOPIC_QA_COLUMNS = ["share_match_thread", "share_lang_en", "mean_comments"]
TOPIC_COLUMNS = ["topic_id"] + TOPIC_COUNT_COLUMNS + ["keywords", "representative_ids", "representative_titles"] + TOPIC_QA_COLUMNS


def counted(assignment) -> np.ndarray:
    """Boolean mask of the posts that count in a topic's aggregates (config.TOPIC_COUNT_ASSIGNMENTS)."""
    return np.isin(np.asarray(assignment).astype(str), list(C.TOPIC_COUNT_ASSIGNMENTS))


def derive_assignment(topic_id_raw, topic_id, topic_id_2) -> np.ndarray:
    """topic_assignment from the three id arrays (cluster / reassigned / dual / outlier), for callers that do not carry it."""
    raw, fin, sec = (np.asarray(x).astype(int) for x in (topic_id_raw, topic_id, topic_id_2))
    return np.where(raw != -1, "cluster", np.where(fin == -1, "outlier", np.where(sec != -1, "dual", "reassigned"))).astype(object)


def topic_counts(topic_id, topic_id_2, assignment, n_modelled: int) -> pd.DataFrame:
    """One row per topic, the −1 row included (K1): n_cluster (HDBSCAN members), n_reassigned (outliers singly reassigned to
    the topic), n_dual_first / n_dual_second (dual posts carrying the topic first / second — recorded, not counted),
    n_total = n_cluster + n_reassigned, share = n_total over the bin's modelled posts (the shares of a bin sum to at most 1).
    The −1 row keeps its earlier meaning: n_cluster = the raw outliers (HDBSCAN's noise set), n_total = the posts left as
    outliers. Model-free: works on the columns of documents.csv, after the overrides in the collect, so topics.csv and
    topic_labels_all_bins.csv are counted by one function."""
    tid = np.asarray(topic_id).astype(int)
    tid2 = np.asarray(topic_id_2).astype(int) if topic_id_2 is not None else np.full(len(tid), -1)
    a = np.asarray(assignment).astype(str)
    topics = set(tid.tolist()) | set(int(t) for t in tid2.tolist() if t != -1)
    if (a != "cluster").any():
        topics.add(-1)
    cl = pd.Series(tid[a == "cluster"]).value_counts(); re_ = pd.Series(tid[a == "reassigned"]).value_counts()
    d1 = pd.Series(tid[a == "dual"]).value_counts(); d2 = pd.Series(tid2[tid2 != -1]).value_counts()
    rows = []
    for t in sorted(topics):
        if t == -1:
            n_cl, n_re, n_d1, n_d2, n_tot = int((a != "cluster").sum()), 0, 0, 0, int((a == "outlier").sum())
        else:
            n_cl, n_re, n_d1, n_d2 = int(cl.get(t, 0)), int(re_.get(t, 0)), int(d1.get(t, 0)), int(d2.get(t, 0))
            n_tot = n_cl + n_re
        rows.append({"topic_id": int(t), "n_cluster": n_cl, "n_reassigned": n_re, "n_dual_first": n_d1, "n_dual_second": n_d2,
                     "n_total": n_tot, "share": round(n_tot / max(1, n_modelled), 5)})
    return pd.DataFrame(rows, columns=["topic_id"] + TOPIC_COUNT_COLUMNS)


def topic_qa(docs: pd.DataFrame, topic_id, assignment) -> pd.DataFrame:
    """share_match_thread, share_lang_en and mean_comments per topic over the counted posts (the −1 row over the posts left
    as outliers). `docs` carries post_type, lang and num_comments aligned with topic_id."""
    tid = np.asarray(topic_id).astype(int); a = np.asarray(assignment).astype(str)
    keep = counted(a) | (a == "outlier")
    d = pd.DataFrame({"topic_id": tid[keep],
                      "mt": (docs["post_type"].to_numpy()[keep] == "match_thread").astype(float),
                      "en": (docs["lang"].to_numpy()[keep] == "en").astype(float),
                      "nc": pd.to_numeric(pd.Series(docs["num_comments"].to_numpy()[keep]), errors="coerce").fillna(0).to_numpy()})
    g = d.groupby("topic_id").agg(share_match_thread=("mt", "mean"), share_lang_en=("en", "mean"), mean_comments=("nc", "mean"))
    g[["share_match_thread", "share_lang_en"]] = g[["share_match_thread", "share_lang_en"]].round(4)
    g["mean_comments"] = g["mean_comments"].round(2)
    return g.reset_index()


def topic_table(model, docs: pd.DataFrame, topic_id_raw: np.ndarray, topic_id: np.ndarray,
                rep_ids: dict[int, list[str]], topic_id_2: np.ndarray | None = None, assignment=None) -> pd.DataFrame:
    """topics.csv of a bin: the K1 sizes (topic_counts), keywords from cluster members, representative posts, and the QA
    covariates over the counted posts (topic_qa). `assignment` is derived from the three id arrays when not given."""
    tid2 = np.asarray(topic_id_2) if topic_id_2 is not None else np.full(len(docs), -1)
    if assignment is None:
        assignment = derive_assignment(topic_id_raw, topic_id, tid2)
    counts = topic_counts(topic_id, tid2, assignment, len(docs))
    title_of = dict(zip(docs["id"].astype(str), docs["title"]))
    deco = []
    for t in counts["topic_id"]:
        words = [w for w, _ in (model.get_topic(int(t)) or [])] if t != -1 else []
        reps = rep_ids.get(int(t), []) if t != -1 else []
        deco.append({"topic_id": int(t), "keywords": ", ".join(words), "representative_ids": "|".join(reps),
                     "representative_titles": " || ".join(str(title_of.get(i, ""))[:140] for i in reps)})
    out = counts.merge(pd.DataFrame(deco), on="topic_id", how="left").merge(topic_qa(docs, topic_id, assignment), on="topic_id", how="left")
    for c in TOPIC_QA_COLUMNS:
        out[c] = out[c].fillna(0.0)
    return out[TOPIC_COLUMNS]


def topics_over_time(model, docs: pd.DataFrame, topic_id: np.ndarray, period_type: str,
                     topic_id_2: np.ndarray | None = None, assignment=None) -> pd.DataFrame:
    """Weekly (tournament bins) or monthly (interim bins) topic frequencies over the counted posts (K1: cluster + reassigned;
    a dual post is passed as −1, so it is counted in neither topic). Needs the fitted model (c-TF-IDF words per period)."""
    freq = C.TIME_BIN[period_type]
    stamps = docs["date"].dt.tz_localize(None).dt.to_period("W" if freq == "W" else "M").dt.start_time
    tops = np.asarray(topic_id).astype(int).copy()
    if assignment is not None:
        tops[~counted(assignment)] = -1
    elif topic_id_2 is not None:
        tops[np.asarray(topic_id_2).astype(int) != -1] = -1          # a dual post is recognisable by its second topic
    tot = model.topics_over_time(docs["document"].tolist(), stamps.tolist(), topics=[int(t) for t in tops], global_tuning=False, evolution_tuning=False)
    tot = tot.rename(columns={"Topic": "topic_id", "Words": "keywords", "Frequency": "n_docs", "Timestamp": "period"})
    return tot[["topic_id", "period", "n_docs", "keywords"]]


def topics_over_time_counts(docs: pd.DataFrame, topic_id: np.ndarray, period_type: str, assignment=None) -> pd.DataFrame:
    """topics_over_time() WITHOUT the model (the reference run, 2026-10-30): the counted posts (K1) per topic and period, the
    uncounted ones (dual, outlier) in the −1 row, `keywords` empty — the same four columns, so report.py and every reader of the
    per-bin file need no change. BERTopic's own routine over 199 months × thousands of topics × 1.29 M posts would add hours
    of c-TF-IDF for words per period that the analysis does not use."""
    freq = C.TIME_BIN[period_type]
    stamps = docs["date"].dt.tz_localize(None).dt.to_period("W" if freq == "W" else "M").dt.start_time
    tops = np.asarray(topic_id).astype(int).copy()
    if assignment is not None:
        tops[~counted(assignment)] = -1
    tot = (pd.DataFrame({"topic_id": tops, "period": stamps.to_numpy()}).groupby(["topic_id", "period"]).size()
           .reset_index(name="n_docs").sort_values(["period", "topic_id"]).reset_index(drop=True))
    tot["keywords"] = ""
    return tot[["topic_id", "period", "n_docs", "keywords"]]


# ---------------------------------------------------------------------------
# Persistence (safetensors; UMAP/HDBSCAN are replayed from the .npy arrays, design §11)
# ---------------------------------------------------------------------------
def save_model(model, path, embedding_model_name: str):
    path.mkdir(parents=True, exist_ok=True)
    model.save(str(path), serialization="safetensors", save_ctfidf=True, save_embedding_model=embedding_model_name)


def load_model(path):
    from bertopic import BERTopic
    return BERTopic.load(str(path))
