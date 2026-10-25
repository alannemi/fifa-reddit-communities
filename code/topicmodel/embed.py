"""
Stage S1 — sentence embeddings with a per-year cache (design §5).
=================================================================
Embeddings are the most expensive step and are identical for the same (model, document) pair, so they are computed
once per calendar year — for ALL documents of the year, too-short ones included — and stored under
config.EMBED_CACHE_DIR/<model-slug>/<year>.npz (ids + float32 matrix).  Every bin draws its rows from the year caches
(bin 7 spans 2022 and 2023), a re-run costs nothing already embedded, and a Colab job that disconnects resumes from
the cache.  The model weights are downloaded into config.LLM_CACHE_DIR.
"""
from __future__ import annotations
import logging
import re
from pathlib import Path
import numpy as np
import pandas as pd
from . import config as C

log = logging.getLogger("topicmodel")


def get_device(requested: str = "auto") -> str:
    """Pick the compute device: CUDA (Colab/NVIDIA) > MPS (Apple) > CPU."""
    import torch
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def model_slug(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def load_embedding_model(name: str, device: str):
    from sentence_transformers import SentenceTransformer
    m = SentenceTransformer(name, device=device, cache_folder=str(C.LLM_CACHE_DIR))
    m.max_seq_length = C.EMBEDDING_MAX_SEQ_LENGTH
    return m


def _cache_path(model_name: str, year: int) -> Path:
    p = C.EMBED_CACHE_DIR / model_slug(model_name)
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{year}.npz"


def cached(model_name: str, year: int) -> bool:
    """Whether a cache file exists for (model, year) — the caller decides whether to load the model."""
    return _cache_path(model_name, year).exists()


def embed_year(year: int, docs: pd.DataFrame, model_name: str, device: str, st_model=None,
               force: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (ids, embeddings) for ALL documents of a year, computing and caching if needed.
    `docs` must contain the full year's documents (id, document, too_short); `force` discards an existing cache."""
    path = _cache_path(model_name, year)
    if path.exists() and force:
        path.unlink()
        log.info("embeddings: --force removed %s", path)
    elif path.exists():
        z = np.load(path, allow_pickle=False)
        if len(z["ids"]) == len(docs) and set(z["ids"]) == set(docs["id"]):
            log.info("embeddings: cache hit %s (%d docs)", path.name, len(docs))
            return z["ids"], z["emb"]
        log.warning("embeddings: cache %s does not match the documents; recomputing", path.name)
    st_model = st_model or load_embedding_model(model_name, device)
    texts = docs["document"].tolist()
    log.info("embeddings: encoding %d docs for %d with %s on %s", len(texts), year, model_name, device)
    emb = st_model.encode(texts, batch_size=C.EMBEDDING_BATCH_SIZE, show_progress_bar=True,
                          convert_to_numpy=True, normalize_embeddings=True).astype(np.float32)
    ids = docs["id"].to_numpy().astype(str)   # fixed-width unicode, loadable with allow_pickle=False
    np.savez(path, ids=ids, emb=emb)
    log.info("embeddings: cached %s", path)
    return ids, emb


def embeddings_for(docs: pd.DataFrame, full_year_docs: dict[int, pd.DataFrame], model_name: str, device: str) -> np.ndarray:
    """Embeddings aligned to `docs` (any subset of one or more years), via the per-year cache."""
    st_model = None
    lookup = {}
    for year, ydocs in full_year_docs.items():
        if not cached(model_name, year) and st_model is None:
            st_model = load_embedding_model(model_name, device)
        ids, emb = embed_year(year, ydocs, model_name, device, st_model)
        lookup.update({i: k for k, i in enumerate(ids)})
        lookup[("emb", year)] = emb
    out = np.empty((len(docs), next(v for k, v in lookup.items() if isinstance(k, tuple)).shape[1]), dtype=np.float32)
    year_of = docs["year"].astype(int).to_numpy()
    for i, (pid, y) in enumerate(zip(docs["id"].to_numpy(), year_of)):
        out[i] = lookup[("emb", y)][lookup[pid]]
    return out
