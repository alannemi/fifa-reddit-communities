"""
Stage S3 — two-agent cluster labelling (S3a) and discourse taxonomy (S3b) with Qwen3-1.7B.
==========================================================================================
Two "analysts" (the same model with two different lenses, design §7.2) must agree on the output of every item.
Agreement protocol = decision D2 (design §7.5):
  round 1   both analysts answer the same prompt; identical normalised answers -> agreed (`identical`)
  round r>1 alternating turns: an analyst sees its own latest and the peer's latest answer and returns either
            {"decision": "accept"} (the peer's already-parsed answer is adopted; `accepted_by_A|B`) or
            {"decision": "revise", "categories": [...]}; a revise identical to the analyst's own previous
            answer ends the debate (`stalemate`); identical to the peer's -> agreed (`identical`)
  cap       MAX_ROUNDS (5): unresolved items keep both answers, Analyst A's is provisional (`cap`)
  no answer a second parse/validation failure after one repair turn (design §7.1; refinement R7):
            in a revision turn the analyst keeps its previous answer (counts as an identical revise);
            in round 1 the peer's answer is provisional and the item is `no_answer`
`round_iterations` is the round in which agreement occurred (computed here, not by the agents; F1).
Every prompt, raw answer, decision, seed and timing is kept in the item's transcript (labels/topic_<id>.json,
taxonomy.json), so a run can be audited and replayed without the model.

S3a items = one BERTopic topic each (15 keywords + up to 40 sample posts).
S3b (decision D5, 2026-10-24; design §7.4 v1.2) is a two-step protocol: step 1 "define" — one debate per bin over all
topic lines (`topic_id | name | tone | keywords | description`) that agrees on the category definitions (no topic
ids; ids canonicalised cat_1..K before step 2); step 2 "assign" — one debate item per topic (config
TAXONOMY_ASSIGN_BATCH) that assigns it to one of the agreed categories — since decision D6 (2026-10-25, after the first
Colab run) the item lists only the topic's TAXONOMY_SHORTLIST most similar categories, most similar first (sentence
embeddings of the category and topic lines), the answer carries the category's id and name (the name wins a conflict),
and a topic whose agreed category ranks worse than TAXONOMY_REVIEW_RANK is flagged. Topics left unassigned are placed by the
deterministic repair (`category_assignment` records it); defined categories that end empty are dropped and listed.
The v1.1 single-prompt protocol (categories with explicit `topics` lists, chunking above 100 items) is kept as
TAXONOMY_PROTOCOL = "partition" for comparison and replay. Approvals: docs/session-06_2026-10-24/00_approvals.md.
Generation is batched across items per (round, analyst) in sorted key order; seeded sampling (the F17 comparison)
runs unbatched so that each call's SHA-256 seed is exact (refinement N4).
"""
from __future__ import annotations
import hashlib
import json
import logging
import math
import random
import re
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import prompts as P

log = logging.getLogger("topicmodel")

A, B = P.AGENTS
LABEL_KEYS = ["name", "purpose", "emotion_tone", "description", "inclusion_cues"]
AGREEMENT_MODES = ("identical", "accepted_by_A", "accepted_by_B", "stalemate", "cap", "no_answer")
ASSIGN_COLS = ["assignment_item", "assignment_round_iterations", "assignment_agreement_mode", "assignment_contested",
               "assignment_needs_review", "assignment_similarity_rank", "assignment_candidates"]   # the last two: D6
# bookkeeping keys kept in parsed answers but never shown to the model in a revision turn
INTERNAL_KEYS = {"id_given", "topics_ignored", "resolved_from", "n_cues_given", "cleanup"}


# ===========================================================================
# 0. Controlled tone vocabulary (design §7.3)
# ===========================================================================
def tone_category(free_text: str) -> str:
    """Map a free-text emotion_tone to the controlled vocabulary config.TONE_SET by word stems (config.TONE_STEMS,
    first match wins); "other" when no stem matches.  Used for `tone_category` in the label and category tables and
    for the identity rules of the taxonomy debates."""
    t = str(free_text).lower()
    for stem, cat in C.TONE_STEMS:
        if stem in t:
            return cat
    return "other"


# ===========================================================================
# 1. The language model
# ===========================================================================
def call_seed(*parts) -> int:
    """Per-call seed for seeded sampling: first 8 hex digits of SHA-256 of "bin-item-agent-round" (design §7.1)."""
    return int(hashlib.sha256("-".join(str(p) for p in parts).encode("utf-8")).hexdigest()[:8], 16)


class LLM:
    """Batched chat generation with a Hugging Face causal LM (Qwen3 by default; thinking mode disabled)."""

    def __init__(self, model_id: str = C.LLM_MODEL, device: str = "cpu", revision: str | None = None,
                 sampling: str = "greedy", batch_size: int | None = None):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.model_id, self.device, self.sampling = model_id, device, sampling
        # the pinned revision belongs to the model it was recorded for; other models use their latest snapshot
        self.revision = revision if revision is not None else (C.LLM_REVISION if model_id == C.LLM_MODEL else None)
        kw = dict(cache_dir=str(C.LLM_CACHE_DIR), revision=self.revision)
        self.tok = AutoTokenizer.from_pretrained(model_id, padding_side="left", **kw)
        if self.tok.pad_token_id is None:
            self.tok.pad_token = self.tok.eos_token
        if device == "cpu":
            dtype = torch.float32
        elif device == "cuda" and torch.cuda.get_device_capability()[0] < 8:   # T4 and older: no bfloat16
            dtype = torch.float16
        else:
            dtype = torch.bfloat16
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=dtype, **kw).to(device).eval()
        self.dtype = str(dtype).replace("torch.", "")
        eos = self.model.generation_config.eos_token_id
        eos = self.tok.eos_token_id if eos is None else eos
        self.eos_ids = set(eos if isinstance(eos, (list, tuple)) else [eos])
        self.batch_size = 1 if sampling == "seeded" else (batch_size or C.LLM_BATCH_SIZE.get(device, 1))
        self.is_qwen3 = "qwen3" in model_id.lower()
        self.stats = {"calls": 0, "prompts": 0, "seconds": 0.0, "tokens_out": 0, "truncated": 0}
        log.info("LLM loaded: %s (revision %s) on %s, %s, batch %d, %s decoding",
                 model_id, self.revision, device, self.dtype, self.batch_size, sampling)

    def describe(self) -> dict:
        return {"llm_model": self.model_id, "llm_revision": self.revision, "device": self.device, "dtype": self.dtype,
                "decoding": "greedy" if self.sampling == "greedy" else f"seeded_sampling_t{C.LLM_SAMPLING_TEMPERATURE}",
                "batch_size": self.batch_size}

    def _render(self, system: str, user: str) -> str:
        msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
        kw = {"enable_thinking": False} if self.is_qwen3 else {}
        return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kw)

    def generate(self, requests: list[tuple[str, str]], max_new_tokens: int, seeds: list[int] | None = None,
                 meta: list[dict] | None = None) -> list[dict]:
        """requests = [(system, user), ...] -> one dict per request: text, truncated, n_tokens, seed, seconds.
        `meta` (item key, analyst, round) is ignored here; a test stub uses it to script answers."""
        import torch
        out = []
        for start in range(0, len(requests), self.batch_size):
            chunk = requests[start:start + self.batch_size]
            enc = self.tok([self._render(s, u) for s, u in chunk], return_tensors="pt", padding=True).to(self.device)
            kw = dict(max_new_tokens=max_new_tokens, pad_token_id=self.tok.pad_token_id)
            seed = None
            if self.sampling == "seeded":
                seed = seeds[start]
                torch.manual_seed(seed)
                kw.update(do_sample=True, temperature=C.LLM_SAMPLING_TEMPERATURE, top_p=0.95, top_k=20)
            else:   # Qwen3's generation_config defaults to sampling: override every sampling knob explicitly (R11)
                kw.update(do_sample=False, temperature=None, top_p=None, top_k=None)
            t0 = time.time()
            with torch.no_grad():
                gen = self.model.generate(**enc, **kw)
            secs = time.time() - t0
            new = gen[:, enc["input_ids"].shape[1]:].cpu()
            for i in range(len(chunk)):
                ids = new[i].tolist()
                n = sum(1 for t in ids if t != self.tok.pad_token_id)
                truncated = not any(t in self.eos_ids for t in ids)
                text = self.tok.decode(ids, skip_special_tokens=True)
                text = re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()
                out.append({"text": text, "truncated": truncated, "n_tokens": n, "seed": seed,
                            "seconds": round(secs / len(chunk), 2)})
                self.stats["tokens_out"] += n
                self.stats["truncated"] += int(truncated)
            self.stats["calls"] += 1
            self.stats["prompts"] += len(chunk)
            self.stats["seconds"] += secs
            log.info("LLM: %d prompt(s) in %.1fs (%.1f s/prompt; cap %d tokens)", len(chunk), secs, secs / len(chunk), max_new_tokens)
        return out


# ===========================================================================
# 2. Parsing, validation, normalisation, comparison
# ===========================================================================
def extract_json(text: str) -> tuple[dict | None, str | None]:
    """First JSON object in the text (raw_decode from the first '{'), or an error message."""
    i = text.find("{")
    if i < 0:
        return None, "no JSON object found in the answer"
    try:
        obj, _ = json.JSONDecoder().raw_decode(text[i:])
    except json.JSONDecodeError as e:
        return None, f"invalid JSON ({e.msg} at character {e.pos})"
    if not isinstance(obj, dict):
        return None, "the JSON value is not an object"
    return obj, None


_CAT_START = re.compile(r'\{\s*"(?:id|name)"\s*:')
_STR_FIELD = r'"%s"\s*:\s*"((?:[^"\\]|\\.)*)"'
_ARR_FIELD = r'"%s"\s*:\s*(\[[^\]]*\])'


def salvage_json(text: str) -> dict | None:
    """Recover an answer whose category objects are well-formed but whose brackets between them are not
    (Qwen3-1.7B wrote `}], [{` between categories and `]]]` endings in the 2026-10-23 dry run). Each category is
    cut at the next category start and its fields are read individually; arrays must be flat. Returns None when
    no category with a name can be recovered. Used only after the strict parse failed (config.LLM_JSON_SALVAGE)."""
    starts = [m.start() for m in _CAT_START.finditer(text)]
    cats = []
    for i, s in enumerate(starts):
        seg = text[s: starts[i + 1] if i + 1 < len(starts) else len(text)]
        cat = {}
        for k in ("id", "name", "purpose", "emotion_tone", "description"):
            m = re.search(_STR_FIELD % k, seg)
            if m:
                try:
                    cat[k] = json.loads('"' + m.group(1) + '"')
                except json.JSONDecodeError:
                    cat[k] = m.group(1)
        for k in ("inclusion_cues", "topics"):
            m = re.search(_ARR_FIELD % k, seg)
            if m:
                try:
                    cat[k] = json.loads(m.group(1))
                except json.JSONDecodeError:
                    pass
        if cat.get("name"):
            cats.append(cat)
    if not cats:
        return None
    decision = "accept" if re.search(r'"decision"\s*:\s*"accept"', text, flags=re.I) else "revise"
    return {"decision": decision, "categories": cats}


def _default_salvage(text: str, truncated: bool) -> dict | None:
    """N11 as approved: category objects are recovered only from answers that were not cut off."""
    return None if truncated else salvage_json(text)


def salvage_definitions(text: str, truncated: bool) -> dict | None:
    """Step-1 answers carry no topic lists that could be wrong, so the complete category objects of a truncated
    answer are recovered too (review finding R21); the last object of a cut-off answer is dropped as incomplete."""
    obj = salvage_json(text)
    if obj is None:
        return None
    if truncated and len(obj["categories"]) > 1:
        obj["categories"] = obj["categories"][:-1]
    return obj


_PAIR_TC = re.compile(r'"topic_id"\s*:\s*"?(-?\d+)"?\s*,\s*"category_id"\s*:\s*"?([^",}\]\n]+)"?')
_PAIR_CT = re.compile(r'"category_id"\s*:\s*"?([^",}\]\n]+)"?\s*,\s*"topic_id"\s*:\s*"?(-?\d+)"?')
_PAIR_MAP = re.compile(r'"(-?\d+)"\s*:\s*"([^"]+)"')


def salvage_assignments(text: str, truncated: bool) -> dict | None:
    """Recover {topic_id, category_id} pairs from a bracket-damaged step-2 answer (object form, reversed field order,
    or the {"12": "cat_3"} mapping form). Never on a truncated answer (N11)."""
    if truncated:
        return None
    pairs = [(int(t), c.strip()) for t, c in _PAIR_TC.findall(text)]
    if not pairs:
        pairs = [(int(t), c.strip()) for c, t in _PAIR_CT.findall(text)]
    if not pairs:
        pairs = [(int(t), c.strip()) for t, c in _PAIR_MAP.findall(text)]
    if not pairs:
        return None
    decision = "accept" if re.search(r'"decision"\s*:\s*"accept"', text, flags=re.I) else "revise"
    return {"decision": decision, "assignments": [{"topic_id": t, "category_id": c} for t, c in pairs]}


def _public_json(obj) -> str:
    """An answer as the peer sees it in a revision turn: the pipeline's bookkeeping keys removed."""
    def strip(x):
        if isinstance(x, dict):
            return {k: strip(v) for k, v in x.items() if k not in INTERNAL_KEYS}
        if isinstance(x, list):
            return [strip(v) for v in x]
        return x
    return json.dumps(strip(obj), ensure_ascii=False)


def _cues(v) -> list[str]:
    if isinstance(v, str):
        v = re.split(r"[;\n]|, (?=[A-Z])", v)
    if not isinstance(v, list):
        return []
    return [str(x).strip() for x in v if str(x).strip()]


def _text_field(c: dict, k: str) -> str | None:
    v = c.get(k)
    if isinstance(v, list):
        v = ", ".join(str(x) for x in v)
    if v is None or not str(v).strip():
        return None
    return " ".join(str(v).split())


def validate_label(obj: dict, allow_accept: bool) -> tuple[dict | None, str | None]:
    """S3a answer: {"decision": "accept"} (revision turns only) or exactly one category with the five fields."""
    decision = str(obj.get("decision", "revise")).strip().lower()
    if decision == "accept":
        if allow_accept:
            return {"decision": "accept"}, None
        return None, "'accept' is only valid in a revision turn; return the full answer"
    cats = obj.get("categories")
    if isinstance(cats, dict):
        cats = [cats]
    if not isinstance(cats, list) or len(cats) != 1:
        return None, f"'categories' must be a list with exactly one category (got {0 if not isinstance(cats, list) else len(cats)})"
    c = cats[0]
    if not isinstance(c, dict):
        return None, "the category must be a JSON object"
    missing = [k for k in LABEL_KEYS if k not in c]
    if missing:
        return None, f"missing field(s): {', '.join(missing)}"
    out = {}
    for k in ("name", "purpose", "emotion_tone", "description"):
        v = _text_field(c, k)
        if v is None:
            return None, f"field '{k}' is empty"
        out[k] = v
    cues = _cues(c["inclusion_cues"])
    if not cues:
        return None, "'inclusion_cues' must list at least one short cue"
    # The prompt asks for 3-6 cues; the 1.7B model often copies 11-21 keywords. An over-long list is content, not a
    # defect: keep the first ten and record how many were given (N12, 2026-10-23 dry run).
    out["inclusion_cues"] = cues[:10]
    if len(cues) > 10:
        out["n_cues_given"] = len(cues)
    return {"decision": "revise", "categories": [out]}, None


# --- S3b step 1: definitions -----------------------------------------------------------------------------------
def validate_definitions(obj: dict, allow_accept: bool) -> tuple[dict | None, str | None]:
    """Step-1 answer: accept (revision turns only) or >= 1 category with the five fields, no topic lists.
    Hard failure (None): no category with a name. Soft problems (duplicated names, an empty text field) are
    returned together with the parsed answer so that the debate keeps it, the repair turn quotes them, and
    `cleanup_definitions` settles what the repair did not cure (review finding R18). Ids are canonicalised here
    (cat_1 .. cat_K in the order given; `id_given` keeps the model's) and never renumbered later (R16)."""
    decision = str(obj.get("decision", "revise")).strip().lower()
    if decision == "accept":
        if allow_accept:
            return {"decision": "accept"}, None
        return None, "'accept' is only valid in a revision turn; return the full list of categories"
    cats = obj.get("categories")
    if isinstance(cats, dict):
        cats = [cats]
    if not isinstance(cats, list) or not cats:
        return None, "'categories' must be a non-empty list"
    out, problems = [], []
    for i, c in enumerate(cats, 1):
        if not isinstance(c, dict):
            problems.append(f"category {i} is not a JSON object"); continue
        name = _text_field(c, "name")
        if name is None:
            problems.append(f"category {i} has no name"); continue
        d = {"id": f"cat_{len(out) + 1}", "name": name}
        given = str(c.get("id") or "").strip()
        if given and given != d["id"]:
            d["id_given"] = given
        for k in ("purpose", "emotion_tone", "description"):
            v = _text_field(c, k)
            if v is None:
                problems.append(f"category '{name}': field '{k}' is empty"); v = ""
            d[k] = v
        cues = _cues(c.get("inclusion_cues", []))
        d["inclusion_cues"] = cues[:10]
        if len(cues) > 10:
            d["n_cues_given"] = len(cues)
        if "topics" in c:
            d["topics_ignored"] = True
        out.append(d)
    if not out:
        return None, "no category with a name was found (" + "; ".join(problems) + ")"
    names = [_norm(d["name"]) for d in out]
    dups = sorted({n for n in names if names.count(n) > 1})
    if dups:
        problems.append(f"duplicated category name(s): {dups}; merge them or rename")
    parsed = {"decision": "revise", "categories": out}
    return parsed, ("; ".join(problems) if problems else None)


def cleanup_definitions(cats: list[dict]) -> tuple[list[dict], dict | None]:
    """Deterministic settlement of soft problems the repair turn did not cure: same-name categories merged into
    the first (cues united, empty fields filled from the duplicate), ids re-canonicalised (still before step 2).
    Returns (categories, log or None)."""
    kept, by_name, merged, filled = [], {}, {}, []
    for c in cats:
        n = _norm(c["name"])
        if n in by_name:
            k = by_name[n]
            for cue in c.get("inclusion_cues", []):
                if cue not in k["inclusion_cues"] and len(k["inclusion_cues"]) < 10:
                    k["inclusion_cues"].append(cue)
            for f in ("purpose", "emotion_tone", "description"):
                if not k.get(f) and c.get(f):
                    k[f] = c[f]; filled.append([k["id"], f])
            merged[c["id"]] = k["id"]
        else:
            k = dict(c); k["inclusion_cues"] = list(c.get("inclusion_cues", []))
            by_name[n] = k; kept.append(k)
    renum = {}
    for i, k in enumerate(kept, 1):
        new = f"cat_{i}"
        if k["id"] != new:
            renum[k["id"]] = new; k["id"] = new
    merged = {old: renum.get(new, new) for old, new in merged.items()}
    empty = [[k["id"], f] for k in kept for f in ("purpose", "emotion_tone", "description") if not k.get(f)]
    log_ = {"merged_same_name": merged, "renumbered": renum, "filled_from_duplicate": filled, "still_empty": empty}
    return kept, (log_ if any(log_.values()) else None)


def definitions_identical(a: dict | None, b: dict | None) -> bool:
    """Same set of normalised category names and, per name, the same tone category (the v1.1 S3b rule without
    the partition)."""
    if not a or not b or "categories" not in a or "categories" not in b:
        return False
    pa = {_norm(c["name"]): tone_category(c["emotion_tone"]) for c in a["categories"]}
    pb = {_norm(c["name"]): tone_category(c["emotion_tone"]) for c in b["categories"]}
    return pa == pb


# --- S3b step 2: assignment ------------------------------------------------------------------------------------
def build_category_index(cats: list[dict]) -> dict:
    """Lookup tables for lenient category references (F30): exact id, bare number, normalised name."""
    ids = [c["id"] for c in cats]
    by_num = {}
    for i in ids:
        m = re.fullmatch(r"cat_(\d+)", i)
        if m:
            by_num[int(m.group(1))] = i
    return {"ids": set(ids), "by_num": by_num, "by_name": {_norm(c["name"]): c["id"] for c in cats}}


_MISSING_REFS = {"", "none", "null", "n/a", "na", "-", "unknown", "other"}


def resolve_category_ref(ref, index: dict) -> tuple[str | None, str]:
    """-> (category_id or None, how): 'id' | 'number' | 'name' | 'id_in_text' | 'missing' | 'unknown'."""
    if ref is None or isinstance(ref, bool):
        return None, "missing"
    if isinstance(ref, int):
        return (index["by_num"][ref], "number") if ref in index["by_num"] else (None, "unknown")
    s = str(ref).strip()
    if s.lower() in _MISSING_REFS:
        return None, "missing"
    if s in index["ids"]:
        return s, "id"
    m = re.fullmatch(r"(?:cat|category)?[\s_\-#:]*(\d+)", s, flags=re.I)
    if m and int(m.group(1)) in index["by_num"]:
        return index["by_num"][int(m.group(1))], "number"
    n = _norm(s)
    if n in index["by_name"]:
        return index["by_name"][n], "name"
    m = re.search(r"cat_\d+", s)
    if m and m.group(0) in index["ids"]:
        return m.group(0), "id_in_text"
    for name_n, cid in index["by_name"].items():
        if name_n and (name_n in n or n in name_n) and min(len(n), len(name_n)) >= 8:
            return cid, "name"
    return None, "unknown"


def _pairs_of(obj: dict) -> list | None:
    """The (topic_id, category reference, category name or None) triples of a step-2 answer in any accepted shape
    (the name column exists since prompt v1.3, D6)."""
    a = obj.get("assignments")
    if a is None and isinstance(obj.get("categories"), list):          # the v1.1 habit: categories with topic lists
        pairs = []
        for c in obj["categories"]:
            if isinstance(c, dict):
                tl = c.get("topics", [])
                for t in (tl if isinstance(tl, list) else [tl]):
                    pairs.append((t, c.get("id") or c.get("category_id") or c.get("name"), c.get("name")))
        return pairs
    if isinstance(a, dict):
        return [(t, c, None) for t, c in a.items()]
    if isinstance(a, list):
        pairs = []
        for x in a:
            if isinstance(x, dict):
                pairs.append((x.get("topic_id", x.get("topic", x.get("id"))), x.get("category_id", x.get("category", x.get("cat"))),
                              x.get("category_name", x.get("name"))))
            elif isinstance(x, (list, tuple)) and len(x) in (2, 3):
                pairs.append((x[0], x[1], x[2] if len(x) == 3 else None))
        return pairs
    return None


def validate_assignments(obj: dict, topic_ids: list[int], index: dict, allow_accept: bool) -> tuple[dict | None, str | None]:
    """Step-2 answer: accept, or one category per topic of the item. Structural failure -> (None, err); a topic without
    a category or an unresolvable category reference -> (cleaned answer, err) so that the debate keeps it, one repair
    turn is tried and the deterministic repair can finish it (review finding R22). Extra pairs (foreign or repeated
    topic ids) and an id/name conflict are settled and recorded in `cleanup` without a repair turn (N13, 2026-10-25)."""
    decision = str(obj.get("decision", "revise")).strip().lower()
    if decision == "accept":
        if allow_accept:
            return {"decision": "accept"}, None
        return None, "'accept' is only valid in a revision turn; return the assignments"
    pairs = _pairs_of(obj)
    if pairs is None:
        return None, "'assignments' must be a list of {\"topic_id\": ..., \"category_id\": ...} objects"
    valid = set(topic_ids)
    mapping, how, foreign, dup, unresolved, bad, conflicts = {}, {}, [], [], [], [], []
    for t_raw, c_raw, n_raw in pairs:
        t = _coerce_topic_id(t_raw)
        if t is None:
            bad.append(t_raw); continue
        if t not in valid:
            foreign.append(t); continue
        if t in mapping:
            dup.append(t); continue
        cid, h = resolve_category_ref(c_raw, index)
        nid = resolve_category_ref(n_raw, index)[0] if n_raw is not None and str(n_raw).strip() else None
        if cid is None and nid is None:
            unresolved.append((t, c_raw if c_raw is not None else n_raw)); continue
        if cid is not None and nid is not None and cid != nid:
            # D6: id and name point to different categories — the name carries the meaning, the id is the token the
            # model confuses; the name wins, the conflict is reported (one repair turn) and recorded
            conflicts.append((t, c_raw, n_raw)); cid, h = nid, "name_over_id"
        elif cid is None:
            cid, h = nid, "name"
        mapping[t] = cid; how[t] = h
    unresolved = [(t, c) for t, c in unresolved if t not in mapping]   # a later pair may have named a category
    missing = [t for t in topic_ids if t not in mapping]
    if not mapping and not pairs:
        return None, "no assignments found"
    # N13 (2026-10-25, after the D6 dry run on bin 1): extra pairs — a topic id that is not in the item, a repeated id, a
    # non-integer id — and an id/name conflict are settled here (dropped, first kept, name wins) and recorded, WITHOUT a
    # repair turn: 16 of 65 items had spent a repair turn on such extras and 5 answers changed category in it (topic 18
    # from right to wrong). The repair turn stays for what cannot be settled: a topic without a category, an
    # unresolvable reference.
    cleanup = {}
    if bad: cleanup["bad_topic_ids"] = [str(x) for x in bad[:10]]
    if foreign: cleanup["foreign"] = sorted(set(foreign))
    if dup: cleanup["duplicated"] = sorted(set(dup))
    if conflicts: cleanup["conflicts"] = [[t, str(c), str(n)] for t, c, n in conflicts]
    msgs = []
    if unresolved: msgs.append("category_id not among the listed ids for topic_id(s) " +
                               ", ".join(f"{t} ({c!r})" for t, c in unresolved[:10]) + " — use the category ids exactly as listed")
    if missing: msgs.append(f"missing topic_id(s): {missing[:20]}")
    parsed = {"decision": "revise", "assignments": [{"topic_id": t, "category_id": mapping[t]} for t in topic_ids if t in mapping],
              "resolved_from": {str(t): h for t, h in how.items() if h != "id"}}
    if cleanup:
        parsed["cleanup"] = cleanup
    if msgs:
        return parsed, "every topic_id must appear exactly once with one of the listed category ids — " + "; ".join(msgs)
    return parsed, None


def _mapping(ans: dict | None) -> dict[int, str]:
    if not ans or "assignments" not in ans:
        return {}
    return {int(p["topic_id"]): p["category_id"] for p in ans["assignments"]}


def assignments_identical(a: dict | None, b: dict | None) -> bool:
    """Identical cleaned mapping topic -> category (two empty mappings are not an agreement)."""
    ma, mb = _mapping(a), _mapping(b)
    return bool(ma) and ma == mb


def _coerce_topic_id(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    m = re.fullmatch(r"\s*(?:topic[_ ]?)?(-?\d+)\s*", str(x))
    return int(m.group(1)) if m else None


def partition_problems(categories: list[dict], topic_ids: list[int]) -> dict:
    """unknown / duplicated / missing topic ids of a taxonomy answer (empty dict = valid partition)."""
    valid = set(topic_ids)
    seen, dup, unknown = set(), [], []
    for c in categories:
        for t in c["topics"]:
            if t not in valid:
                unknown.append(t)
            elif t in seen:
                dup.append(t)
            else:
                seen.add(t)
    missing = [t for t in topic_ids if t not in seen]
    out = {}
    if unknown: out["unknown"] = sorted(set(unknown))
    if dup: out["duplicated"] = sorted(set(dup))
    if missing: out["missing"] = missing
    return out


def validate_taxonomy(obj: dict, topic_ids: list[int], allow_accept: bool) -> tuple[dict | None, str | None]:
    """S3b answer of the partition protocol (v1.1, superseded by the two-step protocol): accept, or ≥ 1 categories
    each with the five fields plus `topics`; the `topics` lists must
    partition `topic_ids` (every id exactly once). Structural errors and partition violations both return an
    error message (the caller grants one repair turn); the parsed object is returned alongside a partition
    error so that the deterministic repair can start from it."""
    decision = str(obj.get("decision", "revise")).strip().lower()
    if decision == "accept":
        if allow_accept:
            return {"decision": "accept"}, None
        return None, "'accept' is only valid in a revision turn; return the full taxonomy"
    cats = obj.get("categories")
    if isinstance(cats, dict):
        cats = [cats]
    if not isinstance(cats, list) or not cats:
        return None, "'categories' must be a non-empty list"
    out = []
    for i, c in enumerate(cats, 1):
        if not isinstance(c, dict):
            return None, f"category {i} is not a JSON object"
        missing = [k for k in LABEL_KEYS + ["topics"] if k not in c]
        if missing:
            return None, f"category {i} misses field(s): {', '.join(missing)}"
        d = {"id": str(c.get("id") or f"cat_{i}").strip()}
        for k in ("name", "purpose", "emotion_tone", "description"):
            v = _text_field(c, k)
            if v is None:
                return None, f"category {i}: field '{k}' is empty"
            d[k] = v
        d["inclusion_cues"] = _cues(c["inclusion_cues"])
        tl = c["topics"] if isinstance(c["topics"], list) else [c["topics"]]
        ids = [_coerce_topic_id(t) for t in tl]
        if any(t is None for t in ids):
            return None, f"category {i}: 'topics' must be a list of topic_id integers"
        d["topics"] = ids
        out.append(d)
    parsed = {"decision": "revise", "categories": out}
    prob = partition_problems(out, topic_ids)
    if prob:
        msg = "; ".join(f"{k} topic_id(s): {v[:20]}{'…' if len(v) > 20 else ''}" for k, v in prob.items())
        return parsed, f"the 'topics' lists must contain every topic_id exactly once — {msg}"
    return parsed, None


def _norm(s) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split())


def norm_label(cat: dict) -> dict:
    return {"name": _norm(cat["name"]), "purpose": _norm(cat["purpose"]), "emotion_tone": _norm(cat["emotion_tone"]),
            "description": _norm(cat["description"]), "inclusion_cues": sorted(_norm(c) for c in cat["inclusion_cues"])}


def labels_identical(a: dict | None, b: dict | None) -> bool:
    if not a or not b or "categories" not in a or "categories" not in b:
        return False
    return norm_label(a["categories"][0]) == norm_label(b["categories"][0])


def partition(tax: dict) -> dict[frozenset, dict]:
    return {frozenset(c["topics"]): c for c in tax["categories"]}


def taxonomies_identical(a: dict | None, b: dict | None) -> bool:
    """Same partition of topic ids and, for each matched category, the same normalised name and tone category."""
    if not a or not b or "categories" not in a or "categories" not in b:
        return False
    pa, pb = partition(a), partition(b)
    if set(pa) != set(pb):
        return False
    return all(_norm(pa[k]["name"]) == _norm(pb[k]["name"]) and
               tone_category(pa[k]["emotion_tone"]) == tone_category(pb[k]["emotion_tone"]) for k in pa)


# ===========================================================================
# 3. The debate engine (one implementation for both stages)
# ===========================================================================
class Debate:
    """Runs the D2 protocol over a list of items; items are independent and generated in batches per
    (round, analyst), in sorted key order.  `validate(obj, item, allow_accept)` and `identical(a, b)` are
    stage-specific; `on_resolved(record)` is called as soon as an item is settled (resume granularity)."""

    def __init__(self, stage: str, llm, validate, identical, max_rounds: int = C.MAX_ROUNDS,
                 seed_prefix: str = "", on_resolved=None, max_new_tokens: int | None = None, salvage=None,
                 answer_key: str | None = None):
        self.stage, self.llm, self.validate, self.identical = stage, llm, validate, identical
        self.max_rounds, self.seed_prefix, self.on_resolved = max_rounds, seed_prefix, on_resolved
        # token budget, salvage function and the answer key named in the revision turn are per stage (v1.2)
        budgets = {"label": C.LLM_MAX_NEW_TOKENS_LABEL, "define": C.LLM_MAX_NEW_TOKENS_DEFINE,
                   "assign": C.LLM_MAX_NEW_TOKENS_ASSIGN, "taxonomy": C.LLM_MAX_NEW_TOKENS_PARTITION}
        self.max_new = int(max_new_tokens) if max_new_tokens else budgets[stage]
        self.salvage = salvage or _default_salvage
        self.answer_key = answer_key or P.ANSWER_KEYS.get(stage, "categories")

    # --- one batched turn for a set of items -------------------------------------------------------------
    def _turn(self, items: list[dict], analyst: str, rnd: int, user_of, allow_accept: bool) -> None:
        system = P.system_prompt(self.stage, analyst)
        reqs = [(system, user_of(it)) for it in items]
        seeds = [call_seed(self.seed_prefix, it["key"], analyst, rnd) for it in items]
        meta = [{"key": it["key"], "analyst": analyst, "round": rnd, "repair": False} for it in items]
        res = self.llm.generate(reqs, self.max_new, seeds=seeds, meta=meta)
        failed = []
        for it, (sys_, usr), r, seed in zip(items, reqs, res, seeds):
            parsed, err = self._parse(r, it, allow_accept)
            it["transcript"].append({"round": rnd, "analyst": analyst, "repair": False, "system": sys_, "user": usr,
                                     "answer": r["text"], "parsed": parsed, "error": err, "truncated": r["truncated"],
                                     "parsed_via": r.get("parsed_via"), "salvage_reason": r.get("salvage_reason"),
                                     "n_tokens": r["n_tokens"], "seed": seed, "seconds": r["seconds"]})
            it["_turn"] = (parsed, err)
            if parsed is None or err:
                failed.append((it, err))
        if failed:   # one repair turn quoting the problem (design §7.1)
            reqs = [(system, P.render_repair(it["items_block"], err)) for it, err in failed]
            seeds = [call_seed(self.seed_prefix, it["key"], analyst, rnd, "repair") for it, _ in failed]
            meta = [{"key": it["key"], "analyst": analyst, "round": rnd, "repair": True} for it, _ in failed]
            res = self.llm.generate(reqs, self.max_new, seeds=seeds, meta=meta)
            for (it, _), (sys_, usr), r, seed in zip(failed, reqs, res, seeds):
                parsed, err = self._parse(r, it, allow_accept)
                it["transcript"].append({"round": rnd, "analyst": analyst, "repair": True, "system": sys_, "user": usr,
                                         "answer": r["text"], "parsed": parsed, "error": err, "truncated": r["truncated"],
                                         "parsed_via": r.get("parsed_via"), "salvage_reason": r.get("salvage_reason"),
                                         "n_tokens": r["n_tokens"], "seed": seed, "seconds": r["seconds"]})
                prev_parsed, _ = it["_turn"]
                if parsed is not None and not err:
                    it["_turn"] = (parsed, None)
                elif parsed is not None and prev_parsed is None:
                    it["_turn"] = (parsed, err)          # structurally fine, partition still invalid -> repaired later
                # else: keep the previous (parsed-with-error or None) -> "no answer"

    def _parse(self, r: dict, it: dict, allow_accept: bool):
        obj, err = extract_json(r["text"])
        r["parsed_via"] = "json"
        if obj is None and C.LLM_JSON_SALVAGE:
            obj = self.salvage(r["text"], r["truncated"])       # stage-specific recovery (N11; R15)
            if obj is not None:
                r["parsed_via"] = "salvage"
                r["salvage_reason"] = ("truncated; " if r["truncated"] else "") + str(err)
        if obj is None:
            if r["truncated"]:
                err = f"the answer was cut off at {self.max_new} tokens ({err}); answer more concisely"
            return None, err
        return self.validate(obj, it, allow_accept)

    # --- protocol ------------------------------------------------------------------------------------------
    def run(self, items: list[dict]) -> dict[str, dict]:
        for it in items:
            it.update(answers={A: None, B: None}, resolved=False, transcript=[], _turn=None)
        items = sorted(items, key=lambda it: (isinstance(it["key"], str), it["key"]))
        results = {}

        def resolve(it, mode, rnd, final, agreed):
            assert mode in AGREEMENT_MODES, mode
            it["resolved"] = True
            rec = {"key": it["key"], "agreed": bool(agreed), "agreement_mode": mode, "round_iterations": rnd,
                   "needs_review": not agreed, "final": final, "analyst_a_final": it["answers"][A],
                   "analyst_b_final": it["answers"][B], "transcript": it["transcript"]}
            results[it["key"]] = rec
            if self.on_resolved:
                self.on_resolved(rec)

        # round 1: both analysts, same prompt
        for analyst in (A, B):
            self._turn(items, analyst, 1, lambda it: it["items_block"], allow_accept=False)
            for it in items:
                parsed, err = it["_turn"]
                it["answers"][analyst] = parsed      # may be None (no answer) or parsed-with-partition-error (S3b)
        for it in items:
            a, b = it["answers"][A], it["answers"][B]
            if a is None and b is None:
                resolve(it, "no_answer", 1, None, False)
            elif a is None or b is None:
                resolve(it, "no_answer", 1, a or b, False)
            elif self.identical(a, b):
                resolve(it, "identical", 1, a, True)
        log.info("%s: round 1 done — %d/%d items resolved", self.stage, sum(it["resolved"] for it in items), len(items))

        # rounds 2..max: alternating turns
        for rnd in range(2, self.max_rounds + 1):
            for analyst, peer in ((A, B), (B, A)):
                pending = [it for it in items if not it["resolved"]]
                if not pending:
                    break
                def user_of(it, analyst=analyst, peer=peer):
                    return P.render_revision(it["items_block"], _public_json(it["answers"][analyst]),
                                             _public_json(it["answers"][peer]), self.answer_key)
                self._turn(pending, analyst, rnd, user_of, allow_accept=True)
                for it in pending:
                    parsed, err = it["_turn"]
                    mode_suffix = "A" if analyst == A else "B"
                    if parsed is None:                       # no answer: keeps its previous answer -> stalemate rule
                        resolve(it, "stalemate", rnd, it["answers"][A], False)
                    elif parsed.get("decision") == "accept":
                        resolve(it, f"accepted_by_{mode_suffix}", rnd, it["answers"][peer], True)
                    elif self.identical(parsed, it["answers"][analyst]):
                        resolve(it, "stalemate", rnd, it["answers"][A], False)
                    else:
                        it["answers"][analyst] = parsed
                        if self.identical(parsed, it["answers"][peer]):
                            resolve(it, "identical", rnd, parsed, True)
            log.info("%s: round %d done — %d/%d items resolved", self.stage, rnd, sum(it["resolved"] for it in items), len(items))
            if all(it["resolved"] for it in items):
                break
        for it in items:
            if not it["resolved"]:
                resolve(it, "cap", self.max_rounds, it["answers"][A], False)
        return results


# ===========================================================================
# 4. Stage S3a — cluster labelling
# ===========================================================================
def _one_line(text: str, n: int = C.LLM_POST_MAX_CHARS) -> str:
    s = " / ".join(part.strip() for part in str(text).split("\n") if part.strip())
    s = " ".join(s.split())
    return s[:n]


def sample_posts(member_docs: pd.DataFrame, representative_ids: list[str], seed_key: str,
                 n: int = C.LLM_SAMPLE_POSTS) -> tuple[list[str], list[str]]:
    """The posts one topic's analysts see: the representative posts first, then seeded-random cluster members."""
    ids = member_docs["id"].tolist()
    text = dict(zip(member_docs["id"], member_docs["document"]))
    reps = [i for i in representative_ids if i in text][: C.N_REPRESENTATIVE_DOCS]
    rest = [i for i in ids if i not in set(reps)]
    random.Random(f"{C.SEED}-{seed_key}").shuffle(rest)
    chosen = reps + rest[: max(0, n - len(reps))]
    return [_one_line(text[i]) for i in chosen], chosen


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_labelling(bin_dir: Path, bin_id: int, docs: pd.DataFrame, topics: pd.DataFrame, llm,
                  max_topics: int | None = None) -> pd.DataFrame:
    """Label every topic of a bin (resumable through labels.jsonl). `docs` = modelled documents with columns
    id, document, topic_id_raw; `topics` = topics.csv. Writes labels/topic_<id>.json, labels.jsonl,
    topics_labelled.csv and the topic_overrides.csv template; returns the labelled table."""
    labels_dir = bin_dir / "labels"; labels_dir.mkdir(parents=True, exist_ok=True)
    jsonl = bin_dir / "labels.jsonl"
    done = {int(r["topic_id"]): r for r in read_jsonl(jsonl)}
    t = topics[topics["topic_id"].astype(int) != -1].copy()
    t["topic_id"] = t["topic_id"].astype(int)
    if max_topics:
        t = t.nlargest(max_topics, "n_cluster")
    todo = t[~t["topic_id"].isin(done)].sort_values("topic_id")
    log.info("labelling bin %d: %d topics, %d already done, %d to do", bin_id, len(t), len(t) - len(todo), len(todo))

    items, meta = [], {}
    by_topic = docs.groupby("topic_id_raw")
    for _, row in todo.iterrows():
        tid = int(row["topic_id"])
        members = by_topic.get_group(tid) if tid in by_topic.groups else docs.iloc[0:0]
        reps = [x for x in str(row.get("representative_ids", "")).split("|") if x]
        posts, post_ids = sample_posts(members, reps, f"{bin_id}-{tid}")
        items.append({"key": tid, "items_block": P.render_label_items(row["keywords"], posts)})
        meta[tid] = {"keywords": row["keywords"], "n_sample_posts": len(posts), "sample_ids": post_ids}

    def on_resolved(rec):
        tid = int(rec["key"])
        final = (rec["final"] or {}).get("categories", [None])[0] if rec["final"] else None
        out = {"period_bin": bin_id, "topic_id": tid, **meta[tid], **llm.describe(), "prompt_version": P.PROMPT_VERSION,
               "agreed": rec["agreed"], "agreement_mode": rec["agreement_mode"], "round_iterations": rec["round_iterations"],
               "needs_review": rec["needs_review"], "label": final, "analyst_a_final": rec["analyst_a_final"],
               "analyst_b_final": rec["analyst_b_final"]}
        (labels_dir / f"topic_{tid}.json").write_text(json.dumps({**out, "transcript": rec["transcript"]}, ensure_ascii=False, indent=1), encoding="utf-8")
        with open(jsonl, "a", encoding="utf-8") as f:
            f.write(json.dumps(out, ensure_ascii=False) + "\n")
        done[tid] = out
        log.info("topic %d: %s [%s, %d round(s)]%s", tid, (final or {}).get("name", "—"), rec["agreement_mode"],
                 rec["round_iterations"], " NEEDS REVIEW" if rec["needs_review"] else "")

    if items:
        Debate("label", llm, lambda obj, it, acc: validate_label(obj, acc), labels_identical,
               seed_prefix=str(bin_id), on_resolved=on_resolved).run(items)

    rows = []
    for tid in sorted(done):
        r = done[tid]; lab = r.get("label") or {}
        rows.append({"topic_id": tid, "keywords": r.get("keywords", ""), "name": lab.get("name", ""),
                     "purpose": lab.get("purpose", ""), "emotion_tone": lab.get("emotion_tone", ""),
                     "tone_category": tone_category(lab.get("emotion_tone", "")) if lab else "",
                     "description": lab.get("description", ""), "inclusion_cues": "; ".join(lab.get("inclusion_cues", [])),
                     "round_iterations": r["round_iterations"], "agreed": r["agreed"], "agreement_mode": r["agreement_mode"],
                     "needs_review": r["needs_review"],
                     "analyst_a_final": json.dumps(r["analyst_a_final"], ensure_ascii=False) if r["analyst_a_final"] else "",
                     "analyst_b_final": json.dumps(r["analyst_b_final"], ensure_ascii=False) if r["analyst_b_final"] else "",
                     "n_sample_posts": r.get("n_sample_posts", ""), "llm_model": r.get("llm_model", ""),
                     "llm_revision": r.get("llm_revision", ""), "prompt_version": r.get("prompt_version", ""),
                     "decoding": r.get("decoding", "")})
    labelled = pd.DataFrame(rows)
    labelled.to_csv(bin_dir / "topics_labelled.csv", index=False)
    ov = bin_dir / "topic_overrides.csv"
    if not ov.exists() and len(labelled):
        pd.DataFrame({"topic_id": labelled["topic_id"], "name": labelled["name"], "needs_review": labelled["needs_review"],
                      "final_name": "", "merge_into": "", "notes": ""}).to_csv(ov, index=False)
    log.info("labelling bin %d: %d topics labelled, %d need review -> %s", bin_id, len(labelled),
             int(labelled["needs_review"].sum()) if len(labelled) else 0, bin_dir / "topics_labelled.csv")
    return labelled


# ===========================================================================
# 5. Overrides (co-author decision files)
# ===========================================================================
def read_overrides(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["final_name", "merge_into", "notes"])
    return pd.read_csv(path, dtype=str, keep_default_na=False)


def _resolve_chain(m: dict) -> dict:
    out = {}
    for k in m:
        t, seen = k, set()
        while t in m and t not in seen:
            seen.add(t); t = m[t]
        out[k] = t
    return out


def apply_topic_overrides(labelled: pd.DataFrame, overrides: pd.DataFrame) -> tuple[pd.DataFrame, dict[int, int]]:
    """final_name replaces the agreed name (name_agent keeps it); merge_into collapses topics (chains resolved).
    Returns the table without the merged-away topics and the {old_topic_id: new_topic_id} map."""
    df = labelled.copy()
    df["topic_id"] = df["topic_id"].astype(int)
    df["name_agent"] = df["name"]
    names, merges = {}, {}
    for _, r in overrides.iterrows():
        tid = _coerce_topic_id(r.get("topic_id", ""))
        if tid is None:
            continue
        if str(r.get("final_name", "")).strip():
            names[tid] = str(r["final_name"]).strip()
        tgt = _coerce_topic_id(r.get("merge_into", "")) if str(r.get("merge_into", "")).strip() else None
        if tgt is not None and tgt != tid:
            merges[tid] = tgt
    merges = _resolve_chain(merges)
    df["name"] = [names.get(t, n) for t, n in zip(df["topic_id"], df["name"])]
    df = df[~df["topic_id"].isin(merges)].reset_index(drop=True)
    return df, merges


def read_final_categories(path: Path) -> dict[int, str]:
    """topic_category_overrides.csv (F28): {topic_id: final_category_id} for the rows a co-author filled in."""
    out = {}
    if not path.exists():
        return out
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    if "final_category_id" not in df.columns:
        return out
    for _, r in df.iterrows():
        tid = _coerce_topic_id(r.get("topic_id", ""))
        cid = str(r.get("final_category_id", "")).strip()
        if tid is not None and cid:
            out[tid] = cid
    return out


def apply_category_overrides(categories: pd.DataFrame, overrides: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, str]]:
    df = categories.copy()
    df["name_agent"] = df["name"]
    names, merges = {}, {}
    for _, r in overrides.iterrows():
        cid = str(r.get("category_id", "")).strip()
        if not cid:
            continue
        if str(r.get("final_name", "")).strip():
            names[cid] = str(r["final_name"]).strip()
        tgt = str(r.get("merge_into", "")).strip()
        if tgt and tgt != cid:
            merges[cid] = tgt
    merges = _resolve_chain(merges)
    df["name"] = [names.get(c, n) for c, n in zip(df["category_id"], df["name"])]
    df = df[~df["category_id"].isin(merges)].reset_index(drop=True)
    return df, merges


# ===========================================================================
# 6. Stage S3b — discourse taxonomy
# ===========================================================================
def repair_partition(categories: list[dict], topic_ids: list[int], emb_of) -> tuple[list[dict], dict, dict]:
    """Deterministic repair (design §7.4): unknown ids dropped, duplicates kept in the first category listed,
    missing topics placed in the category whose member-mean embedding is most similar (cosine); empty categories
    dropped. Returns (categories, {topic_id: agent|repaired|unplaced}, log)."""
    valid, seen, out, rlog = set(topic_ids), set(), [], {"unknown": [], "duplicated": [], "placed": {}}
    for c in categories:
        tl = []
        for t in c["topics"]:
            if t not in valid:
                rlog["unknown"].append(t)
            elif t in seen:
                rlog["duplicated"].append(t)
            else:
                seen.add(t); tl.append(t)
        out.append({**c, "topics": tl})
    assignment = {t: "agent" for t in seen}
    missing = [t for t in topic_ids if t not in seen]
    non_empty = [c for c in out if c["topics"]]
    if missing and non_empty and emb_of is not None:
        means = []
        for c in non_empty:
            m = np.mean([emb_of(t) for t in c["topics"]], axis=0)
            means.append(m / (np.linalg.norm(m) or 1.0))
        means = np.vstack(means)
        for t in missing:
            e = emb_of(t); e = e / (np.linalg.norm(e) or 1.0)
            best = int(np.argmax(means @ e))
            non_empty[best]["topics"].append(t)
            assignment[t] = "repaired"
            rlog["placed"][t] = non_empty[best]["id"]
    else:
        for t in missing:
            assignment[t] = "unplaced"
    out = [c for c in out if c["topics"]]
    ids = [c["id"] for c in out]
    if len(set(ids)) != len(ids) or any(not re.fullmatch(r"cat_\d+", i) for i in ids):
        for i, c in enumerate(out, 1):
            c["id"] = f"cat_{i}"
    return out, assignment, rlog


def _topic_lines(tbl: pd.DataFrame) -> list[str]:
    lines = []
    for _, r in tbl.iterrows():
        kws = [k.strip() for k in str(r["keywords"]).split(",") if k.strip()][: C.TAXONOMY_KEYWORDS_PER_TOPIC]
        name = r["name"] or ", ".join(kws[:3])
        lines.append(P.topic_line(int(r["topic_id"]), name, r["emotion_tone"] or "unknown", kws, r["description"]))
    return lines


def _chunk_topics(tbl: pd.DataFrame, emb_of) -> list[list[int]]:
    from sklearn.cluster import KMeans
    ids = tbl["topic_id"].astype(int).tolist()
    k = math.ceil(len(ids) / C.TAXONOMY_CHUNK_SIZE)
    X = np.vstack([emb_of(t) for t in ids])
    lab = KMeans(n_clusters=k, random_state=C.SEED, n_init=10).fit_predict(X)
    return [[t for t, l in zip(ids, lab) if l == c] for c in range(k) if (lab == c).any()]


def _load_taxonomy_inputs(bin_dir: Path):
    """topics_labelled.csv after topic overrides, sizes after merges, the sorted item table and the topic ids."""
    labelled = pd.read_csv(bin_dir / "topics_labelled.csv", dtype=str, keep_default_na=False)
    topics = pd.read_csv(bin_dir / "topics.csv", dtype=str, keep_default_na=False)
    labelled, merges = apply_topic_overrides(labelled, read_overrides(bin_dir / "topic_overrides.csv"))
    sizes = topics.set_index(topics["topic_id"].astype(int))["n_total"].astype(int).to_dict()
    for old, new in merges.items():
        sizes[new] = sizes.get(new, 0) + sizes.pop(old, 0)
    labelled["n_total"] = labelled["topic_id"].map(sizes).fillna(0).astype(int)
    tbl = labelled.sort_values("topic_id").reset_index(drop=True)
    topic_ids = tbl["topic_id"].astype(int).tolist()
    return tbl, topic_ids, sizes, merges


def run_taxonomy(bin_dir: Path, bin_id: int, emb_of, llm, start: str, end: str, protocol: str | None = None,
                 assign_batch: int | None = None, embed_texts=None) -> pd.DataFrame:
    """Group the bin's labelled topics (after topic overrides) into a discourse taxonomy. `emb_of(topic_id)`
    returns the topic embedding (768-d); `embed_texts(list[str]) -> array` embeds texts with the sentence-transformer
    (the step-2 shortlist of D6; None = every item lists the whole taxonomy). Dispatches on the protocol
    (config.TAXONOMY_PROTOCOL): "two_step" (D5, 2026-10-24) or "partition" (v1.1). Writes taxonomy.json,
    categories.csv, topic_categories.csv and the review templates; returns categories.csv as a DataFrame."""
    protocol = protocol or C.TAXONOMY_PROTOCOL
    if protocol == "two_step":
        return run_taxonomy_two_step(bin_dir, bin_id, emb_of, llm, start, end, assign_batch or C.TAXONOMY_ASSIGN_BATCH, embed_texts)
    if protocol == "partition":
        return run_taxonomy_partition(bin_dir, bin_id, emb_of, llm, start, end)
    raise ValueError(f"unknown taxonomy protocol {protocol!r} (expected 'two_step' or 'partition')")


def _unit_rows(x: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(n == 0, 1.0, n)


def _candidate_order(cats: list[dict], tbl: pd.DataFrame, embed_texts) -> tuple[dict[int, list[str]] | None, int]:
    """D6: for every topic, the category ids ordered by the cosine similarity of the sentence embeddings of the
    category line and the topic line (the texts shown to the analysts, without the ids). (None, 0) when no text
    embedder is given or the shortlist is switched off (config.TAXONOMY_SHORTLIST = 0)."""
    k = int(C.TAXONOMY_SHORTLIST)
    if embed_texts is None or k <= 0 or not cats:
        return None, 0
    cat_texts = [P.category_line("", c["name"], c["emotion_tone"], c["inclusion_cues"], c["description"]).split(" | ", 1)[1] for c in cats]
    topic_texts = [line.split(" | ", 1)[1] for line in _topic_lines(tbl)]
    ce = _unit_rows(np.asarray(embed_texts(cat_texts), dtype=float))
    te = _unit_rows(np.asarray(embed_texts(topic_texts), dtype=float))
    sims = te @ ce.T
    ids = tbl["topic_id"].astype(int).tolist()
    order = {t: [cats[j]["id"] for j in np.argsort(-sims[i], kind="stable")] for i, t in enumerate(ids)}
    return order, min(k, len(cats))


def _union_candidates(order: dict[int, list[str]], topic_ids: list[int], k: int) -> list[str]:
    """The candidates of an item with several topics: the per-topic shortlists interleaved by rank, without repeats."""
    out = []
    for r in range(k):
        for t in topic_ids:
            c = order[t][r]
            if c not in out:
                out.append(c)
    return out


def run_taxonomy_two_step(bin_dir: Path, bin_id: int, emb_of, llm, start: str, end: str, assign_batch: int = 1,
                          embed_texts=None) -> pd.DataFrame:
    """Decision D5: step 1 defines the categories (one debate per bin, checkpointed in taxonomy_definitions.json);
    step 2 assigns the topics (one debate item per `assign_batch` topics) — since D6 (2026-10-25) each item lists
    only the topics' TAXONOMY_SHORTLIST most similar categories, most similar first, and a topic whose agreed
    category ranks worse than TAXONOMY_REVIEW_RANK is flagged for review; the deterministic repair finishes."""
    tbl, topic_ids, sizes, merges = _load_taxonomy_inputs(bin_dir)
    assign_batch = max(1, int(assign_batch))
    log.info("taxonomy bin %d (two-step): %d topics (%d merged away by overrides), %d topic(s) per assignment item",
             bin_id, len(topic_ids), len(merges), assign_batch)
    records, clean_log, rec_def, cats_def = {}, None, None, []

    # --- step 1: definitions (resumed from the checkpoint when the topic set is unchanged) ---------------------
    ckpt = bin_dir / "taxonomy_definitions.json"
    if ckpt.exists():
        saved = json.loads(ckpt.read_text(encoding="utf-8"))
        if saved.get("topic_ids") == topic_ids:
            rec_def, cats_def, clean_log = saved["debate"], saved["categories"], saved.get("cleanup")
            log.info("taxonomy bin %d: step 1 resumed from %s (%d categories)", bin_id, ckpt.name, len(cats_def))
        else:
            log.warning("taxonomy bin %d: %s refers to a different topic set; re-running step 1", bin_id, ckpt.name)
            ckpt.unlink()
    if rec_def is None:
        item = {"key": "definitions", "items_block": P.render_define_items(len(topic_ids), start, end, _topic_lines(tbl))}
        rec_def = Debate("define", llm, lambda o, it, acc: validate_definitions(o, acc), definitions_identical,
                         seed_prefix=f"{bin_id}-def", max_new_tokens=C.LLM_MAX_NEW_TOKENS_DEFINE,
                         salvage=salvage_definitions).run([item])["definitions"]
        final = rec_def["final"]
        if final and final.get("categories"):
            cats_def, clean_log = cleanup_definitions(final["categories"])
        ckpt.write_text(json.dumps({"period_bin": bin_id, "topic_ids": topic_ids, "categories": cats_def, "cleanup": clean_log,
                                    "debate": rec_def, "written": pd.Timestamp.utcnow().isoformat(timespec="seconds")},
                                   ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        log.info("taxonomy bin %d: step 1 done — %d categories defined [%s, %d round(s)]%s%s", bin_id, len(cats_def),
                 rec_def["agreement_mode"], rec_def["round_iterations"], " NEEDS REVIEW" if rec_def["needs_review"] else "",
                 f"; cleanup {clean_log}" if clean_log else "")
    records["definitions"] = rec_def

    # --- step 2: assignment -----------------------------------------------------------------------------------
    items_info, per_topic = [], {}
    if not cats_def:
        cats, assignment = [], {t: "unplaced" for t in topic_ids}
        rlog = {"unknown": [], "duplicated": [], "placed": {}, "note": "no agreed definitions; step 2 skipped"}
        log.warning("taxonomy bin %d: no agreed category definitions — all %d topics unplaced", bin_id, len(topic_ids))
    else:
        index = build_category_index(cats_def)
        line_of = {c["id"]: P.category_line(c["id"], c["name"], c["emotion_tone"], c["inclusion_cues"], c["description"]) for c in cats_def}
        order, k = _candidate_order(cats_def, tbl, embed_texts)          # D6 shortlist (None = whole list)
        chunks = [topic_ids[i:i + assign_batch] for i in range(0, len(topic_ids), assign_batch)]
        items = []
        for i, ch in enumerate(chunks, 1):
            lines = _topic_lines(tbl[tbl["topic_id"].astype(int).isin(ch)])
            cand = _union_candidates(order, ch, k) if order else [c["id"] for c in cats_def]
            items.append({"key": f"assign_{i:03d}", "topic_ids": ch, "candidates": cand,
                          "items_block": P.render_assign_items(start, end, [line_of[c] for c in cand], len(ch), len(topic_ids), lines,
                                                               n_categories=len(cats_def))})
        log.info("taxonomy bin %d: step 2 — %d item(s), %s", bin_id, len(items),
                 f"shortlist of {k} of {len(cats_def)} categories per topic (D6)" if order else f"all {len(cats_def)} categories listed")
        recs = Debate("assign", llm, lambda o, it, acc: validate_assignments(o, it["topic_ids"], index, acc),
                      assignments_identical, seed_prefix=f"{bin_id}-assign", max_new_tokens=C.LLM_MAX_NEW_TOKENS_ASSIGN,
                      salvage=salvage_assignments, answer_key="assignments").run(items)
        records.update(recs)
        mapping = {}
        for it in items:
            rec = recs[it["key"]]
            fin, a_m, b_m = _mapping(rec["final"]), _mapping(rec["analyst_a_final"]), _mapping(rec["analyst_b_final"])
            mapping.update({t: c for t, c in fin.items() if t in it["topic_ids"]})
            n_contested = 0
            for t in it["topic_ids"]:
                # contested unless both analysts' final answers name the same category for the topic
                contested = not (a_m.get(t) is not None and a_m.get(t) == b_m.get(t))
                n_contested += int(contested)
                # D6: where does the agreed category rank among all categories by similarity to the topic? (1 = nearest)
                rank = (order[t].index(mapping[t]) + 1) if (order and t in mapping and mapping[t] in order[t]) else None
                per_topic[t] = {"assignment_item": it["key"], "assignment_round_iterations": rec["round_iterations"],
                                "assignment_agreement_mode": rec["agreement_mode"], "assignment_contested": contested,
                                "assignment_needs_review": ((not rec["agreed"]) and contested) or
                                                           (rank is not None and rank > int(C.TAXONOMY_REVIEW_RANK)),
                                "assignment_similarity_rank": rank if rank is not None else "",
                                "assignment_candidates": "|".join(it["candidates"])}
            items_info.append({"item": it["key"], "topic_ids": it["topic_ids"], "agreement_mode": rec["agreement_mode"],
                               "round_iterations": rec["round_iterations"], "agreed": rec["agreed"],
                               "n_contested": n_contested, "n_assigned": sum(t in mapping for t in it["topic_ids"]),
                               "candidates": it["candidates"]})
        cats_with = [{**c, "topics": [t for t in topic_ids if mapping.get(t) == c["id"]]} for c in cats_def]
        cats, assignment, rlog = repair_partition(cats_with, topic_ids, emb_of)
        rlog["assigned_by_agents"] = len(mapping)
        modes = dict(Counter(r["agreement_mode"] for r in items_info))
        log.info("taxonomy bin %d: step 2 done — %d/%d topics assigned by the agents over %d item(s) (modes %s), %d placed by repair",
                 bin_id, len(mapping), len(topic_ids), len(items), modes, sum(v == "repaired" for v in assignment.values()))
    dropped = [c["id"] for c in cats_def if c["id"] not in {x["id"] for x in cats}]
    ranks = [v["assignment_similarity_rank"] for v in per_topic.values() if v.get("assignment_similarity_rank") != ""]
    shortlist = {"k": int(C.TAXONOMY_SHORTLIST) if (cats_def and embed_texts is not None) else 0, "review_rank": int(C.TAXONOMY_REVIEW_RANK),
                 "text_embeddings": bool(cats_def) and embed_texts is not None,
                 "rank_distribution": {str(r): int(n) for r, n in sorted(Counter(ranks).items())},
                 "n_flagged_by_rank": int(sum(1 for r in ranks if r > int(C.TAXONOMY_REVIEW_RANK)))}
    extra = {"protocol": "two_step", "assign_batch": assign_batch, "categories_defined": cats_def,
             "definitions_cleanup": clean_log, "categories_dropped_empty": dropped, "assignment_items": items_info,
             "shortlist": shortlist, "chunking": None}
    return _write_taxonomy_outputs(bin_dir, bin_id, start, end, llm, tbl, sizes, merges, topic_ids, cats, assignment,
                                   rlog, rec_def, records, extra, per_topic)


def run_taxonomy_partition(bin_dir: Path, bin_id: int, emb_of, llm, start: str, end: str) -> pd.DataFrame:
    """The v1.1 single-prompt protocol (design §7.4 v1.0/v1.1): categories with explicit `topics` lists, compared
    as partitions; chunking fallback above TAXONOMY_MAX_ITEMS. Kept for comparison and replay (F26)."""
    tbl, topic_ids, sizes, merges = _load_taxonomy_inputs(bin_dir)
    log.info("taxonomy bin %d (partition): %d topics (%d merged away by overrides)", bin_id, len(topic_ids), len(merges))

    def validate(obj, it, acc):
        return validate_taxonomy(obj, it["topic_ids"], acc)

    records, chunk_info = {}, None
    if len(topic_ids) <= C.TAXONOMY_MAX_ITEMS:
        item = {"key": "bin", "topic_ids": topic_ids,
                "items_block": P.render_taxonomy_items(len(topic_ids), start, end, _topic_lines(tbl))}
        rec = Debate("taxonomy", llm, validate, taxonomies_identical, seed_prefix=f"{bin_id}-tax").run([item])["bin"]
        final = rec["final"] or {"categories": []}
        cats, assignment, rlog = repair_partition(final["categories"], topic_ids, emb_of)
        records["bin"] = rec
    else:   # chunking fallback (design §7.4)
        chunks = _chunk_topics(tbl, emb_of)
        chunk_info = {"n_chunks": len(chunks), "sizes": [len(c) for c in chunks]}
        log.info("taxonomy bin %d: %d topics > %d -> %d chunks", bin_id, len(topic_ids), C.TAXONOMY_MAX_ITEMS, len(chunks))
        items = [{"key": f"chunk_{i}", "topic_ids": ch,
                  "items_block": P.render_taxonomy_items(len(ch), start, end, _topic_lines(tbl[tbl["topic_id"].astype(int).isin(ch)]))}
                 for i, ch in enumerate(chunks, 1)]
        recs = Debate("taxonomy", llm, validate, taxonomies_identical, seed_prefix=f"{bin_id}-tax").run(items)
        records.update(recs)
        pseudo, pseudo_lines, pseudo_emb = {}, [], {}
        kw_of = {int(r["topic_id"]): [k.strip() for k in str(r["keywords"]).split(",") if k.strip()] for _, r in tbl.iterrows()}
        for it in items:
            rec = recs[it["key"]]
            cats_c, _, _ = repair_partition((rec["final"] or {"categories": []})["categories"], it["topic_ids"], emb_of)
            for c in cats_c:
                pid = len(pseudo) + 1
                pseudo[pid] = c
                kws = [k for t in c["topics"] for k in kw_of.get(t, [])[:3]][: C.TAXONOMY_KEYWORDS_PER_TOPIC]
                pseudo_lines.append(P.topic_line(pid, c["name"], c["emotion_tone"], kws, c["description"]))
                m = np.mean([emb_of(t) for t in c["topics"]], axis=0); pseudo_emb[pid] = m
        item = {"key": "bin", "topic_ids": list(pseudo),
                "items_block": P.render_taxonomy_items(len(pseudo), start, end, pseudo_lines)}
        rec = Debate("taxonomy", llm, validate, taxonomies_identical, seed_prefix=f"{bin_id}-tax2").run([item])["bin"]
        records["bin"] = rec
        cats_p, _, rlog = repair_partition((rec["final"] or {"categories": []})["categories"], list(pseudo), pseudo_emb.get)
        cats = [{**c, "topics": [t for pid in c["topics"] for t in pseudo[pid]["topics"]]} for c in cats_p]
        cats, assignment, rlog2 = repair_partition(cats, topic_ids, emb_of)
        rlog = {"second_pass": rlog, "expanded": rlog2}
    extra = {"protocol": "partition", "assign_batch": None, "categories_defined": None, "definitions_cleanup": None,
             "categories_dropped_empty": [], "assignment_items": [], "chunking": chunk_info}
    return _write_taxonomy_outputs(bin_dir, bin_id, start, end, llm, tbl, sizes, merges, topic_ids, cats, assignment,
                                   rlog, records["bin"], records, extra, {})


def _write_taxonomy_outputs(bin_dir: Path, bin_id: int, start: str, end: str, llm, tbl: pd.DataFrame, sizes: dict,
                            merges: dict, topic_ids: list[int], cats: list[dict], assignment: dict, rlog: dict,
                            rec: dict, records: dict, extra: dict, per_topic: dict) -> pd.DataFrame:
    """categories.csv, topic_categories.csv, taxonomy.json and the review templates — shared by both protocols.
    `rec` is the debate whose outcome describes the taxonomy as a whole (two-step: the definitions debate)."""
    n_posts_total = max(1, int(tbl["n_total"].sum()))
    rows, tmap = [], []
    for c in cats:
        n_posts = int(sum(sizes.get(t, 0) for t in c["topics"]))
        rows.append({"period_bin": bin_id, "category_id": c["id"], "name": c["name"], "purpose": c["purpose"],
                     "emotion_tone": c["emotion_tone"], "tone_category": tone_category(c["emotion_tone"]),
                     "description": c["description"], "inclusion_cues": "; ".join(c["inclusion_cues"]),
                     "topic_ids": "|".join(str(t) for t in c["topics"]), "n_topics": len(c["topics"]), "n_posts": n_posts,
                     "share": round(n_posts / n_posts_total, 5), "round_iterations": rec["round_iterations"],
                     "agreed": rec["agreed"], "agreement_mode": rec["agreement_mode"], "needs_review": rec["needs_review"],
                     "n_repaired": sum(assignment.get(t) == "repaired" for t in c["topics"])})
        tmap += [{"topic_id": t, "category_id": c["id"], "category_assignment": assignment.get(t, "agent"),
                  **{k: per_topic.get(t, {}).get(k, "") for k in ASSIGN_COLS}} for t in c["topics"]]
    for t in topic_ids:
        if assignment.get(t) == "unplaced":
            tmap.append({"topic_id": t, "category_id": "", "category_assignment": "unplaced",
                         **{k: per_topic.get(t, {}).get(k, "") for k in ASSIGN_COLS}})
    cat_cols = ["period_bin", "category_id", "name", "purpose", "emotion_tone", "tone_category", "description", "inclusion_cues",
                "topic_ids", "n_topics", "n_posts", "share", "round_iterations", "agreed", "agreement_mode", "needs_review", "n_repaired"]
    categories = pd.DataFrame(rows, columns=cat_cols)
    categories.to_csv(bin_dir / "categories.csv", index=False)
    tmap_df = pd.DataFrame(tmap, columns=["topic_id", "category_id", "category_assignment"] + ASSIGN_COLS).sort_values("topic_id")
    tmap_df.to_csv(bin_dir / "topic_categories.csv", index=False)
    n_salvaged = sum(1 for d in records.values() for t in d["transcript"] if t.get("parsed_via") == "salvage")
    n_cleaned = sum(1 for d in records.values() for t in d["transcript"] if isinstance(t.get("parsed"), dict) and t["parsed"].get("cleanup"))
    (bin_dir / "taxonomy.json").write_text(json.dumps({
        "period_bin": bin_id, "start": start, "end": end, "n_topics": len(topic_ids), "topic_overrides_merged": merges,
        **llm.describe(), "prompt_version": P.PROMPT_VERSION, **extra,
        "agreed": rec["agreed"], "agreement_mode": rec["agreement_mode"], "round_iterations": rec["round_iterations"],
        "needs_review": rec["needs_review"], "categories": cats,
        "category_assignment": {str(k): v for k, v in assignment.items()}, "repair_log": rlog,
        "n_answers_salvaged": n_salvaged, "n_answers_cleaned": n_cleaned,
        "debates": {k: {kk: vv for kk, vv in v.items()} for k, v in records.items()}}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    _write_review_templates(bin_dir, categories, tmap_df, tbl)
    log.info("taxonomy bin %d: %d categories [%s, %d round(s)], %d topics repaired, %d unplaced, %d answer(s) salvaged -> %s",
             bin_id, len(categories), rec["agreement_mode"], rec["round_iterations"],
             sum(v == "repaired" for v in assignment.values()), sum(v == "unplaced" for v in assignment.values()),
             n_salvaged, bin_dir / "categories.csv")
    return categories


def _template_is_filled(path: Path, columns: tuple[str, ...]) -> bool:
    if not path.exists():
        return False
    old = pd.read_csv(path, dtype=str, keep_default_na=False)
    return any(str(x).strip() for col in columns if col in old.columns for x in old[col])


def _write_review_templates(bin_dir: Path, categories: pd.DataFrame, tmap: pd.DataFrame, tbl: pd.DataFrame) -> None:
    """category_overrides.csv (category -> final_name / merge_into) and topic_category_overrides.csv (topic ->
    final_category_id, F28). A template is (re)written unless a co-author has filled it in, in which case it is kept
    and a warning says its ids may be stale after this re-run (review finding R34)."""
    if not len(categories):
        return
    ov = bin_dir / "category_overrides.csv"
    if _template_is_filled(ov, ("final_name", "merge_into")):
        log.warning("%s has entries and is kept; its category ids may be stale after this re-run", ov.name)
    else:
        pd.DataFrame({"category_id": categories["category_id"], "name": categories["name"], "n_topics": categories["n_topics"],
                      "needs_review": categories["needs_review"], "final_name": "", "merge_into": "", "notes": ""}).to_csv(ov, index=False)
    tov = bin_dir / "topic_category_overrides.csv"
    if _template_is_filled(tov, ("final_category_id",)):
        log.warning("%s has entries and is kept; its category ids may be stale after this re-run", tov.name)
    else:
        name_of = dict(zip(categories["category_id"], categories["name"]))
        tname = dict(zip(tbl["topic_id"].astype(int), tbl["name"]))
        t = tmap.sort_values("topic_id")
        pd.DataFrame({"topic_id": t["topic_id"], "name": [tname.get(int(x), "") for x in t["topic_id"]],
                      "category_id_agent": t["category_id"], "category_name": [name_of.get(c, "") for c in t["category_id"]],
                      "category_assignment": t["category_assignment"], "assignment_needs_review": t["assignment_needs_review"],
                      "similarity_rank": t["assignment_similarity_rank"], "final_category_id": "", "notes": ""}).to_csv(tov, index=False)
