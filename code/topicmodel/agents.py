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

S3a items = one BERTopic topic each (15 keywords + up to 40 sample posts); S3b items = all topics of a bin as
compact lines (`topic_id | name | tone | keywords | description`), with a chunking fallback above 100 topics
(design §7.4) and a deterministic repair of invalid partitions (`category_assignment` records it).
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
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import prompts as P
from .label import tone_category

log = logging.getLogger("topicmodel")

A, B = P.AGENTS
LABEL_KEYS = ["name", "purpose", "emotion_tone", "description", "inclusion_cues"]
AGREEMENT_MODES = ("identical", "accepted_by_A", "accepted_by_B", "stalemate", "cap", "no_answer")


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
    if not 1 <= len(cues) <= 10:
        return None, f"'inclusion_cues' must list 1-10 short cues (got {len(cues)})"
    out["inclusion_cues"] = cues
    return {"decision": "revise", "categories": [out]}, None


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
    """S3b answer: accept, or ≥ 1 categories each with the five fields plus `topics`; the `topics` lists must
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
                 seed_prefix: str = "", on_resolved=None):
        self.stage, self.llm, self.validate, self.identical = stage, llm, validate, identical
        self.max_rounds, self.seed_prefix, self.on_resolved = max_rounds, seed_prefix, on_resolved
        self.max_new = C.LLM_MAX_NEW_TOKENS_LABEL if stage == "label" else C.LLM_MAX_NEW_TOKENS_TAXONOMY

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
        if obj is None and C.LLM_JSON_SALVAGE and not r["truncated"]:
            obj = salvage_json(r["text"])
            if obj is not None:
                r["parsed_via"] = "salvage"
                r["salvage_reason"] = err
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
                    return P.render_revision(it["items_block"], json.dumps(it["answers"][analyst], ensure_ascii=False),
                                             json.dumps(it["answers"][peer], ensure_ascii=False))
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


def run_taxonomy(bin_dir: Path, bin_id: int, emb_of, llm, start: str, end: str) -> pd.DataFrame:
    """Group the bin's labelled topics (after topic overrides) into a discourse taxonomy. `emb_of(topic_id)`
    returns the topic embedding (768-d). Writes taxonomy.json, categories.csv, topic_categories.csv and the
    category_overrides.csv template; returns categories.csv as a DataFrame."""
    labelled = pd.read_csv(bin_dir / "topics_labelled.csv", dtype=str, keep_default_na=False)
    topics = pd.read_csv(bin_dir / "topics.csv", dtype=str, keep_default_na=False)
    labelled, merges = apply_topic_overrides(labelled, read_overrides(bin_dir / "topic_overrides.csv"))
    sizes = topics.set_index(topics["topic_id"].astype(int))["n_total"].astype(int).to_dict()
    for old, new in merges.items():
        sizes[new] = sizes.get(new, 0) + sizes.pop(old, 0)
    labelled["n_total"] = labelled["topic_id"].map(sizes).fillna(0).astype(int)
    tbl = labelled.sort_values("topic_id").reset_index(drop=True)
    topic_ids = tbl["topic_id"].astype(int).tolist()
    log.info("taxonomy bin %d: %d topics (%d merged away by overrides)", bin_id, len(topic_ids), len(merges))

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

    rec = records["bin"]
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
        tmap += [{"topic_id": t, "category_id": c["id"], "category_assignment": assignment.get(t, "agent")} for t in c["topics"]]
    for t in topic_ids:
        if assignment.get(t) == "unplaced":
            tmap.append({"topic_id": t, "category_id": "", "category_assignment": "unplaced"})
    cat_cols = ["period_bin", "category_id", "name", "purpose", "emotion_tone", "tone_category", "description", "inclusion_cues",
                "topic_ids", "n_topics", "n_posts", "share", "round_iterations", "agreed", "agreement_mode", "needs_review", "n_repaired"]
    categories = pd.DataFrame(rows, columns=cat_cols)
    categories.to_csv(bin_dir / "categories.csv", index=False)
    pd.DataFrame(tmap, columns=["topic_id", "category_id", "category_assignment"]).sort_values("topic_id").to_csv(bin_dir / "topic_categories.csv", index=False)
    n_salvaged = sum(1 for d in records.values() for t in d["transcript"] if t.get("parsed_via") == "salvage")
    (bin_dir / "taxonomy.json").write_text(json.dumps({
        "period_bin": bin_id, "start": start, "end": end, "n_topics": len(topic_ids), "topic_overrides_merged": merges,
        **llm.describe(), "prompt_version": P.PROMPT_VERSION, "agreed": rec["agreed"], "agreement_mode": rec["agreement_mode"],
        "round_iterations": rec["round_iterations"], "needs_review": rec["needs_review"], "categories": cats,
        "category_assignment": {str(k): v for k, v in assignment.items()}, "repair_log": rlog, "chunking": chunk_info,
        "n_answers_salvaged": n_salvaged,
        "debates": {k: {kk: vv for kk, vv in v.items()} for k, v in records.items()}}, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    ov = bin_dir / "category_overrides.csv"
    if not ov.exists() and len(categories):
        pd.DataFrame({"category_id": categories["category_id"], "name": categories["name"], "n_topics": categories["n_topics"],
                      "needs_review": categories["needs_review"], "final_name": "", "merge_into": "", "notes": ""}).to_csv(ov, index=False)
    log.info("taxonomy bin %d: %d categories [%s, %d round(s)], %d topics repaired, %d unplaced -> %s", bin_id, len(categories),
             rec["agreement_mode"], rec["round_iterations"], sum(v == "repaired" for v in assignment.values()),
             sum(v == "unplaced" for v in assignment.values()), bin_dir / "categories.csv")
    return categories
