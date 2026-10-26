"""
Stage S3 — two-agent cluster labelling with Qwen3-1.7B.
=======================================================
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
Every prompt, raw answer, decision, seed and timing is kept in the item's transcript (labels/topic_<id>.json), so a
run can be audited and replayed without the model.

S3a items = one BERTopic topic each (15 keywords + up to 40 sample posts). The agreed label — name, purpose,
emotion_tone (free text, mapped to the controlled tone vocabulary), description, inclusion cues — is the topic's label
in the chapter's analysis (decision D7, 2026-10-25: the unit of analysis is the individual BERTopic topic; the earlier
second stage that grouped topics into categories per bin, decisions D5/D6, was removed — design §7.4 history).
Generation is batched across items per (round, analyst) in sorted key order; seeded sampling (the F17 comparison)
runs unbatched so that each call's SHA-256 seed is exact (refinement N4).
"""
from __future__ import annotations
import hashlib
import json
import logging
import random
import re
import time
from pathlib import Path

import pandas as pd

from . import config as C
from . import prompts as P

log = logging.getLogger("topicmodel")

A, B = P.AGENTS
LABEL_KEYS = ["name", "purpose", "emotion_tone", "description", "inclusion_cues"]
AGREEMENT_MODES = ("identical", "accepted_by_A", "accepted_by_B", "stalemate", "cap", "no_answer")
# bookkeeping keys kept in parsed answers but never shown to the model in a revision turn
INTERNAL_KEYS = {"n_cues_given"}


# ===========================================================================
# 0. Controlled tone vocabulary (design §7.3)
# ===========================================================================
def tone_category(free_text: str) -> str:
    """Map a free-text emotion_tone to the controlled vocabulary config.TONE_SET by word stems (config.TONE_STEMS,
    first match wins); "other" when no stem matches.  Used for `tone_category` in the label table and, as `topic_tone`,
    on every post of the topic (D7)."""
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


def _coerce_topic_id(x):
    if isinstance(x, bool):
        return None
    if isinstance(x, int):
        return x
    m = re.fullmatch(r"\s*(?:topic[_ ]?)?(-?\d+)\s*", str(x))
    return int(m.group(1)) if m else None


def _norm(s) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]+", " ", str(s).lower()).split())


def norm_label(cat: dict) -> dict:
    return {"name": _norm(cat["name"]), "purpose": _norm(cat["purpose"]), "emotion_tone": _norm(cat["emotion_tone"]),
            "description": _norm(cat["description"]), "inclusion_cues": sorted(_norm(c) for c in cat["inclusion_cues"])}


def labels_identical(a: dict | None, b: dict | None) -> bool:
    if not a or not b or "categories" not in a or "categories" not in b:
        return False
    return norm_label(a["categories"][0]) == norm_label(b["categories"][0])


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
        budgets = {"label": C.LLM_MAX_NEW_TOKENS_LABEL}
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
                it["answers"][analyst] = parsed      # may be None (no answer) or parsed-with-error
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
                  max_topics: int | None = None, redo: list[int] | None = None, max_new_tokens: int | None = None) -> pd.DataFrame:
    """Label every topic of a bin (resumable through labels.jsonl). `docs` = modelled documents with columns
    id, document, topic_id_raw; `topics` = topics.csv. Writes labels/topic_<id>.json, labels.jsonl,
    topics_labelled.csv and the topic_overrides.csv template; returns the labelled table.
    `redo` (author's decision of 2026-10-26, F45): topics to label again — their previous transcripts move to
    labels/superseded/ (nothing deleted) and their lines leave labels.jsonl before the debate; `max_new_tokens` replaces
    the answer budget C.LLM_MAX_NEW_TOKENS_LABEL for this run (recorded per topic) — the case of the two topics whose
    analysts were both cut off at 600 tokens."""
    labels_dir = bin_dir / "labels"; labels_dir.mkdir(parents=True, exist_ok=True)
    jsonl = bin_dir / "labels.jsonl"
    done = {int(r["topic_id"]): r for r in read_jsonl(jsonl)}
    budget = int(max_new_tokens) if max_new_tokens else int(C.LLM_MAX_NEW_TOKENS_LABEL)
    redo = [int(x) for x in (redo or [])]
    if redo:
        stamp = pd.Timestamp.utcnow().strftime("%Y-%m-%dT%H-%M-%S")
        sup = labels_dir / "superseded"; sup.mkdir(exist_ok=True)
        for tid in redo:
            old_json = labels_dir / f"topic_{tid}.json"
            if old_json.exists():
                old_json.rename(sup / f"topic_{tid}_{stamp}.json")
            done.pop(tid, None)
        with open(jsonl, "w", encoding="utf-8") as f:
            for tid in sorted(done):
                f.write(json.dumps(done[tid], ensure_ascii=False) + "\n")
        log.info("labelling bin %d: relabelling topic(s) %s with an answer budget of %d tokens; previous transcripts in %s",
                 bin_id, redo, budget, sup)
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
               "max_new_tokens": budget, "relabelled": tid in redo,
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
               seed_prefix=str(bin_id), on_resolved=on_resolved, max_new_tokens=budget).run(items)

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
