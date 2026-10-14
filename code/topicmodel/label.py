"""
LLM topic labelling — two-analyst debate (adapted from the template paper).
===========================================================================
For every topic, two instances of the same open LLM ("Analyst A", "Analyst B")
receive the topic's keywords and up to LLM_SAMPLE_POSTS sample posts and must
return JSON with: name, purpose, emotion_tone, description, inclusion_cues.
If the two answers disagree, each analyst is asked to revise "to improve
agreement with the peer while preserving quality", for up to LLM_MAX_TURNS
turns.  Both final answers are kept; a topic whose analysts still disagree is
flagged `needs_review`.  Free-text tones are also mapped to a controlled
vocabulary (config.TONE_SET) so tones can be counted.

Model: config.LLM_MODEL (Qwen3-4B by default; any Hugging Face chat model id
works).  Runs locally (MPS/CUDA/CPU); no data leaves the machine.
Qwen3 "thinking" is disabled for speed and determinism.
"""
from __future__ import annotations
import json
import logging
import re
import pandas as pd
from . import config as C

log = logging.getLogger("topicmodel")

SCHEMA_KEYS = ["name", "purpose", "emotion_tone", "description", "inclusion_cues"]

SYSTEM = ("You are {agent}, a discourse analyst studying posts from the football community r/soccer. "
          "You label clusters of posts for a research codebook. Return JSON only; no prose, no markdown.")

INITIAL = ("Here is one cluster of r/soccer posts (one post per line, title first) and its most distinctive keywords.\n"
           "Keywords: {keywords}\n\nPosts:\n{posts}\n\n"
           "Describe the cluster. Output schema:\n"
           '{{"name": "short topic name (2-6 words)", "purpose": "communicative purpose of these posts", '
           '"emotion_tone": "dominant emotional tone, 1-3 words", "description": "one or two sentences", '
           '"inclusion_cues": ["3-6 short cues that identify posts belonging here"]}}\n'
           "Rules: be specific to football/r/soccer; name must not be generic (e.g. not 'Football news'); JSON only.")

REVISE = ("Your previous answer:\n{mine}\n\nYour peer's answer:\n{peer}\n\n"
          "Revise your answer to improve agreement with the peer while preserving quality. "
          "Keep the same JSON schema. JSON only.")


def load_llm(model_id: str, device: str):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_id, cache_dir=str(C.LLM_CACHE_DIR))
    dtype = torch.float32 if device == "cpu" else torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, cache_dir=str(C.LLM_CACHE_DIR), torch_dtype=dtype)
    model.to(device).eval()
    log.info("LLM loaded: %s on %s (%s)", model_id, device, dtype)
    return tok, model


def _chat(tok, model, device, system: str, user: str, history: list | None = None) -> str:
    import torch
    msgs = [{"role": "system", "content": system}] + (history or []) + [{"role": "user", "content": user}]
    kwargs = {}
    if "qwen3" in getattr(tok, "name_or_path", "").lower():
        kwargs["enable_thinking"] = False
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, **kwargs)
    inputs = tok(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=C.LLM_MAX_NEW_TOKENS, do_sample=C.LLM_TEMPERATURE > 0,
                             temperature=C.LLM_TEMPERATURE if C.LLM_TEMPERATURE > 0 else None, top_p=0.9,
                             pad_token_id=tok.eos_token_id)
    text = tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return re.sub(r"<think>.*?</think>", "", text, flags=re.S).strip()


def parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, flags=re.S)
    if not m:
        return None
    try:
        d = json.loads(m.group(0))
    except json.JSONDecodeError:
        return None
    if not all(k in d for k in SCHEMA_KEYS):
        return None
    if isinstance(d["inclusion_cues"], str):
        d["inclusion_cues"] = [c.strip() for c in d["inclusion_cues"].split(";") if c.strip()]
    return {k: d[k] for k in SCHEMA_KEYS}


# Stem -> controlled category, checked in order (first match wins), so that
# "frustration and anger" -> frustrated, "anxious and desperate" -> anxious, etc.
_TONE_STEMS = [("frustrat", "frustrated"), ("anger", "angry"), ("angry", "angry"), ("anxi", "anxious"),
               ("despair", "sad"), ("desperat", "anxious"), ("sad", "sad"), ("grie", "sad"),
               ("enthusias", "enthusiastic"), ("excit", "excited"), ("critic", "critical"),
               ("support", "supportive"), ("curio", "curious"), ("inquisit", "curious"), ("cautio", "cautious"),
               ("humor", "humorous"), ("humour", "humorous"), ("funny", "humorous"), ("mixed", "mixed"),
               ("neutral", "neutral"), ("negativ", "negative"), ("positiv", "positive"), ("celebrat", "positive"),
               ("hopeful", "positive"), ("optimis", "positive"), ("disappoint", "negative"), ("dissatisf", "negative")]


def tone_category(free_text: str) -> str:
    """Map a free-text tone to the controlled vocabulary (config.TONE_SET) by word stems."""
    t = str(free_text).lower()
    for stem, cat in _TONE_STEMS:
        if stem in t:
            return cat
    return "other"


def _agree(a: dict, b: dict) -> bool:
    """Agreement = similar names (token overlap >= 0.5) and same controlled tone."""
    ta = set(re.findall(r"[a-z]+", a["name"].lower())); tb = set(re.findall(r"[a-z]+", b["name"].lower()))
    j = len(ta & tb) / max(1, len(ta | tb))
    return j >= 0.5 and tone_category(a["emotion_tone"]) == tone_category(b["emotion_tone"])


def label_topic(tok, model, device, keywords: str, sample_posts: list[str]) -> dict:
    posts = "\n".join(f"- {p[:300]}" for p in sample_posts[: C.LLM_SAMPLE_POSTS])
    user0 = INITIAL.format(keywords=keywords, posts=posts)
    answers, histories = {}, {}
    for agent in C.LLM_AGENTS:
        sysm = SYSTEM.format(agent=agent)
        txt = _chat(tok, model, device, sysm, user0)
        d = parse_json(txt) or parse_json(_chat(tok, model, device, sysm, user0 + "\nReturn valid JSON only."))
        answers[agent] = d
        histories[agent] = [{"role": "user", "content": user0}, {"role": "assistant", "content": txt}]
    turns = 1
    while turns < C.LLM_MAX_TURNS and all(answers.values()) and not _agree(*answers.values()):
        new = {}
        for agent, peer in zip(C.LLM_AGENTS, reversed(C.LLM_AGENTS)):
            u = REVISE.format(mine=json.dumps(answers[agent]), peer=json.dumps(answers[peer]))
            txt = _chat(tok, model, device, SYSTEM.format(agent=agent), u, histories[agent])
            new[agent] = parse_json(txt) or answers[agent]
            histories[agent] += [{"role": "user", "content": u}, {"role": "assistant", "content": txt}]
        answers = new
        turns += 1
    a, b = answers[C.LLM_AGENTS[0]], answers[C.LLM_AGENTS[1]]
    ok = bool(a and b)
    agreed = ok and _agree(a, b)
    final = a or b or {k: "" for k in SCHEMA_KEYS}
    return {"label_name": final["name"], "label_purpose": final["purpose"], "label_tone": final["emotion_tone"],
            "label_tone_category": tone_category(final["emotion_tone"]), "label_description": final["description"],
            "label_cues": "; ".join(final["inclusion_cues"]) if isinstance(final["inclusion_cues"], list) else str(final["inclusion_cues"]),
            "analyst_a": json.dumps(a, ensure_ascii=False) if a else "", "analyst_b": json.dumps(b, ensure_ascii=False) if b else "",
            "debate_turns": turns, "analysts_agree": agreed, "needs_review": (not agreed) or (not ok)}


def label_all(topics: pd.DataFrame, docs: pd.DataFrame, doc_topics: list[int], model_id: str, device: str,
              progress_path=None) -> pd.DataFrame:
    """Label every topic (except -1); resumable through progress_path (JSONL)."""
    import random
    rng = random.Random(C.SEED)
    done = {}
    if progress_path and progress_path.exists():
        for line in progress_path.read_text(encoding="utf-8").splitlines():
            r = json.loads(line); r["label_tone_category"] = tone_category(r.get("label_tone", "")); done[r["topic"]] = r
        log.info("labelling: resuming, %d topics already labelled", len(done))
    tok, model = load_llm(model_id, device)
    d = docs.assign(topic=doc_topics)
    rows = []
    for _, t in topics.iterrows():
        tid = int(t["topic"])
        if tid == -1:
            continue
        if tid in done:
            rows.append(done[tid]); continue
        sub = d[d["topic"] == tid]["document"].tolist()
        rng.shuffle(sub)
        res = label_topic(tok, model, device, t["keywords"], sub)
        res["topic"] = tid; res["llm_model"] = model_id
        rows.append(res)
        if progress_path:
            with open(progress_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(res, ensure_ascii=False) + "\n")
        log.info("labelled topic %d: %s (%s)%s", tid, res["label_name"], res["label_tone"], "" if res["analysts_agree"] else " [needs review]")
    return pd.DataFrame(rows)


def write_merge_table(topics: pd.DataFrame, path) -> None:
    """Editable table for co-authors: set `merge_into` to another topic id to merge."""
    cols = ["topic", "size", "keywords"] + [c for c in ("label_name", "label_tone") if c in topics.columns]
    t = topics[cols].copy()
    t["merge_into"] = ""
    t["final_name"] = t.get("label_name", "")
    t["notes"] = ""
    t.to_csv(path, index=False)


def apply_merges(doc_topics: list[int], merge_table: pd.DataFrame) -> list[int]:
    m = {int(r["topic"]): int(r["merge_into"]) for _, r in merge_table.iterrows()
         if str(r.get("merge_into", "")).strip() not in ("", "nan")}
    # resolve chains a->b->c
    def res(t):
        seen = set()
        while t in m and t not in seen:
            seen.add(t); t = m[t]
        return t
    return [res(int(t)) for t in doc_topics]
