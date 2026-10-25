"""
Prompts for the two-agent cluster labelling — version 1.4 (logged verbatim in every bin's config.json).
=====================================================================================================
One stage uses the analysts: S3a cluster labelling — one label per BERTopic topic (15 keywords + up to 40 sample posts),
agreed by the two analysts under the D2 protocol. The analysts differ in one instruction only (the "lens", design §7.2)
so that greedy decoding still yields two independent judgements. Revision turns are stateless: the items block is
repeated, followed by the two latest answers; no chat history is carried (design §7.4, F20).
Version history: 1.1 (design §7.6, 2026-10-22: F1, F2, F3, F16, F17, F20, F21) → 1.2 (2026-10-24: the two-step S3b texts,
`{answer_key}` in the revision turn) → 1.3 (2026-10-25: the S3b shortlist text) → 1.4 (2026-10-25, decision D7: the
taxonomy stage and its texts removed; the S3a text is byte-identical to v1.1, the revision text to v1.2).
Approvals: docs/session-05_2026-10-23/00_approvals.md; docs/session-08_2026-10-25/00_colab-run-log.md (D7).
"""
from __future__ import annotations

PROMPT_VERSION = "1.4"

AGENTS = ("Analyst A", "Analyst B")
LENSES = {
    "Analyst A": "Start from the distinctive keywords, then check them against the posts.",
    "Analyst B": "Start from reading the posts, then check them against the keywords.",
}

# ---------------------------------------------------------------------------
# S3a — cluster labelling (unchanged since v1.1)
# ---------------------------------------------------------------------------
SYSTEM_LABEL = (
    "You are {agent_name}, a discourse analyst studying posts from the football community r/soccer.\n"
    "{lens}\n"
    "Label one cluster of posts as one category. Return JSON only; no prose or markdown."
)

USER_LABEL = (
    "Keywords: {keywords}\n"
    "Posts (one per line, title first):\n"
    "{posts}\n"
    "Output schema:\n"
    '{{"decision": "revise",\n'
    ' "categories": [\n'
    '  {{"name": "...", "purpose": "...", "emotion_tone": "...", "description": "...", "inclusion_cues": ["..."]}}\n'
    " ]}}\n"
    "Rules: exactly one category; name short (2–6 words), specific to football/r/soccer, not generic; capture both the\n"
    "communicative purpose and the emotional tone; 3–6 inclusion cues."
)

# ---------------------------------------------------------------------------
# Shared turns
# ---------------------------------------------------------------------------
# Revision turn. The stage's items block is repeated above this text; no chat history. {answer_key} = "categories".
USER_REVISION = (
    "Your previous answer:\n"
    "{mine}\n"
    "Your peer's latest answer:\n"
    "{peer}\n"
    'If the peer\'s answer is at least as good as yours, return {{"decision": "accept"}} and nothing else.\n'
    'Otherwise return {{"decision": "revise", "{answer_key}": [...]}} with a revised answer that improves agreement with the\n'
    "peer while preserving quality. Same schema. JSON only."
)

# Repair turn: appended when an answer could not be parsed or failed validation (design §7.1).
USER_REPAIR = (
    "Your previous answer could not be used: {error}\n"
    "Return the corrected answer with the same schema. JSON only; no prose or markdown."
)

SYSTEM_PROMPTS = {"label": SYSTEM_LABEL}
ANSWER_KEYS = {"label": "categories"}


def system_prompt(stage: str, agent_name: str) -> str:
    """System text of a stage for one analyst; raises on an unknown stage (never falls back silently)."""
    if stage not in SYSTEM_PROMPTS:
        raise KeyError(f"unknown prompt stage {stage!r}; expected one of {sorted(SYSTEM_PROMPTS)}")
    return SYSTEM_PROMPTS[stage].format(agent_name=agent_name, lens=LENSES[agent_name])


def render_label_items(keywords: str, posts: list[str]) -> str:
    """The items block of S3a: keywords + sample posts (one per line, title first)."""
    body = "\n".join(f"- {p}" for p in posts)
    return USER_LABEL.format(keywords=keywords, posts=body)


def render_revision(items_block: str, mine: str, peer: str, answer_key: str = "categories") -> str:
    return items_block + "\n\n" + USER_REVISION.format(mine=mine, peer=peer, answer_key=answer_key)


def render_repair(items_block: str, error: str) -> str:
    return items_block + "\n\n" + USER_REPAIR.format(error=error)


def all_prompts() -> dict:
    """Everything a run needs to log so the prompts are reproducible from config.json alone."""
    return {"prompt_version": PROMPT_VERSION, "agents": list(AGENTS), "lenses": LENSES,
            "system_label": SYSTEM_LABEL, "user_label": USER_LABEL,
            "user_revision": USER_REVISION, "user_repair": USER_REPAIR}


def stage_prompts(stage: str) -> dict:
    """The texts one stage actually uses (written into that stage's section of config.json, review finding R36)."""
    keys = {"label": ["system_label", "user_label"]}
    allp = all_prompts()
    return {k: allp[k] for k in ["prompt_version", "agents", "lenses"] + keys[stage] + ["user_revision", "user_repair"]}
