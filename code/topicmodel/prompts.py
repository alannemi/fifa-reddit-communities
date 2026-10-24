"""
Prompts for the two-agent labelling — version 1.2 (logged verbatim in every bin's config.json).
=================================================================================================
Stages that share the analysts, the lenses and the revision turn:
  S3a  cluster labelling   — one category per BERTopic topic (keywords + up to 40 sample posts)   [text = v1.1]
  S3b  discourse taxonomy  — two-step protocol (decision D5, 2026-10-24; design §7.4 v1.2):
         step 1 "define":  the bin's labelled topics are the items; the analysts define the categories (no topic ids)
         step 2 "assign":  one item per topic (configurable batch); the analysts assign it to one of the agreed categories
       The v1.1 single-prompt protocol ("taxonomy": categories with explicit `topics` lists, chunked above 100 items)
       is kept for comparison and replay (config.TAXONOMY_PROTOCOL = "partition").
The analysts differ in one instruction only (the "lens", design §7.2) so that greedy decoding still yields two
independent judgements.  Revision turns are stateless: the items block is repeated, followed by the two latest
answers; no chat history is carried (design §7.4, F20).  The revision text names the stage's answer key
(`categories` or `assignments`) — the only change to the v1.1 revision turn.
Version history: 1.1 (design §7.6, 2026-10-22: F1, F2, F3, F16, F17, F20, F21) → 1.2 (2026-10-24: step-1 and step-2
texts of the two-step S3b, `{answer_key}` in the revision turn; the S3a texts are byte-identical to v1.1).
Approvals: docs/session-06_2026-10-24/00_approvals.md §3 and §7.
"""
from __future__ import annotations

PROMPT_VERSION = "1.2"

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
# S3b, two-step protocol (v1.2) — step 1: definitions
# ---------------------------------------------------------------------------
SYSTEM_DEFINE = (
    "You are {agent_name}, a discourse analyst studying posts from the football community r/soccer.\n"
    "{lens}\n"
    "Build a discourse taxonomy from the items: define its categories. Use the keywords and the labelled sample of each item.\n"
    "Return JSON only; no prose or markdown."
)

USER_DEFINE = (
    "Items: the {n_topics} topics found in r/soccer posts between {start} and {end}, one per line:\n"
    "topic_id | name | emotion_tone | keywords | description\n"
    "{topic_lines}\n"
    "Output schema:\n"
    '{{"decision": "revise",\n'
    ' "categories": [\n'
    '  {{"id": "cat_1", "name": "...", "purpose": "...", "emotion_tone": "...", "description": "...", "inclusion_cues": ["..."]}}\n'
    " ]}}\n"
    "Rules: keep category names short and non-overlapping; capture both communicative purpose and emotional tone.\n"
    "There is no minimum or maximum number of categories: define a category for a group of topics only if they are\n"
    "sufficiently similar; the categories together must cover all topics above. Do not list topic ids; the topics are\n"
    "assigned to the categories in a second step."
)

# ---------------------------------------------------------------------------
# S3b, two-step protocol (v1.2) — step 2: assignment
# ---------------------------------------------------------------------------
SYSTEM_ASSIGN = (
    "You are {agent_name}, a discourse analyst studying posts from the football community r/soccer.\n"
    "{lens}\n"
    "Assign each topic to exactly one category of the agreed discourse taxonomy. Use the keywords and the labelled sample\n"
    "of each item. Return JSON only; no prose or markdown."
)

USER_ASSIGN = (
    "Categories (the agreed taxonomy of r/soccer posts between {start} and {end}), one per line:\n"
    "category_id | name | emotion_tone | inclusion_cues | description\n"
    "{category_lines}\n"
    "Topics to assign ({n_batch} of the bin's {n_topics} topics), one per line:\n"
    "topic_id | name | emotion_tone | keywords | description\n"
    "{topic_lines}\n"
    "Output schema:\n"
    '{{"decision": "revise",\n'
    ' "assignments": [\n'
    '  {{"topic_id": <topic_id>, "category_id": "<category_id>"}}\n'
    " ]}}\n"
    "Rules: list every topic_id above exactly once, each with the single best-fitting category_id; use only the category\n"
    "ids listed; if no category fits well, choose the closest one."
)

# ---------------------------------------------------------------------------
# S3b, single-prompt protocol (v1.1; config.TAXONOMY_PROTOCOL = "partition")
# ---------------------------------------------------------------------------
SYSTEM_TAXONOMY = (
    "You are {agent_name}, a discourse analyst studying posts from the football community r/soccer.\n"
    "{lens}\n"
    "Build a discourse taxonomy from the items. Use the keywords and the labelled sample of each item.\n"
    "Return JSON only; no prose or markdown."
)

USER_TAXONOMY = (
    "Items: the {n_topics} topics found in r/soccer posts between {start} and {end}, one per line:\n"
    "topic_id | name | emotion_tone | keywords | description\n"
    "{topic_lines}\n"
    "Output schema:\n"
    '{{"decision": "revise",\n'
    ' "categories": [\n'
    '  {{"id": "cat_1", "name": "...", "purpose": "...", "emotion_tone": "...", "description": "...",\n'
    '   "inclusion_cues": ["..."], "topics": [topic_id, ...]}}\n'
    " ]}}\n"
    "Rules: keep category names short and non-overlapping; capture both communicative purpose and emotional tone.\n"
    "There is no minimum or maximum number of categories: group topics into one category only if they are sufficiently\n"
    "similar; every topic_id must appear in exactly one category."
)

# ---------------------------------------------------------------------------
# Shared turns
# ---------------------------------------------------------------------------
# Revision turn (all stages). The stage's items block is repeated above this text; no chat history.
# {answer_key} = "categories" (S3a, step 1, v1.1 taxonomy) or "assignments" (step 2).
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

SYSTEM_PROMPTS = {"label": SYSTEM_LABEL, "define": SYSTEM_DEFINE, "assign": SYSTEM_ASSIGN, "taxonomy": SYSTEM_TAXONOMY}
ANSWER_KEYS = {"label": "categories", "define": "categories", "assign": "assignments", "taxonomy": "categories"}


def system_prompt(stage: str, agent_name: str) -> str:
    """System text of a stage for one analyst; raises on an unknown stage (never falls back silently)."""
    if stage not in SYSTEM_PROMPTS:
        raise KeyError(f"unknown prompt stage {stage!r}; expected one of {sorted(SYSTEM_PROMPTS)}")
    return SYSTEM_PROMPTS[stage].format(agent_name=agent_name, lens=LENSES[agent_name])


def render_label_items(keywords: str, posts: list[str]) -> str:
    """The items block of S3a: keywords + sample posts (one per line, title first)."""
    body = "\n".join(f"- {p}" for p in posts)
    return USER_LABEL.format(keywords=keywords, posts=body)


def render_define_items(n_topics: int, start: str, end: str, topic_lines: list[str]) -> str:
    """The items block of S3b step 1: all topic lines of the bin."""
    return USER_DEFINE.format(n_topics=n_topics, start=start, end=end, topic_lines="\n".join(topic_lines))


def render_assign_items(start: str, end: str, category_lines: list[str], n_batch: int, n_topics: int,
                        topic_lines: list[str]) -> str:
    """The items block of S3b step 2: the agreed categories + the topic line(s) of one item."""
    return USER_ASSIGN.format(start=start, end=end, category_lines="\n".join(category_lines), n_batch=n_batch,
                              n_topics=n_topics, topic_lines="\n".join(topic_lines))


def render_taxonomy_items(n_topics: int, start: str, end: str, topic_lines: list[str]) -> str:
    """The items block of the v1.1 single-prompt taxonomy."""
    return USER_TAXONOMY.format(n_topics=n_topics, start=start, end=end, topic_lines="\n".join(topic_lines))


def render_revision(items_block: str, mine: str, peer: str, answer_key: str = "categories") -> str:
    return items_block + "\n\n" + USER_REVISION.format(mine=mine, peer=peer, answer_key=answer_key)


def render_repair(items_block: str, error: str) -> str:
    return items_block + "\n\n" + USER_REPAIR.format(error=error)


def topic_line(topic_id: int, name: str, tone: str, keywords: list[str], description: str) -> str:
    """One compact topic item (≈ 60 tokens): topic_id | name | emotion_tone | keywords | one-sentence description."""
    desc = " ".join(str(description).split())
    first_sentence = desc.split(". ")[0].rstrip(".") + "." if desc else ""
    return f"{topic_id} | {name} | {tone} | {', '.join(keywords)} | {first_sentence}"


def category_line(category_id: str, name: str, tone: str, cues: list[str], description: str) -> str:
    """One compact category item for step 2: category_id | name | emotion_tone | inclusion cues | one-sentence description."""
    desc = " ".join(str(description).split())
    first_sentence = desc.split(". ")[0].rstrip(".") + "." if desc else ""
    return f"{category_id} | {name} | {tone} | {'; '.join(cues)} | {first_sentence}"


def all_prompts() -> dict:
    """Everything a run needs to log so the prompts are reproducible from config.json alone."""
    return {"prompt_version": PROMPT_VERSION, "agents": list(AGENTS), "lenses": LENSES,
            "system_label": SYSTEM_LABEL, "user_label": USER_LABEL,
            "system_define": SYSTEM_DEFINE, "user_define": USER_DEFINE,
            "system_assign": SYSTEM_ASSIGN, "user_assign": USER_ASSIGN,
            "system_taxonomy_v1_1": SYSTEM_TAXONOMY, "user_taxonomy_v1_1": USER_TAXONOMY,
            "user_revision": USER_REVISION, "user_repair": USER_REPAIR}


def stage_prompts(stage: str) -> dict:
    """The texts one stage actually uses (written into that stage's section of config.json, review finding R36)."""
    keys = {"label": ["system_label", "user_label"],
            "taxonomy": ["system_define", "user_define", "system_assign", "user_assign", "system_taxonomy_v1_1", "user_taxonomy_v1_1"]}
    allp = all_prompts()
    return {k: allp[k] for k in ["prompt_version", "agents", "lenses"] + keys[stage] + ["user_revision", "user_repair"]}
