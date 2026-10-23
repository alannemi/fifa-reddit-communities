"""
Prompt v1.1 for the two-agent labelling (design §7.6, logged verbatim in every bin's config.json).
=====================================================================================================
Two stages share the analysts and the revision turn:
  S3a  cluster labelling — one category per BERTopic topic (keywords + up to 40 sample posts)
  S3b  discourse taxonomy — the bin's labelled topics are grouped into categories with explicit `topics` lists
The analysts differ in one instruction only (the "lens", design §7.2) so that greedy decoding still yields two
independent judgements.  Revision turns are stateless: the items block is repeated, followed by the two latest
answers; no chat history is carried (design §7.4, F20).
Refinements of the brief's prompt are documented in design §13: F1, F2, F3, F16, F17, F20, F21.
"""
from __future__ import annotations

PROMPT_VERSION = "1.1"

AGENTS = ("Analyst A", "Analyst B")
LENSES = {
    "Analyst A": "Start from the distinctive keywords, then check them against the posts.",
    "Analyst B": "Start from reading the posts, then check them against the keywords.",
}

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

# Revision turn (both stages). The stage's items block is repeated above this text; no chat history.
USER_REVISION = (
    "Your previous answer:\n"
    "{mine}\n"
    "Your peer's latest answer:\n"
    "{peer}\n"
    'If the peer\'s answer is at least as good as yours, return {{"decision": "accept"}} and nothing else.\n'
    'Otherwise return {{"decision": "revise", "categories": [...]}} with a revised answer that improves agreement with the\n'
    "peer while preserving quality. Same schema. JSON only."
)

# Repair turn: appended when an answer could not be parsed or failed validation (design §7.1).
USER_REPAIR = (
    "Your previous answer could not be used: {error}\n"
    "Return the corrected answer with the same schema. JSON only; no prose or markdown."
)


def system_prompt(stage: str, agent_name: str) -> str:
    tpl = SYSTEM_LABEL if stage == "label" else SYSTEM_TAXONOMY
    return tpl.format(agent_name=agent_name, lens=LENSES[agent_name])


def render_label_items(keywords: str, posts: list[str]) -> str:
    """The items block of S3a: keywords + sample posts (one per line, title first)."""
    body = "\n".join(f"- {p}" for p in posts)
    return USER_LABEL.format(keywords=keywords, posts=body)


def render_taxonomy_items(n_topics: int, start: str, end: str, topic_lines: list[str]) -> str:
    return USER_TAXONOMY.format(n_topics=n_topics, start=start, end=end, topic_lines="\n".join(topic_lines))


def render_revision(items_block: str, mine: str, peer: str) -> str:
    return items_block + "\n\n" + USER_REVISION.format(mine=mine, peer=peer)


def render_repair(items_block: str, error: str) -> str:
    return items_block + "\n\n" + USER_REPAIR.format(error=error)


def topic_line(topic_id: int, name: str, tone: str, keywords: list[str], description: str) -> str:
    """One compact S3b item (≈ 60 tokens): topic_id | name | emotion_tone | keywords | one-sentence description."""
    desc = " ".join(str(description).split())
    first_sentence = desc.split(". ")[0].rstrip(".") + "." if desc else ""
    return f"{topic_id} | {name} | {tone} | {', '.join(keywords)} | {first_sentence}"


def all_prompts() -> dict:
    """Everything a run needs to log so the prompt is reproducible from config.json alone."""
    return {"prompt_version": PROMPT_VERSION, "agents": list(AGENTS), "lenses": LENSES,
            "system_label": SYSTEM_LABEL, "user_label": USER_LABEL, "system_taxonomy": SYSTEM_TAXONOMY,
            "user_taxonomy": USER_TAXONOMY, "user_revision": USER_REVISION, "user_repair": USER_REPAIR}
