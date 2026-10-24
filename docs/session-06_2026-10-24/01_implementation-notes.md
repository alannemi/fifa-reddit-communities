# Session 06 (2026-10-24) — Implementation notes: the two-step taxonomy protocol (D5) and the Colab integration

**Author of the code: Claude; decisions: Alan Nemirovski** (`00_approvals.md`). Read with the design
(`../session-04_2026-10-22/01_analytical-pipeline-design.md`, v1.2 — §7.4 for the protocol, §7.6 for prompt v1.2) and the
session-05 notes (`../session-05_2026-10-23/01_implementation-notes.md`, which still describe everything that did not
change). This note says what changed in the code and why, which outputs gained columns, and how the Colab notebook was
prepared for the bin-1 test batch. The plan approved before implementation is reproduced in §7.

## 1. What changed (design §7.4 v1.2 → files)

| File | Change |
|---|---|
| `code/topicmodel/prompts.py` | `PROMPT_VERSION = "1.2"`; new texts `SYSTEM_DEFINE` / `USER_DEFINE` (step 1) and `SYSTEM_ASSIGN` / `USER_ASSIGN` (step 2); `USER_REVISION` names the stage's answer key (`{answer_key}` = `categories` or `assignments`, the only change to the v1.1 revision text); `system_prompt()` is a dict lookup that raises on an unknown stage; `render_define_items()`, `render_assign_items()`, `category_line()`, `render_revision(…, answer_key)`; `all_prompts()` logs every text, `stage_prompts(stage)` the texts a stage uses. The v1.1 S3b texts stay (`SYSTEM_TAXONOMY`, `USER_TAXONOMY`) for the `partition` protocol; the S3a texts are byte-identical to v1.1 |
| `code/topicmodel/config.py` | `TAXONOMY_PROTOCOL = "two_step"`, `TAXONOMY_ASSIGN_BATCH = 1`, `LLM_MAX_NEW_TOKENS_DEFINE = 4000`, `LLM_MAX_NEW_TOKENS_ASSIGN = 800`; `TAXONOMY_MAX_ITEMS` / `TAXONOMY_CHUNK_SIZE` / `LLM_MAX_NEW_TOKENS_TAXONOMY` marked as `partition`-only; `LLM_MODEL_LARGE` marked unused (the Qwen3-4B comparison was dropped) |
| `code/topicmodel/agents.py` | `Debate(…, max_new_tokens, salvage, answer_key)`: the token budget, the recovery function and the revision answer key are per stage; `_parse` calls the stage's salvage (`_default_salvage` = N11 as approved, never on truncated answers; `salvage_definitions` recovers the complete category objects of a truncated step-1 answer and drops the cut-off one; `salvage_assignments` recovers `{topic_id, category_id}` pairs); revision turns show the peer's answer without the pipeline's bookkeeping keys (`_public_json`). Step 1: `validate_definitions` (canonical ids `cat_1…cat_K`, `id_given` kept; hard failure only without any named category; duplicated names / empty fields are soft problems returned with the parsed answer), `cleanup_definitions` (same-name merge, re-canonicalisation, log), `definitions_identical`. Step 2: `build_category_index`, `resolve_category_ref` (exact id, bare number, `cat 3`, normalised name, id inside text; `null`/"none"/unknown = missing), `_pairs_of` (object list, mapping form, the v1.1 habit of categories with topic lists), `validate_assignments` (structural failure → no answer; coverage problems → cleaned answer kept with the error; `resolved_from` recorded), `assignments_identical`. `run_taxonomy()` dispatches on the protocol; `run_taxonomy_two_step()` (definitions debate → `taxonomy_definitions.json` checkpoint → one item per topic → assignment debate → merge → `repair_partition` → writer); `run_taxonomy_partition()` = the v1.1 body; `_write_taxonomy_outputs()` shared by both; `_write_review_templates()` regenerates `category_overrides.csv` and writes `topic_category_overrides.csv` unless filled in; `read_final_categories()` for S5 |
| `code/topicmodel/tagging.py` | `LABEL_COLUMNS` + `assignment_item`, `assignment_round_iterations`, `assignment_agreement_mode`, `assignment_contested`, `assignment_needs_review` (joined from `topic_categories.csv`; empty for `partition` outputs); F28 applied in `load_bin_outputs` in this order: `category_id_agent` = the agents' value → `final_category_id` moves on the surviving topic after `merge_into` (an entry on a merged-away topic is ignored with a warning; an entry on a merge target moves all merged posts) → category merges → sizes recomputed; a category emptied by moves stays with `n_topics = 0`; an unknown target fails the S5 check |
| `code/06-analysis-pipeline.py` | `--taxonomy-protocol two_step|partition`, `--assign-batch N`; `--force` removes the step-1 checkpoint; the skip message names the existing result's protocol; `config.json → taxonomy` records protocol, batch, categories defined / final / dropped, cleanup, assignment modes and contested items, repaired / unplaced counts, salvaged answers, token budgets and the stage's prompt texts; `config.json → label` records its prompt texts too (review finding R36) |
| `code/tests/test_pipeline_units.py` | 19 checks (15 before): `test_two_step_validators`, `test_salvage_assignments_and_truncated_definitions`, `test_taxonomy_two_step` (45 fixture topics: definitions with duplicated names and non-canonical ids, `accept` in round 2, an unused category dropped, assignment answers in list / mapping / name form, `null`, a duplicated topic, a foreign id, salvage, `no_answer`, resume from the checkpoint, batching with 3 topics per item, the revision-turn regression check), `test_taxonomy_two_step_cap`; `test_collect_partial` extended with the F28 cases; the partition tests call `protocol="partition"`; the stub model understands `TRUNC:<text>` as a cut-off answer |
| `code/06b-analysis-pipeline-colab.ipynb` | §3 below |

Unchanged: `bins.py`, `embed.py`, `model.py`, `sentiment.py`, `report.py`, `07-tag-corpus.R`, `08-validate-analysis.R`,
the eight codebook variables and `post_variables.csv`.

## 2. Definitions that are new or more specific than the design text

* **Canonical category ids.** Step 1 assigns `cat_1 … cat_K` in the order the model gave the categories (the model's own
  id is kept as `id_given` when it differs); the cleanup after a failed repair re-canonicalises once more; from step 2
  on nothing renumbers, so dropped empty categories leave gaps (`cat_1, cat_2, cat_4`).
* **Soft validation of definitions.** Duplicated names and empty text fields do not void an answer: the answer is kept
  with the error, the repair turn quotes it, and if the repair fails the cleanup merges same-name categories (first texts
  kept, cues united, empty fields filled from the duplicate) and logs it (`taxonomy.json → definitions_cleanup`).
* **Lenient assignment references** (F30). A reference is resolved to a category id by exact id, bare number
  (`3`, `cat 3`, `Category 3`), normalised name, an id inside the text, or a long name fragment; `null`, `""`, `none`,
  `n/a`, `other`, `unknown` and unresolvable references count as *missing* (the topic is then placed by the repair unless
  a later pair names a category). `resolved_from` in the parsed answer records every non-exact resolution.
* **Contested / needs review per topic.** `assignment_contested` = the two analysts' final answers do not name the
  same category for the topic (a missing answer counts as contested); `assignment_needs_review` = the item was not
  agreed **and** the topic is contested. With one topic per item, `assignment_round_iterations` and
  `assignment_agreement_mode` are exact per topic.
* **Resume.** `taxonomy_definitions.json` holds the agreed definitions, the cleanup log, the definitions debate and the
  topic ids it was built for; a re-run of the stage reuses it when the topic set is unchanged (a Colab disconnect during
  step 2 costs only step 2). `--force` deletes it, so both steps run again.
* **Templates.** `category_overrides.csv` and `topic_category_overrides.csv` are (re)written by every taxonomy run
  unless a co-author has filled them in; then they are kept and a warning says their ids may be stale.
* **`n_answers_salvaged`** counts salvaged answers of both steps; `truncated` in `llm_stats` counts cut-off generations.

## 3. The Colab notebook (`code/06b-analysis-pipeline-colab.ipynb`, 28 cells)

* Cell 0.1 opens a **session log on Drive** (`MyDrive/fifa-reddit-communities/logs/colab_<date>.log`); `slog()` prints
  and appends; cells 0.2 and 0.3 write the commit, `pip check`, GPU and versions into it.
* Cell 0.4's `stage()` runs the program **unbuffered** (`python -u`, progress bars off), streams every line to the
  notebook and the log, and raises on a non-zero exit (review finding R33); `taxonomy_summary(b)` prints the two-step
  outcome (definitions debate, categories defined / final / dropped, assignment modes, contested items, repaired /
  unplaced topics, salvaged answers, the category table).
* Pass 1 runs the two-step taxonomy over the 12 labelled topics and asserts `protocol == "two_step"`; cell 3.1 runs the
  taxonomy over all topics with `--force`, prints the summary, runs a partial collect and marks the **stop point of the
  session-06 test batch**; a new markdown cell explains the hand-back.
* The comparison cell keeps only the seeded-sampling branch (the Qwen3-4B comparison was dropped on 2026-10-24).
* Section 5 zips `results/analysis/bin-1/` (without `model/`), the aggregated tables, `data/analysis/*.csv|json|txt`
  and the session log into `MyDrive/fifa-reddit-communities/handback/bin-1_<date>.zip`.
* `requirements-colab.txt` unchanged. Not verifiable from the Mac (F27): the `--no-deps` install on the current Colab
  image, cuML's UMAP with the cosine metric, the Drive mount from the VS Code kernel — all asserted by the notebook.

## 4. Local verification before Colab (step C of the plan)

Unit checks: 19/19. The two-step taxonomy with the real model (Qwen3-1.7B on MPS) was started on a copy of the 92-topic
dry run (`results/dry-runs/bin1-twostep/`, `--stage taxonomy --bin 1 --force`) at 15:27 and **stopped at 15:35 by the
author's decision** (`00_approvals.md` §8): round 1 of the definitions debate alone took 7 minutes on the Mac (142 s and
260 s per answer; the answers differed, so a revision round followed), and the whole stage was estimated at 1.5–2 hours.
The first run of the protocol with the real model is therefore the Colab test batch (`02_colab-run-log.md`). The partial
local folder is kept (gitignored) for the record; its `run-twostep.log` ends with the stop note.

## 5. Running

See `memory/workflows.md` §7 (`--taxonomy-protocol`, `--assign-batch`, the dry-run recipe for the two-step stage) and
the Colab notebook. After any change of `topic_overrides.csv`, `--stage taxonomy --bin N --force` re-runs both steps.

## 6. Known limits to keep in mind

* The repair places a topic only against categories that already have members; an agreed but unused category is
  dropped rather than used (F23).
* Lenient name matching could, in principle, pick the wrong category when two category names share a long fragment;
  `resolved_from` makes every such resolution visible.
* Seeded-sampling mode uses the per-item seed prefixes `-def` / `-assign`; it was unit-tested only through the
  deterministic stub.

## 7. The approved plan (2026-10-24)

The plan file approved before implementation (`~/.claude/plans/groovy-dancing-acorn.md` on the author's machine) is
reproduced verbatim below.

## Session 06 plan — two-step taxonomy (S3b), Colab integration, bin-1 test batch on the GPU

**Prepared 2026-10-24 for Alan Nemirovski (decisions) · FIFA Communities chapter.** Basis: design v1.1
(`docs/session-04_2026-10-22/01_analytical-pipeline-design.md`), the session-05 records (`docs/session-05_2026-10-23/`),
the 92-topic dry run (`results/dry-runs/bin1-full92/`), the code as it stands (unit checks 15/15 on the working tree).
Reviewed by an independent reviewer on 2026-10-24 (27 findings, all folded in; §10). Markers: **[decided]** = your
answers of 2026-10-24 (the first round before planning and the planning round), **[default]** = applied when this plan
is approved unless you object, **[flag]** = a consequence you should know about (F22–F30, continuing the design's list).

---

### 1. Context

Session 05 built the pipeline and dry-ran it on bin 1. The background run over all 92 bin-1 topics finished after the
session and showed one blocking defect (finding D2): Qwen3-1.7B cannot partition a whole bin's topics in one prompt.
Both round-1 answers hit the 8,000-token cap; the repair answers each proposed 10 categories but listed 443–605 topic
ids for 92 topics (whole id ranges repeated); the final answer's 373 duplicated ids collapsed under the repair's
first-listed-wins rule to 4 non-empty categories, with 23 of 92 topics placed by embedding similarity, after 99 minutes
on the Mac. The model did not "find 4 categories"; it could not enumerate 92 ids. The labelling stage, by contrast, is
sound: 92 topics, `needs_review` 2 (2.2 %), and the 12 "no answer" topics seen before N12 were all caused by the
validator's own 1–10 cue cap.

Decisions taken today **[decided]**:

| # | Decision | Chosen |
|---|---|---|
| D5 | S3b strategy for finding D2 | **Two-step protocol: the agents first agree on the category definitions of a bin, then assign the topics to those categories** (alternatives declined: chunking at 30/25; Qwen3-4B for S3b; a single assignment-style prompt) |
| N11 | Tolerant JSON recovery (`LLM_JSON_SALVAGE = True`) | **Confirmed** |
| N12 | Over-long inclusion-cue lists kept (first ten, `n_cues_given` recorded) | **Confirmed** |
| — | Scope of the Colab test batch | **Pass 1 (every stage once, 12 labelled topics) + the full bin 1 (all topics labelled, two-step taxonomy)**; the large-bin gate, `T`, sentiment hand-coding and bins 2–9 come after |
| — | My access to the Colab outputs | **Google Drive connector** (you authorize it in claude.ai → Settings → Connectors; its tools were not visible in this session at planning time, so the plan also works with pasted cell output and a download of the results folder) |
| — | Prompt v1.2 (planning round) | **Approved as drafted** (§2.3); no example topic ids in the definitions |
| — | Assignment granularity (planning round) | **One topic per debate item** (`TAXONOMY_ASSIGN_BATCH = 1`); batching stays a config option |
| — | Running the Colab cells (planning round) | **You run the cells in VS Code on the Colab kernel**; the 17 yearly files are **already on Drive**; failures come back as pasted output or through the Drive log |
| — | Comparisons (planning round) | **No Qwen3-4B comparison at all**: the pipeline stays with Qwen3-1.7B (design F9's optional 4B run is dropped); the seeded-sampling comparison (F17) remains a gate-4 option for later |

Goal of this session: implement D5, finish the Colab integration, run the test batch on an A100, and leave the
repository documented so that the next session runs the gates and bins 2–9.

### 2. The two-step taxonomy protocol (design §7.4 v1.2)

Unchanged from the approved design: the two analysts and their lenses; the D2 debate engine (explicit acceptance,
alternating turns, cap 5, stalemate rule, one repair turn, no-answer rule); stateless revision turns; the topic-line
items (`topic_id | name | emotion_tone | 8 keywords | one-sentence description`); the deterministic repair with
`category_assignment ∈ {agent, repaired, unplaced}`; the output files consumed by S5 and the R scripts; the eight
codebook variables.

#### 2.1 Step 1 — definitions (one debate per bin)

* **Input:** all topic lines of the bin (92 lines ≈ 6k tokens; the model's context was never the problem, its output
  was). **Task:** define the categories: `id, name, purpose, emotion_tone, description, inclusion_cues`; no topic ids.
  Output ≈ 10–20 categories × ≈ 100 tokens; budget `LLM_MAX_NEW_TOKENS_DEFINE = 4000`.
* **Validation (`validate_definitions`).** Hard failure (no answer after the repair turn): no category with a name.
  Soft problems, returned *with* the parsed object so the debate keeps the answer and the repair turn quotes them,
  exactly as the partition validator does today: duplicated names, an empty text field in a category, a `topics`
  field (ignored). If the repair turn does not cure them, a deterministic cleanup applies and is logged in
  `taxonomy.json → definitions_repair`: same-name categories merged (first texts kept, cues united), empty fields left
  empty. Cues follow N12 (first ten). **Ids are canonicalised here**: `cat_1 … cat_K` in the order given, whatever the
  model wrote; nothing renumbers them afterwards, so the transcripts, the assignment prompts and `categories.csv` all
  speak of the same ids (dropped empties leave gaps, e.g. `cat_1, cat_2, cat_4`).
* **Salvage.** `salvage_json` as for every answer, and additionally for a *truncated* definitions answer the complete
  category objects are recovered (they carry no topic lists that could be wrong); `parsed_via = "salvage"`,
  `salvage_reason = "truncated"`.
* **Agreement rule (`definitions_identical`):** same set of normalised names and, per name, the same `tone_category`
  (the v1.1 S3b rule without the partition). Under two lenses this rarely fires in round 1; agreement will mostly come
  through `accept` in round 2, as in S3a, which means Analyst B's definitions are adopted in most bins **[flag F22]**.
* **Checkpoint.** The agreed definitions are written to `taxonomy_definitions.json` before step 2 and reused on a
  re-run after a disconnect; `--force` deletes it.
* **Record:** `round_iterations`, `agreed`, `agreement_mode`, `needs_review` of this debate go on every row of
  `categories.csv` (as the single debate's values did before).

#### 2.2 Step 2 — assignment (one debate item per topic, batchable)

* **Items.** Default **one topic per debate item** (`TAXONOMY_ASSIGN_BATCH = 1`): exactly the S3a structure, exact
  per-topic rounds and modes, no batch artefact, and the 92 items are still generated 16 at a time on the GPU
  (≈ 1 min per analyst-round on an A100) **[decided]**. Batching several topics per prompt stays available through the
  same setting (keys zero-padded `assign_001 …`, the item recorded per topic) **[flag F24]**.
* **Input per item:** the agreed category list (`category_id | name | emotion_tone | inclusion_cues | one-sentence
  description`) and the item's topic line(s). **Task:** one `category_id` per `topic_id`. Budget
  `LLM_MAX_NEW_TOKENS_ASSIGN = 800`.
* **Validation (`validate_assignments`).** Structural failure → `(None, err)`; coverage problems → `(cleaned mapping,
  err)` so the answer is kept for the deterministic repair (the contract `validate_taxonomy` uses today). Accepted
  forms: a list of `{topic_id, category_id}` objects or a `{"topic_id": "category_id"}` mapping. Category references
  are resolved leniently — exact id, `cat 3` / `3`, or the category's normalised name (unique after step 1); `null`,
  `"none"`, `""` or an unknown reference count as *missing*, never as a failed batch. Topic ids: every id of the item
  exactly once (`partition_problems` reused); duplicates → first occurrence. The schema example uses placeholders, not
  real ids, so the model cannot copy `0 / cat_1`.
* **Salvage.** `salvage_assignments()` recovers `topic_id`/`category_id` pairs from bracket-damaged answers; the
  `Debate` engine gets a per-stage `salvage` hook (today `_parse` calls `salvage_json` unconditionally and its regex
  cannot see assignment objects).
* **Agreement rule (`assignments_identical`):** identical cleaned mapping. Forced choice, no "none": the model knows
  more than a cosine to a member mean; a bad fit is for the co-authors (§2.5, F28).
* **Deterministic repair after all items:** mappings merged; missing topics placed by cosine similarity of the topic
  embedding to the member-mean of the non-empty categories (`repair_partition`, unchanged); a defined category that
  ends with no topic is dropped from `categories.csv` and listed by the caller in `taxonomy.json →
  categories_dropped_empty` (such a category is also never a placement candidate, having no members) **[flag F23]**.
  Degenerate paths: definitions with no final answer → step 2 skipped, empty outputs, bin state `labelled` (the
  existing non-fatal path); every assignment item `no_answer` → all topics `unplaced`.
* **Record per topic** in `topic_categories.csv`: `assignment_item` (the item key), `assignment_round_iterations`,
  `assignment_agreement_mode`, `assignment_contested` (A's and B's round-1 choices differed), `assignment_needs_review`
  (item not agreed **and** contested) next to `category_id`, `category_assignment`.

#### 2.3 Prompt v1.2 (S3b texts and the revision turn's answer key; S3a texts identical to v1.1) **[decided: approved as drafted]**

```
SYSTEM — S3b step 1 (definitions)
You are {agent_name}, a discourse analyst studying posts from the football community r/soccer.
{lens}
Build a discourse taxonomy from the items: define its categories. Use the keywords and the labelled sample of each item.
Return JSON only; no prose or markdown.

USER — S3b step 1
Items: the {n_topics} topics found in r/soccer posts between {start} and {end}, one per line:
topic_id | name | emotion_tone | keywords | description
{topic_lines}
Output schema:
{"decision": "revise",
 "categories": [
  {"id": "cat_1", "name": "...", "purpose": "...", "emotion_tone": "...", "description": "...", "inclusion_cues": ["..."]}
 ]}
Rules: keep category names short and non-overlapping; capture both communicative purpose and emotional tone.
There is no minimum or maximum number of categories: define a category for a group of topics only if they are
sufficiently similar; the categories together must cover all topics above. Do not list topic ids; the topics are
assigned to the categories in a second step.

SYSTEM — S3b step 2 (assignment)
You are {agent_name}, a discourse analyst studying posts from the football community r/soccer.
{lens}
Assign each topic to exactly one category of the agreed discourse taxonomy. Use the keywords and the labelled sample
of each item. Return JSON only; no prose or markdown.

USER — S3b step 2
Categories (the agreed taxonomy of r/soccer posts between {start} and {end}), one per line:
category_id | name | emotion_tone | inclusion_cues | description
{category_lines}
Topics to assign ({n_batch} of the bin's {n_topics} topics), one per line:
topic_id | name | emotion_tone | keywords | description
{topic_lines}
Output schema:
{"decision": "revise",
 "assignments": [
  {"topic_id": <topic_id>, "category_id": "<category_id>"}
 ]}
Rules: list every topic_id above exactly once, each with the single best-fitting category_id; use only the category
ids listed; if no category fits well, choose the closest one.

USER — revision turn (all stages; the items block is repeated above; no chat history)
Your previous answer:
{mine}
Your peer's latest answer:
{peer}
If the peer's answer is at least as good as yours, return {"decision": "accept"} and nothing else.
Otherwise return {"decision": "revise", "{answer_key}": [...]} with a revised answer that improves agreement with the
peer while preserving quality. Same schema. JSON only.
   ({answer_key} = "categories" for S3a and step 1, "assignments" for step 2 — the only change to the v1.1 revision text)

USER — repair turn: unchanged ("Your previous answer could not be used: {error} …").
```

`PROMPT_VERSION` becomes `"1.2"` for every record from now on; the S3a text is byte-identical to v1.1, so the bin-1
labels stay comparable with the dry run **[flag F25]**. Each stage now logs its own prompt texts into its section of
`config.json` (today only `--stage topics` logs them, so a bin whose S2 predates a prompt change would show stale
texts). The v1.1 single-prompt S3b texts stay in `prompts.py` for the `partition` protocol.

#### 2.4 Settings (`config.py`)

```
TAXONOMY_PROTOCOL = "two_step"     # D5, 2026-10-24; "partition" = the v1.1 single-prompt protocol with chunking (kept for comparison / replay)
TAXONOMY_ASSIGN_BATCH = 1          # topics per assignment prompt (1 = one debate per topic; decided 2026-10-24)
LLM_MAX_NEW_TOKENS_DEFINE = 4000
LLM_MAX_NEW_TOKENS_ASSIGN = 800
```
`TAXONOMY_MAX_ITEMS`, `TAXONOMY_CHUNK_SIZE`, `LLM_MAX_NEW_TOKENS_TAXONOMY` remain for the `partition` protocol
**[flag F26]**. CLI: `--taxonomy-protocol two_step|partition` and `--assign-batch N` override the config; both are
recorded in `config.json`.

#### 2.5 Outputs and schema changes

| File | Change |
|---|---|
| `bin-N/categories.csv` | columns unchanged; the debate columns describe the definitions debate |
| `bin-N/topic_categories.csv` | + `assignment_item`, `assignment_round_iterations`, `assignment_agreement_mode`, `assignment_contested`, `assignment_needs_review` |
| `bin-N/taxonomy.json` | + `protocol`, `categories_defined` (agreed definitions, canonical ids), `definitions_repair`, `categories_dropped_empty`, `assignment_items` (topic ids, mode, rounds, contested per item), `debates = {definitions, assign_001 …}`; `n_answers_salvaged` counts both steps |
| `bin-N/taxonomy_definitions.json` | new checkpoint after step 1 (deleted by `--force`) |
| `bin-N/topic_category_overrides.csv` | **new review file** (F28): `topic_id, name, category_id_agent, category_name, final_category_id, notes`, written by the taxonomy stage next to `category_overrides.csv`, so its ids always match the taxonomy it references |
| `bin-N/category_overrides.csv` | unchanged; on `--force` the template is regenerated when nothing is filled in, otherwise kept with a warning (today a stale 12-topic template would survive the full run) |
| `bin-N/config.json → taxonomy` | + `protocol`, `assign_batch`, `n_categories_defined`, `n_items`, `assignment_agreement_modes`, token budgets, the stage's prompt texts |
| `results/analysis/topic_labels_all_bins.csv` | + `assignment_round_iterations`, `assignment_agreement_mode`, `assignment_contested`, `assignment_needs_review` (after `category_assignment`) **[flag F22]** |
| `results/analysis/categories_all_bins.csv` | unchanged |
| `post_variables.csv`, master / yearly files, codebook §4.9 variables, `07`/`08` R scripts | **unchanged** |

**F28, how the topic-level override is applied in S5** (`tagging.load_bin_outputs`), in this order: (a)
`category_id_agent` = the agents' value from `topic_categories.csv`, untouched; (b) `final_category_id` applies to the
surviving topic after `merge_into` (an entry on a merged-away topic is ignored with a warning; an entry on a merge
target moves all merged posts); (c) then `category_overrides.csv` merges apply (chains resolved); (d) topic lists and
sizes are recomputed; a category emptied by moves stays in the table with `n_topics = 0`, so validation A6 still holds.
An unknown target fails the S5 check. A `read_final_categories()` helper keeps `apply_topic_overrides()`'s signature.

### 3. Code changes

| File | What changes |
|---|---|
| `code/topicmodel/prompts.py` | `PROMPT_VERSION = "1.2"`; `SYSTEM_DEFINE`, `USER_DEFINE`, `SYSTEM_ASSIGN`, `USER_ASSIGN`, `USER_REVISION` with `{answer_key}`; `category_line()`, `render_define_items()`, `render_assign_items()`, `render_revision(…, answer_key)`; `system_prompt()` becomes a dict lookup over `label / taxonomy / define / assign` that raises on an unknown stage; v1.1 S3b texts kept; `all_prompts()` extended |
| `code/topicmodel/agents.py` | `Debate.__init__(…, max_new_tokens=None, salvage=salvage_json, answer_key="categories")` (defaults keep S3a and the partition path byte-identical); `validate_definitions` (canonical ids, soft errors), `cleanup_definitions`, `definitions_identical`, `validate_assignments` (lenient references, mapping form, cleaned-with-error contract), `assignments_identical`, `salvage_assignments`, `salvage_truncated_definitions`; `run_taxonomy_two_step()` (definitions debate → checkpoint → items → assignment debate → merge → `repair_partition` → writers); the tail of `run_taxonomy()` becomes `_write_taxonomy_outputs()` shared by both protocols, now also writing `topic_category_overrides.csv` and regenerating an unfilled `category_overrides.csv` on `--force`; `run_taxonomy(…, protocol, assign_batch)` dispatches; seed prefixes `f"{bin}-def"` / `f"{bin}-assign"`; `read_final_categories()` |
| `code/topicmodel/tagging.py` | `LABEL_COLUMNS` + the four assignment columns (joined from `topic_categories.csv`, empty for `partition` outputs); F28 applied in the order of §2.5 |
| `code/06-analysis-pipeline.py` | `--taxonomy-protocol`, `--assign-batch`; `stage_taxonomy` passes them, records `protocol`, `n_items`, `assignment_agreement_modes`, `n_categories_defined` and the stage's prompts; the skip message names the existing result's protocol; `stage_label` records its prompt texts; header docstring |
| `code/topicmodel/config.py` | §2.4 |
| `code/tests/test_pipeline_units.py` | `test_taxonomy_two_step` with a `StubLLM` answer function parsing `^(\d+) \| ` and `^(cat_\d+) \| ` from the prompt; covers: definitions A ≠ B then `accept` in round 2; duplicate names (soft path → cleanup); non-canonical ids (`"1"`, `"C1"`) canonicalised before step 2; definitions `cap` (every `categories.csv` row `needs_review`); definitions `no_answer` for both (step 2 skipped, non-fatal); an unused category → `categories_dropped_empty`; assignment answers in list form, mapping form, by name, with `null`, with a duplicated topic, with an id of another item; both round-1 answers with-error and identical → `identical` + `repaired`; an item `no_answer`; batch size 1 and 3; `assignment_contested`; the revision prompt of the assign stage contains `"assignments"` and not `"categories"`; `test_salvage_assignments` (`}], [{` and a missing `]}`); `test_salvage_truncated_definitions`; `test_collect_partial` with `final_category_id` on a merged-away topic, on a merge target, and naming a merged-away category; the existing partition tests call `run_taxonomy(…, protocol="partition")`; `test_taxonomy_empty_is_not_fatal` for both protocols |
| `code/07-tag-corpus.R`, `code/08-validate-analysis.R` | no change |

Reused unchanged: `Debate.run`, `extract_json`, `salvage_json`, `_cues`, `_text_field`, `_coerce_topic_id`,
`partition_problems`, `repair_partition`, `_topic_lines`, `tone_category`, the override readers, `render_report`.

### 4. Colab notebook (`code/06b-analysis-pipeline-colab.ipynb`) and `requirements-colab.txt`

Orchestration only, as before; edited with a small Python script so the cell ids survive. Changes:

1. **Header cell:** two-step taxonomy; the stop point of this session after the full bin 1; the Drive layout and the
   session log named.
2. **Cell 0.2:** records the commit the run used in the session log.
3. **Cell 0.3:** `pip check`, GPU and version lines also written to the session log.
4. **Cell 0.4 `stage()`:** runs the child **unbuffered** (`python -u`, also under `-m cuml.accel`), stderr merged,
   progress bars off (`HF_HUB_DISABLE_PROGRESS_BARS=1`, `TQDM_DISABLE=1`), every line echoed to the notebook and
   appended with a flush to `MyDrive/fifa-reddit-communities/logs/colab_<date>.log`; a non-zero exit still raises.
   `unit_checks()` output likewise.
5. **Pass 1 taxonomy check cell and cell 3.1:** print the definitions debate (labelled as such), the assignment modes
   over the items, `n_repaired`, `unplaced`, `n_answers_salvaged`, categories defined / final; the `S3B_DECISION`
   guard is replaced by the recorded decision (D5 and the commit sha); a markdown cell after 3.1 marks **the stop point
   of this session** with the hand-back step.
6. **Hand-back cell (new):** zips `results/analysis/bin-1/` without `model/` (regenerable) — `log.txt`, `config.json`,
   `labels/`, the CSVs, `taxonomy*.json`, `outlier_calibration/` included — plus `data/analysis/*.csv|json` into
   `MyDrive/fifa-reddit-communities/handback/bin-1_<date>.zip`.
7. The comparison cell (3.2) loses its Qwen3-4B branch (F9 dropped by your decision); the seeded-sampling branch
   stays optional. Cells 3.3–5 (large-bin gate, `T`, bins 2–9, sentiment/collect) stay for the next session.

`requirements-colab.txt` unchanged. Risks verifiable only at run time, each caught by an assertion in cell 0.3 or by
`config.json → backend` **[flag F27]**: `hdbscan==0.8.44` builds from source under `--no-deps`; `sentence-transformers
6.1.0` against Colab's `transformers`; cuML's UMAP with the cosine metric (a silent CPU fall-back shows as backend
`cpu`). The runbook's rule applies: fix the one package named by `pip check`, never a plain `pip install -r`.

### 5. Execution steps (in order)

**A. Housekeeping (local, first commit).** Unstage the 686 collector JSON files and
`reddit-soccer-collector/.github/workflows/trigger.yml` (left untracked as before session 05; nothing deleted)
**[default]**. Write `docs/session-06_2026-10-24/00_approvals.md` (D5 with the alternatives declined, N11/N12 confirmed,
test scope, your planning-round answers (prompt v1.2, one topic per item, the Colab loop, no 4B comparison), the flags
F22–F30 with what the plan does); mark N11/N12 as confirmed in the session-05 approvals. Complete `docs/session-05_2026-10-23/02_dry-run-log.md` §3.8 with the 92-topic numbers, stated
as in §1 (10-category answers, duplicated ids, collapse under the repair rule). Commit the pending session-05 changes
(N12 code + test, runbook §6, notebook guard) with this: *"Session 06: approvals, N11/N12 confirmed, 92-topic dry-run
results"*.

**B. Implement D5 (§3) and the notebook (§4).** Unit checks must pass (expected ≈ 19 checks).

**C. Local dry run of the two-step S3b with the real model** (background, budget ≈ 1 h on the Mac: prefill of the
long definitions prompts is slow on MPS): copy `results/dry-runs/bin1-full92` → `results/dry-runs/bin1-twostep`, run
`--stage taxonomy --bin 1 --force --out-root results/dry-runs/bin1-twostep` (Qwen3-1.7B, MPS), then `collect
--allow-partial` and `report`. Acceptance: definitions agreed or `needs_review` with both answers kept; `unplaced = 0`;
`n_repaired` well below the 23 of the single-prompt run; readable categories in `report.html`. If the 1.7B model still
fails here, I stop and report before Colab (the protocol or the batch setting would then need your decision).

**D. Second commit and push:** *"S3b two-step protocol (prompt v1.2), Colab notebook for the bin-1 test batch"* with
design doc v1.2, `docs/session-06_2026-10-24/01_implementation-notes.md`, workflows, README, codebook wording (§7
below). The notebook clones from GitHub, so nothing runs on Colab before this push.

**E. The Colab test batch** (you run the cells in VS Code on the Colab kernel; A100 40 GB, high RAM):

| Step | Cells | What must come out (I check it in the log / files) |
|---|---|---|
| 0 | `data/yearly datasets/` is already on Drive (your answer); cell 0.1 asserts the 17 files | — |
| 1 | 0.1–0.4 | Drive mounted, 17 files; HEAD = the pushed sha; `pip check: OK`; GPU name; pins honoured; unit checks all PASS |
| 2 | 1 | nine bin counts = design table, Σ 1,302,021 |
| 3 | Pass 1 (2.1–2.4) | `backend = cuml`, guard converged, 50–100 topics, raw outliers ≈ 30–45 %; 12 labels, `truncated = 0`; two-step taxonomy over 12 topics; sentiment 2010; `bins_complete = [1]`, no errors |
| 4 | 3.1 | all bin-1 topics labelled, `needs_review ≤ 15 %`; two-step taxonomy over all topics: `unplaced = 0`, `n_repaired` small, categories readable; `report.html` shows the 2010 themes |
| 5 | hand-back cell | zip on Drive (no comparison runs in this batch) |

Loop on failure: you paste the failing cell's output, or I read the Drive log through the connector; I fix, commit,
push; you re-run cell 0.2 (pull) and the failing cell. Each fix is its own small commit, listed in the run log.

**F. Hand-back and record.** The zip (or the connector) → `results/analysis/bin-1/` and `data/analysis/` in the
repository; I compare with the dry run (topic count, outlier share, label modes, taxonomy) and write
`docs/session-06_2026-10-24/02_colab-run-log.md` (every number, versions, backend, timings, fixes),
`03_runbook-session-07.md` (gates 2–5, `T`, bins 2–9, S6 — the session-05 runbook carried forward for D5), the design
§14 rows, `memory/2026-10-24_session-06.md`, `memory/workflows.md`, the auto-memory status file. Final commit and push
with `results/analysis/bin-1/` (tracked) included.

### 6. Verification

* `python code/tests/test_pipeline_units.py` → all PASS (new: two-step protocol incl. the paths of §3, assignment and
  truncated-definitions salvage, the revision-key regression check, F28 in collect).
* Step C's local run: `taxonomy.json` has `protocol = two_step`, `debates.definitions` + `debates.assign_*`,
  `unplaced = 0`; `taxonomy_definitions.json` exists; `topic_categories.csv` has the five new columns;
  `topic_category_overrides.csv` written; `collect --allow-partial` passes and `topic_labels_all_bins.csv` has 31
  columns; `report.html` renders.
* Replay check: `--stage taxonomy --bin 1 --taxonomy-protocol partition --out-root <copy>` still runs the v1.1 path
  (unit-tested; not run with the model unless you ask).
* Colab: the table in step E; `config.json → topics.backend.backend == "cuml"`; `label.llm_stats.truncated == 0`;
  `taxonomy.protocol == "two_step"`; `post_variables.json → bins_complete == [1]`.
* Nothing is written under `data/` except `data/analysis/` (the Colab copies), and `results/analysis/bin-1/` comes
  from the GPU run only (local dry runs stay under `results/dry-runs/`).

### 7. Documents touched (consistency checklist)

Design v1.2: status line; §3 table (S3b outputs incl. `topic_categories.csv`, `topic_category_overrides.csv`); §7.1
(budgets); §7.4 rewritten; §7.5 item 1 (S3b identity rules); §7.6 prompt v1.2; §7.7 outputs; §12 gate 2 and item 6
(wording "partition valid without repair" → "topics assigned without repair"); §13 F22–F30; §14 rows of 2026-10-24;
§15 row "Prompt v1.1" → v1.2; §16 agents.py row; §17 notes. Codebook (stays 3.0 draft): §1 rows for
`topic_labels_all_bins.csv` / `categories_all_bins.csv`, §4.9 paragraph (label-table columns; `category_id` "after
co-author overrides" now includes topic-level moves), §8.1 sentence (two steps). `00_approvals.md` of session 05
(N11/N12 confirmed, dated), `02_dry-run-log.md` §3.8, `03_runbook-session-06.md` §6 (dated pointer to D5). Module
docstrings of `prompts.py`, `agents.py`, the header of `06-analysis-pipeline.py`. `memory/workflows.md` §7 (protocol
switch, `--assign-batch`, the new review file), `README.md` §4, `docs/README.md` row for session 06, the session-06
files of §5, the auto-memory status file.

### 8. Flags for your review (continuing design §13)

| # | Issue | What the plan does |
|---|---|---|
| F22 | Two debates per bin instead of one; under two lenses the definitions are mostly adopted through `accept` (Analyst B's wording wins, as in S3a) | `round_iterations` in `categories.csv` = definitions debate; per-topic assignment rounds, modes and `contested` in `topic_categories.csv` and the label table; design §7.5/§7.7 and the codebook paragraph say which debate each number describes |
| F23 | A defined category to which no topic is assigned has no posts and is never a placement candidate for missing topics | dropped from `categories.csv`, listed in `taxonomy.json → categories_dropped_empty`; ids keep gaps. Alternative: keep it with `n_topics = 0` |
| F24 | Assignment can be batched; batches would be a procedural artefact | one topic per item (decided); batching remains a config value, the item recorded per topic |
| F25 | Prompt version 1.2 applies to all records although only the S3b texts and the revision turn's answer key changed | S3a text byte-identical to v1.1; every stage logs its own texts into `config.json` |
| F26 | Keeping the v1.1 partition protocol in the code adds a switch nobody may use | kept: tested, replays the dry run, allows a comparison; one config value |
| F27 | The Colab environment is verified only at run time | assertions in cell 0.3 and `backend` in `config.json`; the fix-commit-push loop of §5 E |
| F28 | Co-authors could not move a single topic to another category; with assignment as its own step this is the likeliest correction | `topic_category_overrides.csv` written by the taxonomy stage (ids always current), applied in S5 in the order of §2.5; `category_id_agent` keeps the agents' value; a re-run of the taxonomy regenerates the template |
| F29 | Transcripts store every prompt in full; two debates per bin ≈ 1 MB of `taxonomy.json` per bin in git | accepted (≈ 9 MB for the corpus; complete transcripts are the audit trail) |
| F30 | Assignment references are resolved leniently (names, bare numbers) and `null` counts as missing | the resolution is recorded per answer (`resolved_from`) so the audit trail shows it; strictness is one function |

### 9. Defaults accepted with this plan unless you object

* Token budgets 4,000 / 800; soft-error cleanup of definitions; empty categories dropped (F23); F28 as a separate
  review file; protocol switch kept (F26); lenient references (F30); transcripts complete (F29).
* Collector files unstaged and left untracked (nothing deleted). Pending session-05 changes committed in step A.
* Session record under `docs/session-06_2026-10-24/`; design document → v1.2; codebook stays 3.0 draft with wording
  changes only; `prompt_version` 1.2.
* Runtime A100 40 GB, high RAM; Drive layout as documented; nothing analytical in the notebook.

### 10. Independent review (2026-10-24)

27 findings, all folded in. The three that changed the design: the revision-turn prompt named `categories`, which
would have made every contested assignment fail in round 2 (→ `{answer_key}`); the assignment salvage would have been
unreachable (→ per-stage salvage hook in `Debate`); category ids had to be canonicalised before step 2 and never
renumbered afterwards. Also from the review: soft validation of definitions, lenient assignment references, the
validator contract that keeps with-error answers, the checkpoint between the steps, the separate topic-level override
file and its S5 order, unbuffered logging in the notebook, per-stage prompt logging, the regeneration rule for
`category_overrides.csv`, the one-topic-per-item default put to you as [Q2], the corrected reading of the 92-topic
evidence, and the longer budget for step C.

### 11. Not decided now (gates, next session, with the Colab evidence)

Outlier threshold `T` (bin 1 + a large bin), greedy vs seeded sampling, `min_samples` for large bins, the D1
floor-or-leaf question if a bin collapses, the D2 name+tone refinement (not needed at 2.2 % `needs_review`), the
seeded-sampling comparison (gate 4). The Qwen3-4B comparison is no longer open: dropped by your decision of
2026-10-24 (`LLM_MODEL_LARGE` stays in `config.py` as an unused setting with a dated note).
