# Analytical pipeline design — period-binned topic modelling, two-agent labelling and sentiment (v1.0)

**Session 04, 2026-10-22/23 · FIFA Communities chapter, McGill University.**
Decisions: Alan Nemirovski. Drafting and technical design: Claude. Brief: `00_brief.md`. Template method: the
ICWSM submission *AI-Generated Sexual Content on Reddit* (BERTopic → Qwen3-1.7B two-analyst multi-turn labels →
manual merge → aggregate reporting), used as a guide, not a recipe.

**Status.** Design approved on 2026-10-22 (decisions D1–D4, §14). Implementation and the test run on bin 1 follow
in session 05 (§16). This document supersedes the *run structure* of `../session-02_2026-10-13/02_topic-modelling-design.md`
(calendar-year, ±2-month window and era models); the modelling settings approved there are kept unless stated.

---

## 0. How to use this document

* **To understand the method:** read §1–§3, then the stage sections §4–§9. Every setting has a value and a reason;
  everything configurable lives in one file (`code/topicmodel/config.py`), and every run writes its effective settings,
  prompts, software versions and compute backend to a `config.json` next to its outputs.
* **To rebuild a similar pipeline** (another subreddit, other periods): keep the stage contracts of §3 (what each stage
  reads and writes), change the inputs in §2, the bin table in §4 and the settings in §6–§8. The code layout in §16
  shows where each piece lives.
* **To review the design:** §13 lists every flagged issue and refinement (F1–F21) with what was decided; §14 is the
  decisions log; §15 maps each element of the brief to the design choice that answers it.
* **To find outputs:** §9.3 (file layout) and the codebook §4.9 (variable definitions).

## 1. Purpose and scope

The pipeline turns the cleaned corpus (codebook v2.1: 1,302,021 r/soccer posts, 1 Jan 2010 – 31 Jul 2026, posts only,
88 % title-only) into the inputs of the chapter's discourse analysis:

1. **Per post** (appended to the master and yearly CSVs): the analysis period (`period_bin`), the BERTopic topic
   (`topic_id`) and the discourse category (`category_id`) of the post, and its VADER sentiment (`post_sentiment`).
2. **Per topic** (one aggregated CSV, the brief's "separate CSV"): the two-agent label of every topic cluster
   (name, purpose, emotional tone, description, inclusion cues), how many debate rounds the agents needed
   (`round_iterations`), and the period bin.
3. **Per period bin**: a discourse taxonomy (categories grouping similar topics), produced by the same two agents.

Out of scope here (hooks left): the cross-period alignment of categories (a manual crosswalk, §13 F4), a political
content layer (session 03 question, F15), and sentiment measures beyond VADER (F10).

## 2. Inputs and the unit of analysis

* **Files.** `data/yearly datasets/soccer.data.cleaned-YYYY.csv` (17 files, v2.1), read with every column as text
  (codebook §5). The master `data/soccer_data.clean.csv` is only touched in the final write-back (§9).
* **Unit.** One post = one document. The document text is the cleaned title, a newline, and the cleaned body
  truncated to 1,000 characters (`topicmodel.data.clean_text`: strips Markdown link syntax, bare URLs, Reddit sprite
  codes, HTML entities and extra whitespace; it never changes words, case or spelling). Posts with fewer than three
  word tokens in title + body are kept in every table but not modelled (`topic_assignment = not_modelled`; ≤ 1–2 % per
  bin).
* **What stays in.** Human-posted match and daily threads (decision D4 of session 01; `post_type` carried as a QA
  covariate), posts whose `lang` is not `en` (the S7 filter already removed confidently non-English titles; the
  remaining short, name-heavy titles are modelled like all others and bilingual topics are visible through a per-topic
  `share_lang_en`, F14).

## 3. Pipeline overview and stage contracts

```
yearly CSVs ─► S0 period bins ─► S1 documents + sentence embeddings (cached per year)
            ─► S2 BERTopic, one model per bin (9 models): topic_id per post, keywords, sample posts
            ─► S3a two-agent cluster labelling (per topic)      ─► co-author review / topic_overrides.csv
            ─► S3b two-agent discourse taxonomy (per bin)       ─► co-author review / category_overrides.csv
            ─► S4 VADER sentiment per post
            ─► S5 collect: post_variables.csv (every post) + topic_labels_all_bins.csv + categories_all_bins.csv
            ─► S6 write-back (local, R): master + yearly files gain 8 columns ─► codebook v3.0 ─► validation
```

| Stage | Reads | Writes | Where it runs |
|---|---|---|---|
| S0 bins | yearly CSVs | `data/analysis/period_bins.csv` (id, period_bin, period_type) + bin summary | Colab or local |
| S1 embed | yearly CSVs | `results/topic-models/embeddings/<model>/<year>.npz` | Colab GPU |
| S2 topics | S0, S1 | `results/analysis/bin-N/` (§6.4) | Colab GPU (cuML) |
| S3a label | S2 | `bin-N/labels/topic_<id>.json`, `labels.jsonl`, `topics_labelled.csv` | Colab GPU |
| S3b taxonomy | S3a (+ overrides) | `bin-N/taxonomy.json`, `categories.csv` | Colab GPU |
| S4 sentiment | yearly CSVs | `data/analysis/sentiment.csv` (id, post_sentiment, post_sentiment_full, label) | CPU |
| S5 collect | S0–S4 (+ overrides) | `data/analysis/post_variables.csv` (+ `.json` sidecar), `results/analysis/topic_labels_all_bins.csv`, `categories_all_bins.csv` | any |
| S6 write-back | S5 | `data/soccer_data.clean.csv`, 17 yearly files (v3.0), validation report | local (R) |

Dependency chain to respect: S3b consumes S3a's final names, so any S3a override requires re-running S3b; overrides
are applied before S5; S6 runs only after the co-author review of S3a and S3b. One command-line program runs the
stages (`python code/06-analysis-pipeline.py --stage {bins,embed,topics,label,taxonomy,sentiment,collect} --bin N`);
every stage is idempotent (skips when its output exists) and resumable (the agent stage appends to `labels.jsonl`).
The Colab notebook only orchestrates these commands. S6 is an R script using the same `readr::write_csv()` call as the
cleaning pipeline, so the data files keep one writer and one format.

## 4. Stage S0 — period bins

| `period_bin` | Start (UTC date, inclusive) | End (inclusive) | `period_type` | World Cup (kick-off → final) |
|---|---|---|---|---|
| 1 | 2010-01-01 (corpus start) | 2010-09-30 | tournament | 11 Jun → 11 Jul 2010 (South Africa) |
| 2 | 2010-10-01 | 2014-01-31 | interim | — |
| 3 | 2014-02-01 | 2014-09-30 | tournament | 12 Jun → 13 Jul 2014 (Brazil) |
| 4 | 2014-10-01 | 2018-01-31 | interim | — |
| 5 | 2018-02-01 | 2018-09-30 | tournament | 14 Jun → 15 Jul 2018 (Russia) |
| 6 | 2018-10-01 | 2022-06-30 | interim | — |
| 7 | 2022-07-01 | 2023-02-28 | tournament | 20 Nov → 18 Dec 2022 (Qatar) |
| 8 | 2023-03-01 | 2026-01-31 | interim | — |
| 9 | 2026-02-01 | 2026-07-31 (corpus end) | tournament | 11 Jun → 19 Jul 2026 (Canada/Mexico/USA) |

**Rule.** A post's bin is determined by the UTC calendar date of `created_datetime`. The bins are contiguous and cover
the whole corpus, so every post receives exactly one bin; the stage checks that the bin sizes sum to the corpus size.
`period_type` is derivable from `period_bin` and is added as a convenience column (drop it if unwanted). Bin 1 starts
on 1 January rather than 1 February because the corpus starts then; bin 9 ends 12 days after the 2026 final because
the corpus ends on 31 July 2026.

**Bin sizes** (clean posts, v2.1; computed 2026-10-22 from the yearly files; Σ = 1,302,021):

| bin | posts | self posts | with body | match threads | `lang = en` | titles < 3 tokens* | min topic size at 0.25 % (§6.2) |
|---|---|---|---|---|---|---|---|
| 1 | 10,002 | 20.0 % | 17.0 % | 1.0 % | 93.1 % | 237 | 25 |
| 2 | 186,988 | 22.5 % | 15.9 % | 2.3 % | 91.9 % | 4,839 | 467 |
| 3 | 83,147 | 24.6 % | 13.2 % | 2.0 % | 92.6 % | 1,732 | 208 |
| 4 | 331,216 | 15.7 % | 11.2 % | 2.9 % | 88.2 % | 3,908 | 828 |
| 5 | 88,648 | 5.0 % | 5.0 % | 2.0 % | 85.4 % | 818 | 222 |
| 6 | 305,653 | 7.2 % | 7.2 % | 4.8 % | 78.0 % | 1,780 | 764 |
| 7 | 41,438 | 6.6 % | 6.9 % | 5.4 % | 69.3 % | 45 | 104 |
| 8 | 217,303 | 7.3 % | 10.7 % | 5.2 % | 71.9 % | 719 | 543 |
| 9 | 37,626 | 7.4 % | 16.9 % | 3.8 % | 77.6 % | 149 | 94 |

\* Title-only count; the pipeline's not-modelled rule counts title + body tokens, so this is an upper bound. The falling
`lang = en` share from bin 6 on reflects the conservative language detector on short, name-heavy titles (codebook
§7.6), not a language shift. The bins differ in size by a factor of 33; the consequences for topic granularity are in
§6.2 and F5.

## 5. Stage S1 — documents and embeddings

* **Embedding model:** `sentence-transformers/all-mpnet-base-v2` (768 dimensions; Reimers & Gurevych 2019), the
  higher-quality model approved in session 02 for GPU runs (`all-MiniLM-L6-v2` was used for the local 2010 test).
  `max_seq_length = 256` tokens (title + 1,000 characters of body fit), embeddings L2-normalised, batch size 256.
* **Cache:** one `.npz` per calendar year (ids + float32 matrix; ≈ 4 GB for the corpus) under
  `results/topic-models/embeddings/<model>/`, stored on Google Drive. Bin runs take their rows from the year caches
  (existing `topicmodel.embed.embeddings_for`), so bin 7, which spans two yearly files, and any re-run reuse them, and
  a Colab disconnect costs nothing already embedded.

## 6. Stage S2 — BERTopic, one model per bin

BERTopic (Grootendorst 2022) chains: pre-computed embeddings → UMAP dimensionality reduction (McInnes, Healy & Melville
2018) → HDBSCAN density clustering (Campello, Moulavi & Sander 2013; McInnes, Healy & Astels 2017) → class-based TF-IDF
keywords → keyword re-ranking. No topic count is imposed; the number of topics is data-driven within a guard.

### 6.1 Settings

| Setting | Value | Why |
|---|---|---|
| UMAP | `n_neighbors=15, n_components=5, min_dist=0.0, metric="cosine", random_state=42` | BERTopic defaults, seeded |
| UMAP, how it is run | **fitted once per bin** on the 768-d embeddings; the 5-d result is saved (`umap_5d.npy`) and wrapped in a `FrozenUMAP` object (`fit` is a no-op, `transform` returns the saved array) that BERTopic uses as its `umap_model`; the 768-d embeddings are still what BERTopic receives | BERTopic refits UMAP on every `fit_transform`; freezing it makes the topic-count guard cheap and keeps topic embeddings in the 768-d space, which the keyword re-ranking and the outlier step need |
| HDBSCAN `min_cluster_size` | start at `max(25, round(0.0025 × n_modelled))` (0.25 % of the bin, floor 25), then the **topic-count guard** (§6.2) | a relative rule keeps topic *counts* comparable across bins that differ 33× in size |
| HDBSCAN `min_samples` | 10 | the setting approved and tested in session 02; revisited only if the large-bin gate (§12) shows > 45 % raw outliers |
| HDBSCAN other | `metric="euclidean"`, `cluster_selection_method="eom"`, `prediction_data=True` | standard |
| Keywords: vectorizer | `CountVectorizer(ngram_range=(1,2), min_df=2, stop_words=English + EXTRA_STOPWORDS)`; `EXTRA_STOPWORDS` = the session-02 scaffolding list (`thread`, `match thread`, `post match`, `amp`, `http`, `reddit`, `imgur`, `youtube`, `twitter`, `streamable`, `edit`, `deleted`, `removed`, …) **plus** `vs`, `soccer` (F11) | stop-words affect only the keyword descriptions, never the documents or embeddings |
| Keywords: weighting | c-TF-IDF with `reduce_frequent_words=True`; `top_n_words=15`; representation chain KeyBERTInspired (30 candidates) → MaximalMarginalRelevance (`diversity=0.3`, 15 words) | readable, non-redundant keywords |
| Keywords: which posts | computed from **cluster members only** (posts assigned by HDBSCAN), not from reassigned outliers | the description stays pure; reassigned posts count towards prevalence only |
| Outliers | §6.3 | decision D3 |
| Representative posts | 5 nearest to the topic's c-TF-IDF representation (BERTopic) + 35 seeded random cluster members → the 40 posts the agents see | the paper used a random sample; the five anchors stabilise naming |
| Topics over time | weekly bins for tournament periods, monthly for interim periods (`topics_over_time`, no global/evolution tuning) | matches the period length |
| GPU | on Colab the stage runs under `python -m cuml.accel …` (NVIDIA RAPIDS cuML is pre-installed on Colab; its accelerator runs the unchanged `umap-learn`/`hdbscan` calls on the GPU); the program detects which backend actually ran and writes `backend: "cuml" | "cpu"` into `config.json`; the CPU path is the same code | the largest bins (≈ 330k × 768) take ≈ 1 h on CPU and minutes on an A100 |

### 6.2 Topic-count guard

Target: **50–100 raw topics per bin** (before any outlier handling). Rationale: the taxonomy stage (§7.4) feeds all of
a bin's topics to a 1.7-billion-parameter model in one prompt, which is reliable up to about 100 compact items; and a
comparable number of topics per bin makes the per-bin taxonomies comparable in resolution.

Procedure: fit HDBSCAN on the frozen 5-d array with the starting `min_cluster_size`; record (`min_cluster_size`,
`n_topics`); if `n_topics` is outside 50–100, bisect between the bracketing values already tried (initially ×1.5 or
÷1.5), at most four HDBSCAN fits (seconds each on the frozen array); keep the value whose count is nearest to the
range and flag the bin (`guard_converged = False`) if the range was not reached. All trials are logged in
`config.json`. The guard is disabled for debugging subsamples (`--max-docs`), where small samples cannot reach the
range.

Consequence to keep in mind (F5): with a relative minimum, a topic in bin 1 has ≥ 25 posts and a topic in bin 4 ≥
≈ 800; interim-period topics are broader themes, tournament-period topics are closer to events. Comparable counts,
not comparable granularity.

### 6.3 Outlier handling (decision D3: thresholded reassignment)

HDBSCAN leaves posts that sit in no dense region unassigned (37–40 % in the 2010 test). Reassigning them all (the
session-02 default) gave every post a topic but diluted topic purity; keeping them all as outliers would leave ~40 %
of posts without a topic or category.

* `topic_id_raw` = the HDBSCAN label, saved before anything else (`hdbscan_labels_raw.npy`).
* Reassignment: `reduce_outliers(documents, topics, strategy="embeddings", embeddings=<768-d>, threshold=T)` — an
  outlier joins the topic whose embedding (mean of its members) is most similar, **only if** the cosine similarity is
  ≥ `T`; otherwise it stays `-1`. The similarity of every reassigned post is stored.
* Choosing `T`: on bin 1 **and** one large interim bin, take the distribution of outlier-to-nearest-topic similarity,
  evaluate candidate thresholds at its quantiles (e.g. 50th, 65th, 80th), and have a co-author judge a 100-post sample
  per candidate for whether the post belongs to the topic; the chosen `T` (one value, or one per bin if the
  distributions differ) is recorded in `config.json` and in §14.
* `topic_assignment ∈ {cluster, reassigned, outlier, not_modelled}` records how each post got its `topic_id`.
* Topic sizes are reported both ways: `n_cluster` (HDBSCAN members) and `n_total` (after reassignment); shares and
  prevalence over time use `n_total`.

### 6.4 Outputs per bin (`results/analysis/bin-N/`)

| File | Content |
|---|---|
| `umap_5d.npy`, `hdbscan_labels_raw.npy` | the persisted intermediate arrays (replay, §11) |
| `documents.csv` | `id, created_datetime, period_bin, post_type, lang, topic_id_raw, topic_id, topic_assignment, similarity` (no text; join to the corpus by `id`) |
| `topics.csv` | `topic_id, n_cluster, n_total, share, keywords, representative_ids, representative_titles, share_match_thread, share_lang_en, mean_comments` |
| `topics_over_time.csv` | `topic_id, period, n_docs, keywords` (weekly / monthly) |
| `model/` | the BERTopic model in safetensors format (topic embeddings, c-TF-IDF). UMAP and HDBSCAN are **not** saved in this format; the raw labels come from the `.npy` files |
| `config.json`, `log.txt`, `report.html` | effective settings, guard trials, backend, versions; run log; charts and topic table for review |

## 7. Stage S3 — two-agent labelling with Qwen3-1.7B

### 7.1 Model and inference

* **Model:** `Qwen/Qwen3-1.7B` (Qwen Team 2025; Apache-2.0), Hugging Face revision
  `70d244cc86ccca08cf5af4e1e306ecf908b1ad5e` (the snapshot already cached locally), 40,960 positions (32k tokens used
  as the working limit). bfloat16 on Ampere-class GPUs (A100, L4); float16 on a T4. Qwen3's "thinking" mode is
  disabled (`enable_thinking=False`) for speed and clean JSON.
* **Decoding:** greedy (temperature 0) by default, for reproducibility. A seeded-sampling alternative (temperature
  0.7, per-call seed = first 8 hex digits of SHA-256 of `"{bin}-{topic}-{agent}-{round}"`, seed logged in the
  transcript) is implemented and compared with greedy decoding on bin 1 (§12, F17).
* **Backend:** Hugging Face `transformers`, with generation **batched across topics** (all pending "round r, Analyst
  A" turns in batches of `LLM_BATCH_SIZE`, batches formed in sorted topic-id order and independent of resume state);
  vLLM is an optional accelerator with the same prompts. `max_new_tokens` 600 (S3a) / 4,000 (S3b).
* **Parsing and validation:** the first JSON object is extracted with `json.JSONDecoder().raw_decode` from the first
  `{`; the object is validated against the stage schema (§7.3/§7.4). An invalid answer triggers one repair turn that
  quotes the validation error; a second failure counts as "no answer" for that turn (S3a: the topic is flagged
  `needs_review`; S3b: deterministic repair, §7.4).

### 7.2 Two analysts with different lenses

Under greedy decoding, two agents that receive identical prompts are the same deterministic function: their first
answers are either identical (trivial agreement) or diverge only by chance, and every later accept/revise decision is
fixed by the inputs, so deadlocks persist to the cap. To give the second analyst an independent judgement (the point
of the paper's two-analyst setting), the analysts differ in one instruction only:

* **Analyst A — keyword-first:** "Start from the distinctive keywords, then check them against the posts."
* **Analyst B — post-first:** "Start from reading the posts, then check them against the keywords."

### 7.3 Stage S3a — cluster labelling (one category per topic)

For every topic of a bin, each analyst receives the topic's 15 keywords and up to 40 sample posts (§6.1; title first,
each post truncated to 300 characters) and must describe the cluster as **exactly one** category (`name`, `purpose`,
`emotion_tone`, `description`, `inclusion_cues`). The free-text tone is additionally mapped to a controlled vocabulary
(`TONE_SET`: positive, negative, neutral, mixed, excited, enthusiastic, frustrated, angry, anxious, sad, humorous,
critical, supportive, curious, cautious; stem-based mapper from session 02) as `tone_category`, so tones can be
counted.

### 7.4 Stage S3b — discourse taxonomy (per bin)

Items are the bin's topics, one compact line each — `topic_id | agreed name | emotion_tone | 8 keywords |
one-sentence description` (≈ 60 tokens per topic; ≤ 100 topics ≈ 6k tokens). The analysts group topics into
categories; each category lists the `topics` it contains. The brief's rule applies here: no minimum or maximum number
of categories; topics are grouped only if sufficiently similar.

* **Validity** = the `topics` lists form a partition of the bin's topic ids (every id exactly once). Violations get one
  repair turn quoting the problem; what is still invalid is then repaired deterministically: unknown ids dropped,
  duplicated ids kept in the first category listed, missing topics placed in the category whose member-mean topic
  embedding is most similar (cosine). Each topic records `category_assignment ∈ {agent, repaired, unplaced}`.
* **Stateless turns.** Revision turns resend the items plus the two latest answers and no chat history, so the context
  never grows beyond items + two taxonomies (≈ 14k tokens at 100 topics).
* **Chunking fallback.** If a bin still has more than 100 topics after the guard, the topics are split into
  `ceil(n/80)` chunks by k-means on their topic embeddings, a taxonomy is built per chunk, and a second S3b pass groups
  the chunk categories (items = chunk categories with their union keywords) into the bin's taxonomy; `topics` lists are
  expanded back to topic ids.
* **Comparison** of two S3b answers treats them as partitions (sets of topic-id sets) and matches categories by their
  topic sets; `cat_N` numbering is ignored.

### 7.5 Agreement protocol (decision D2: explicit acceptance, alternating turns, cap 5)

Every answer carries a field `"decision": "accept" | "revise"`.

1. **Round 1.** Both analysts answer the same prompt independently (A's batch, then B's). If the two answers are
   identical after normalisation (lower-case; whitespace and punctuation collapsed; order of cues/topics ignored;
   S3b compared as partitions with matched categories' names and tone categories equal), the topic is agreed
   (`agreement_mode = identical`, `round_iterations = 1`).
2. **Round r ≥ 2 (alternating).** Analyst A receives its own latest answer and B's latest answer and returns either
   `{"decision": "accept"}` — the pipeline then adopts B's already-parsed answer as the agreed output (the model never
   has to copy a long JSON) — or `{"decision": "revise", "categories": […]}`. Then Analyst B does the same against A's
   new answer.
3. **Agreement** = an `accept`, or identical normalised answers after a revision. `round_iterations` = the round in
   which agreement occurred (computed by the pipeline; the brief's `round_iterations` field is therefore produced by
   the pipeline, not by the agent, F1).
4. **Stalemate exit.** An analyst that returns a `revise` identical to its own previous answer ends the debate
   (`agreement_mode = stalemate`).
5. **Cap.** `MAX_ROUNDS = 5`. Unresolved topics are `agreed = False`, `needs_review = True`, keep both final answers,
   and use Analyst A's last answer provisionally; the co-author review that gates S3b and S6 resolves them through
   `topic_overrides.csv` / `category_overrides.csv` (`final_name`, `merge_into`, `notes`), the same mechanism as the
   session-02 merge table. Nothing is resolved silently.

Outcome fields per topic / category: `round_iterations`, `agreed`, `agreement_mode ∈ {identical, accepted_by_A,
accepted_by_B, stalemate, cap}`, `needs_review`, `analyst_a_final`, `analyst_b_final`. Batch definition: round r, all
unresolved topics for Analyst A; topics A accepted leave the set; the remainder for Analyst B.

### 7.6 Prompt v1.1 (final; logged verbatim in every run's `config.json`)

Refinements of the brief's prompt, each flagged in §13: F1 (duplicated `inclusion_cues` removed, `round_iterations`
moved to the pipeline), F2 (S3b schema gains `topics`), F3 (two stages), F16 (`decision` field), F17 (lenses), F20
(stateless S3b turns), F21 ("Use keywords and tagged" completed as "the tagged sample posts"; the r/soccer context
sentence kept from the 2010 test because a small model otherwise produces generic names such as "Football news").

```
SYSTEM — S3a (cluster labelling)
You are {agent_name}, a discourse analyst studying posts from the football community r/soccer.
{lens}
Label one cluster of posts as one category. Return JSON only; no prose or markdown.

  {lens} for Analyst A: Start from the distinctive keywords, then check them against the posts.
  {lens} for Analyst B: Start from reading the posts, then check them against the keywords.

USER — S3a
Keywords: {keywords}
Posts (one per line, title first):
{posts}
Output schema:
{"decision": "revise",
 "categories": [
  {"name": "...", "purpose": "...", "emotion_tone": "...", "description": "...", "inclusion_cues": ["..."]}
 ]}
Rules: exactly one category; name short (2–6 words), specific to football/r/soccer, not generic; capture both the
communicative purpose and the emotional tone; 3–6 inclusion cues.

SYSTEM — S3b (taxonomy)
You are {agent_name}, a discourse analyst studying posts from the football community r/soccer.
{lens}
Build a discourse taxonomy from the items. Use the keywords and the labelled sample of each item.
Return JSON only; no prose or markdown.

USER — S3b
Items: the {n_topics} topics found in r/soccer posts between {start} and {end}, one per line:
topic_id | name | emotion_tone | keywords | description
{topic_lines}
Output schema:
{"decision": "revise",
 "categories": [
  {"id": "cat_1", "name": "...", "purpose": "...", "emotion_tone": "...", "description": "...",
   "inclusion_cues": ["..."], "topics": [topic_id, ...]}
 ]}
Rules: keep category names short and non-overlapping; capture both communicative purpose and emotional tone.
There is no minimum or maximum number of categories: group topics into one category only if they are sufficiently
similar; every topic_id must appear in exactly one category.

USER — revision turn (both stages; the items block of the stage is repeated above this text; no chat history)
Your previous answer:
{mine}
Your peer's latest answer:
{peer}
If the peer's answer is at least as good as yours, return {"decision": "accept"} and nothing else.
Otherwise return {"decision": "revise", "categories": [...]} with a revised answer that improves agreement with the
peer while preserving quality. Same schema. JSON only.
```

### 7.7 Outputs

* Per topic: `results/analysis/bin-N/labels/topic_<id>.json` — the agreed category, both analysts' final answers, the
  full transcript (every prompt and answer, decisions, seeds), `round_iterations`, `agreement_mode`; `labels.jsonl`
  (one line per topic, append-only, the resume file).
* Per bin: `taxonomy.json` (same structure for the categories) and `categories.csv`.
* **`results/analysis/topic_labels_all_bins.csv`** — the brief's single aggregated CSV, one row per topic cluster:
  `period_bin, topic_id, name, purpose, emotion_tone, tone_category, description, inclusion_cues, round_iterations,
  agreed, agreement_mode, needs_review, analyst_a_final, analyst_b_final, category_id, category_name,
  category_assignment, n_cluster, n_total, share, keywords, llm_model, llm_revision, prompt_version, decoding`.
  (The S3a answer's `id`, always `cat_1`, is not carried: it would collide with the post `id`.)
* `results/analysis/categories_all_bins.csv` — one row per S3b category: `period_bin, category_id, name, purpose,
  emotion_tone, tone_category, description, inclusion_cues, topic_ids, n_topics, n_posts, share, round_iterations,
  agreed, agreement_mode, needs_review`.
* `topic_overrides.csv`, `category_overrides.csv` — editable decision files (applied before S5).

## 8. Stage S4 — sentiment (VADER)

VADER (Hutto & Gilbert 2014) is the lexicon-and-rules scorer specified by the brief; its `compound` score lies in
[−1, 1]. Two measures are computed per post, both deterministic and cheap (≈ 15 min on CPU for the corpus):

* **`post_sentiment`** — compound score of the cleaned **title**. The title is the unit every post has; the compound
  score saturates with text length (summed valence normalised as x/√(x²+15)), and the self-post share falls from
  20–25 % in bins 1–3 to 5–7 % in bins 5–9, so scoring title + body would partly measure the shift from self posts to
  link titles across bins — the very comparison the chapter draws (F18).
* **`post_sentiment_full`** — compound score of title + body (≤ 1,000 characters), kept as the second measure for
  analyses of self posts.
* **`post_sentiment_label`** — from `post_sentiment`: `positive` > 0.05, `negative` < −0.05, otherwise `neutral`
  (the authors' conventional thresholds).

Sentiment does not depend on the topic stages; it is run after them as the brief specifies, but can run at any time.
**Validation:** a 300-post sample stratified by bin, label and `is_self`, hand-coded by two co-authors; agreement
(percent and Cohen's κ) reported in the session-05 record. Known limits (F10): headline-length text, sarcasm and
football vocabulary ("killer pass", "brutal", "crash out", "sick goal") that the general lexicon mis-scores; the
neutral share is high. A transformer-based measure can be added later as a further column without changing anything
else.

## 9. Stages S5 and S6 — tagging the corpus

### 9.1 S5 collect (after the reviews and overrides)

`data/analysis/post_variables.csv`, one row per post (1,302,021): `id, period_bin, period_type, topic_id,
topic_assignment, category_id, post_sentiment, post_sentiment_full, post_sentiment_label`, plus audit columns
`topic_id_raw` and `category_id_agent` (the pre-override category) that are **not** written to the master. A sidecar
`post_variables.json` records the taxonomy version, the run ids and software versions. Checks: every corpus id present
once; ids valid for their bin; scores within range.

### 9.2 S6 write-back (local, R; gated on the S3a/S3b review)

`code/07-tag-corpus.R` left-joins `post_variables.csv` by `id` into `data/soccer_data.clean.csv` and the 17 yearly
files, appending the **8 analysis columns** after the existing 42 (which stay byte-identical), with the same
`readr::write_csv(na = "")` call as the cleaning pipeline. Side files are not analysed and keep their 43 columns.
`code/08-validate-analysis.R` checks: row counts unchanged; every post has one bin consistent with its date; bin sizes
sum to the corpus; `topic_id`/`category_id` exist in their bin's tables; `topic_assignment` consistent with `topic_id`;
sentiment within [−1, 1] and labels consistent with the thresholds; the original 42 columns unchanged (hash); master
and `post_variables.csv` identical on the new columns.

**Keys.** `topic_id` and `category_id` are bin-local: the composite keys are (`period_bin`, `topic_id`) and
(`period_bin`, `category_id`). `category_id` uses the agents' `cat_N` ids after overrides. The taxonomy version is
recorded in the codebook's version row and the sidecar, not per row.

### 9.3 File layout

```
data/analysis/period_bins.csv, sentiment.csv, post_variables.csv (+ .json)        join tables (by id)
results/analysis/bin-1 … bin-9/   umap_5d.npy, hdbscan_labels_raw.npy, documents.csv, topics.csv,
                                  topics_over_time.csv, model/, labels/, labels.jsonl, taxonomy.json,
                                  categories.csv, topic_overrides.csv, category_overrides.csv,
                                  config.json, log.txt, report.html
results/analysis/topic_labels_all_bins.csv, categories_all_bins.csv, category_crosswalk.csv (manual, F4)
data/soccer_data.clean.csv, data/yearly datasets/*.csv                             v3.0 after S6 (50 columns)
```

## 10. Running on Google Colab

* **Runtime.** GPU, A100 40 GB preferred (L4 acceptable; T4 works in float16, roughly three times slower), high-RAM.
  Check in the account which plan currently provides the A100 and background execution: secondary sources report that
  Google folded Colab into its AI subscription plans in September 2026 and moved background execution to the top tier.
  The pipeline does not rely on background execution — every stage checkpoints to Drive and resumes.
* **Layout.** Code: `git clone https://github.com/alannemi/fifa-reddit-communities` in the runtime (public repository)
  or the Drive copy of `code/`. Data: upload `data/yearly datasets/` (2.4 GB) once to
  `MyDrive/fifa-reddit-communities/data/yearly datasets/`; at session start copy them to `/content/data` (local SSD —
  Drive I/O is slow for large CSVs). Results, the embedding cache and `.hf-cache` (model weights) live on Drive.
* **Environment.** Install `code/requirements-topicmodel.txt` without disturbing the pre-installed RAPIDS stack
  (install the `numpy`/`scikit-learn` lines with `--no-deps`, or pin them to the versions RAPIDS ships); the notebook
  asserts that `import cuml` works after the install and that CUDA is visible; the program writes the backend it
  actually used into `config.json`, so a silent fall-back to CPU is visible. Every `--stage topics` call is prefixed
  with `python -m cuml.accel`.
* **Notebook `code/06b-analysis-pipeline-colab.ipynb`** (orchestration only): (1) mount Drive, clone, copy data —
  the cell tests `drive.mount()` and falls back to downloading the data folder from a shared Drive link (`gdown`) if
  the kernel cannot mount; (2) install and assert GPU/cuML; (3) `--stage bins`; (4) loop over bins: `--stage embed`,
  `--stage topics`; (5) `--stage label` per bin (resumable) → **review pause** (co-authors read
  `topics_labelled.csv`, fill `topic_overrides.csv`); (6) `--stage taxonomy` per bin → **review pause**
  (`category_overrides.csv`); (7) `--stage sentiment`; (8) `--stage collect`; (9) copy `results/analysis/` and
  `data/analysis/` back to the repository / OneDrive.
* **VS Code Colab extension.** Open the notebook in VS Code, choose the Colab kernel (New Colab Server → GPU type);
  local files stay local, the kernel runs on the Colab machine. Whether `drive.mount()` works from the extension's
  kernel is not documented in the sources consulted; the first cell tests it (hence the fallback).
* **Estimates (A100).** Embeddings 15–20 min for 1.3 M posts; UMAP + HDBSCAN 1–5 min per bin with cuML (up to ≈ 1 h
  per large bin on CPU); S3a ≈ 700 topics × 2 analysts × ≈ 1.5 rounds ≈ 2,000 generations, batched ≈ 45 min (to be
  re-estimated from bin 1 before the full run); S3b minutes per bin; VADER 15 min; ≈ 3 h in total (≈ 35–50 compute
  units at the historical ≈ 12 units/h for an A100).

## 11. Reproducibility

Defined as **replay from persisted artefacts**, because GPU UMAP/HDBSCAN are not bit-deterministic even with a fixed
seed, batched text generation can differ marginally with batch composition, and the safetensors model omits UMAP and
HDBSCAN. Given `umap_5d.npy`, `hdbscan_labels_raw.npy`, `documents.csv` and the transcripts, every downstream output
(keywords, labels, taxonomies, prevalence, sentiment, tags) is identical on re-run. A fresh re-clustering on the same
backend is expected to match topic counts and sizes closely but not bit-for-bit; differences are reported, not treated
as errors. Fixed: seed 42 for UMAP, post sampling and shuffles; greedy decoding; batches formed in sorted topic-id
order; pinned package versions (`requirements-topicmodel.txt`) and Hugging Face revisions; per bin a `config.json` with
the effective settings, guard trials, backend, CUDA/cuML versions, `pip freeze`, the prompt text and
`prompt_version`.

## 12. Quality gates before the full run

1. **Bin 1 (2010) end-to-end on Colab.** Compare with `results/topic-models/window-wc2010` and `year-2010`: the
   tournament's known themes must surface (vuvuzelas, Jabulani ball, Lampard's disallowed goal, Suárez handball,
   Spain's win, Paul the octopus).
2. **Large-bin gate (bin 4 or 6).** Raw outlier share ≤ 45 %, guard converges within four trials, S3b partition valid
   without deterministic repair for most topics; otherwise revisit `min_samples` and the topic range before the other
   bins.
3. **Outlier threshold.** `T` from the similarity quantiles + the 100-post human check, on bin 1 and the large bin.
4. **Agents.** Round distribution and `agreement_mode` counts; `needs_review` ≤ 15 % of topics; greedy versus
   seeded-sampling comparison on bin 1; 20 topics read by a co-author against their posts; optional relabel of bin 1
   with Qwen3-4B (weights cached) to see what the larger model changes.
5. **Sentiment.** The 300-post hand-coded sample.
6. **Automatic checks** per bin (counts, coverage, partition validity, ids valid) and the S6 validation script.

## 13. Flagged issues and refinements (for the author's review)

| # | Issue | What the design does |
|---|---|---|
| F1 | The brief's schema repeats `inclusion_cues` (a string and a list; in JSON the last key wins) and asks the agent to output `round_iterations`, which only the debate loop knows | `inclusion_cues` is a list; `round_iterations` is computed by the pipeline (§7.5) |
| F2 | Which "id" tags a post: the BERTopic topic id or the agents' `cat_N`? | Both (D1): `topic_id` and `category_id`; the S3b schema gains `topics` so the mapping is explicit |
| F3 | A taxonomy of a whole bin from raw posts (e.g. 100 topics × 40 posts) does not fit a 32k context, and a 1.7B model reasons poorly over very long inputs | Two stages: label clusters, then group labelled topics (§7.3–7.4); ≤ 100 topics per bin; chunking fallback |
| F4 | Nine separate models → topic and category ids are bin-local; comparing World Cups needs a mapping layer | Editable `category_crosswalk.csv` (co-authors map bin categories to chapter themes, as the paper's manual merge); an automated step over the 9 taxonomies is designed as optional (S3c) but not run until S3b output has been reviewed |
| F5 | Bins differ 33× in size; a relative minimum topic size means ≥ 25 posts in bin 1 and ≥ 800 in bin 4 | Comparable topic counts enforced (§6.2); granularity differs by construction — interim topics are themes |
| F6 | The bins supersede the session-02 structure (year models, ±2-month windows, era alignment). Tournament bins start ≈ 4.3 months before kick-off and end ≈ 2.5 months after the final (bin 9: 12 days, corpus end) | Accepted at plan approval; the year/window runs stay as exploratory artefacts and are not re-run. Confirm the pre/post asymmetry is intended |
| F7 | Outlier policy decides which posts get a topic/category (37–40 % raw outliers in 2010) | D3: thresholded reassignment, calibrated on two bins (§6.3) |
| F8 | "Must agree" needs an operational rule, a cap and a fallback | D2: explicit acceptance, alternating turns, cap 5, `needs_review` (§7.5) |
| F9 | Qwen3-1.7B is specified; in the 2010 test it left 12–16 of 93–94 topics flagged; S3b is the harder task for a small model | Validation, repair and review steps (§7.4–7.5); optional 4B comparison on bin 1 (§12) |
| F10 | VADER on headline text: sarcasm and sport vocabulary mis-scored; high neutral share | Validation sample (§8); a transformer measure can be added later as a further column |
| F11 | Keyword lists showed scaffolding (`vs`, `soccer soccer`) | `vs`, `soccer` added to the keyword stop-list (keywords only) |
| F12 | Sampling at temperature 0.2 (session 02) is not reproducible | Greedy decoding; backend recorded; see F19 |
| F13 | Appending columns to the master changes the file spec: codebook v3.0, validation extension, a 1.9 GB rewrite plus yearly files (OneDrive sync time) | S6 gated on review; `post_variables.csv` is the auditable intermediate; side files unchanged |
| F14 | Posts with < 3 tokens get no topic/category (they do get sentiment). Non-English leftovers **are** modelled — there is no `lang` filter | Documented; `share_lang_en` per topic makes bilingual topics visible |
| F15 | The political-content layer (session 03 question) is outside this brief | Hook: inclusion cues + a later classifier on `post_variables.csv` |
| F16 | Acceptance by copying the peer's JSON verbatim is undetectable with a small model (paraphrase, reordering) | `decision` field; the pipeline substitutes the accepted answer; stalemate exit (§7.5) |
| F17 | Identical prompts + greedy decoding make the two analysts one deterministic function | Different lenses (§7.2); seeded sampling evaluated on bin 1 |
| F18 | VADER on title + body confounds sentiment with text length, and the self-post share varies 5–25 % across bins | Title-only `post_sentiment`; `post_sentiment_full` kept (§8) |
| F19 | Bit-identical re-runs are unattainable on GPU; the saved model lacks UMAP/HDBSCAN | Replay-based reproducibility with persisted arrays (§11) |
| F20 | S3b with history-carrying turns would overrun the context; a 1.7B model may not emit a valid partition | Stateless turns, ≤ 100 items, partition validation, deterministic repair with `category_assignment` (§7.4) |
| F21 | Deviations from the brief made explicit | §15 |

## 14. Decisions log (author: Alan Nemirovski, 2026-10-22)

| # | Decision | Chosen | Why |
|---|---|---|---|
| D1 | Labelling structure / which id tags a post | Two-level per bin: S3a one category per topic, S3b taxonomy with explicit `topics`; posts get `topic_id` + `category_id` | Honours both the brief's cluster labelling and its "group only if similar" rule; fits the model's context |
| D2 | Agreement rule and stopping | Explicit acceptance, alternating turns, cap 5 rounds; unresolved → `needs_review`, both answers kept | "Must agree on the output" taken literally; bounded runtime; nothing silently resolved |
| D3 | Outliers | Thresholded reassignment (embedding similarity ≥ T, calibrated); raw and final ids kept | Middle path between purity and coverage |
| D4 | Session scope | Design document + codebook now; implementation and test in session 05 | Author reviews F1–F21 first |
| Defaults accepted with the plan (2026-10-22) | bins supersede year/window/era runs (F6); `all-mpnet-base-v2`; Qwen3-1.7B, greedy, thinking off, lenses; `period_type` column; bin-local ids with composite keys; stop-list additions; title-only `post_sentiment` with `post_sentiment_full` (F18); topic range 50–100; `min_samples` 10 | As proposed; each is a single value in `config.py` |
| Open | `T` (outlier threshold), greedy vs seeded sampling, `min_samples` for large bins | Set at the quality gates (§12) and recorded here |

## 15. Brief versus design

| Brief | Design | Where |
|---|---|---|
| BERTopic on the corpus; parameters stated in the design document | One BERTopic model per period bin; all settings with rationale | §6 |
| Tag the relevant topic id ("id" of the schema) to each post in the main CSV | `topic_id` (BERTopic) **and** `category_id` (agents' `cat_N`) per post; written to the master and yearly files in S6 | §9, D1, F2 |
| Two Qwen3-1.7B agents agree on topic name and attributes through multi-turn cluster labelling; debate on disagreement; multiple rounds | S3a per topic with explicit acceptance, alternating turns, cap 5; unresolved flagged | §7.3, §7.5, D2 |
| Tag how many rounds it took in the output | `round_iterations` per topic and per category, computed by the pipeline | §7.5, F1 |
| The given prompt (may be refined; final version logged) | Prompt v1.1: schema fixed (F1), `decision` and `topics` fields added, lenses, stage-specific wording; the r/soccer context sentence; "Use keywords and tagged" read as "the tagged sample posts" (alternative reading, "tagged keywords", rejected because the items carry both) | §7.6 |
| "No minimum or maximum number of categories; cluster topics together only if sufficiently similar" | Applies to S3b (the taxonomy). S3a must return exactly one category per cluster, otherwise posts could not be assigned | §7.3–7.4, F21 |
| Posts fed to the agents | S3a: 40 posts per topic. S3b: topic lines (name, tone, keywords, description) instead of posts, for context reasons | §7.4, F3 |
| JSON per topic cluster aggregated into one separate CSV with `period_bin` | `topic_<id>.json` per topic → `topic_labels_all_bins.csv` (one row per topic, with the category columns joined); `categories_all_bins.csv` is an additional convenience table | §7.7 |
| Nine batches with the given dates; `period_bin` 1–9 per post in the main CSV | S0; `period_bin` (+ `period_type` convenience) | §4 |
| VADER after the agent stage; `post_sentiment` in the main CSV | `post_sentiment` on the title; `post_sentiment_full` and `post_sentiment_label` added | §8, F18 |
| Colab Pro+ GPUs, Colab VS Code extension | §10; cuML acceleration; resumable stages; the extension's Drive mount tested in the first cell | §10 |
| Variables logged in the codebook | Codebook v3.0 §4.9 (8 variables), §8 | codebook |
| Flaws flagged for review; ask before assuming | F1–F21; four questions asked and answered (D1–D4) before the plan was finalised | §13–§14 |

## 16. Code layout for the implementation (session 05)

Reused unchanged: `code/topicmodel/data.py` (documents), `embed.py` (cache), `sentiment.py` (VADER; gains the
title/full switch), `report.py`; `model.py` functions `build_topic_model`, `topic_table`, `topics_over_time`,
`save_model`; `label.py`'s `tone_category` mapper. `code/05-topic-model.py` stays as the exploratory year/window/era
tool.

| File | Role |
|---|---|
| `code/topicmodel/config.py` | adds `PERIOD_BINS`, `MIN_CLUSTER_FLOOR = 25`, `TOPIC_COUNT_RANGE = (50, 100)`, `HDBSCAN_MIN_SAMPLES = 10`, `OUTLIER_STRATEGY = "embeddings"`, `OUTLIER_THRESHOLD` (set after calibration, per bin allowed), `LLM_MODEL = "Qwen/Qwen3-1.7B"`, `LLM_REVISION`, `MAX_ROUNDS = 5`, `LLM_TEMPERATURE = 0` (+ `LLM_SAMPLING_SEEDED`), `LLM_BATCH_SIZE`, `PROMPT_VERSION = "1.1"`, `EXTRA_STOPWORDS += ["vs", "soccer"]`, `SENTIMENT_TEXT = "title"` |
| `code/topicmodel/bins.py` (new) | assign and validate `period_bin` / `period_type`; bin summary |
| `code/topicmodel/model.py` (extended) | `FrozenUMAP`; one UMAP fit per bin with persisted array; bisecting topic-count guard; thresholded outlier reassignment with `embeddings=`; `topic_id_raw`; cluster-only keywords and samples; backend detection |
| `code/topicmodel/prompts.py` (new) | prompt v1.1 texts and lenses; `PROMPT_VERSION` |
| `code/topicmodel/agents.py` (new) | batched alternating debate with `decision`, normalisation and partition comparison, stalemate detection, S3a and S3b, chunking fallback, deterministic repair, transcripts, aggregation, override application |
| `code/topicmodel/tagging.py` (new) | `post_variables.csv` + sidecar, `topic_labels_all_bins.csv`, `categories_all_bins.csv`, per-bin checks |
| `code/06-analysis-pipeline.py` (new) | the stage CLI (`--stage`, `--bin`, `--device`, `--embedding-model`, `--llm`, `--max-docs`) |
| `code/06b-analysis-pipeline-colab.ipynb` (new) | orchestration on Colab (§10) |
| `code/07-tag-corpus.R`, `code/08-validate-analysis.R` (new) | S6 write-back and validation |
| `code/requirements-topicmodel.txt` | RAPIDS-compatible pins; optional `vllm` extra |

Test sequence: local dry run on bin 1 (`--max-docs 3000`, MiniLM, guard disabled) → bin 1 in full on Colab → the
large-bin gate → gates of §12 → bins 2–9 → reviews → S5 → S6 → validation → review pack for co-authors.

## 17. References

* Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density
  estimates. *PAKDD 2013*, LNCS 7819, 160–172.
* Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
* Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media
  text. *Proceedings of ICWSM*, 8(1), 216–225.
* McInnes, L., Healy, J., & Astels, S. (2017). hdbscan: Hierarchical density based clustering. *Journal of Open Source
  Software*, 2(11), 205.
* McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for dimension
  reduction. arXiv:1802.03426.
* Qwen Team (2025). Qwen3 Technical Report. arXiv:2505.09388.
* Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of
  EMNLP-IJCNLP 2019*, 3982–3992.
* Template paper: anonymous ICWSM submission, *AI-Generated Sexual Content on Reddit: Analyzing Producer, Consumer and
  Victim Spheres* (methodology and appendix prompts), attached to the session-04 brief.
* Google Developers Blog (2025). Google Colab is coming to VS Code. https://developers.googleblog.com/google-colab-is-coming-to-vs-code/
* NVIDIA. cuML: GPU-accelerated machine learning (cuML pre-installed on Google Colab; `cuml.accel`).
  https://developer.nvidia.com/cuml and https://rapids.ai/cuml-accel/
