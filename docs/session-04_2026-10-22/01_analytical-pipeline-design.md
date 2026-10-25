# Analytical pipeline design — period-binned topic modelling, two-agent labelling and sentiment (v1.5)

**Session 04, 2026-10-22/23 · FIFA Communities chapter, McGill University.**
Decisions: Alan Nemirovski. Drafting and technical design: Claude. Brief: `00_brief.md`. Template method: the
ICWSM submission *AI-Generated Sexual Content on Reddit* (BERTopic → Qwen3-1.7B two-analyst multi-turn labels →
manual merge → aggregate reporting), used as a guide, not a recipe.

**Status.** Design approved on 2026-10-22 (decisions D1–D4, §14). **Implemented in session 05 (2026-10-23)**:
`code/06-analysis-pipeline.py` + `code/topicmodel/`, the Colab notebook, `07-tag-corpus.R`, `08-validate-analysis.R`;
the open flags F4/F6/F18 were approved as designed and a few implementation-level refinements were accepted — see
§14 (rows of 2026-10-23), §17 and `../session-05_2026-10-23/00_approvals.md`. Version 1.1 of this document records
them; the method itself is unchanged. **Version 1.2 (2026-10-24)** changes one stage: the taxonomy stage S3b runs a
two-step protocol — the agents first agree on a bin's category definitions, then assign each topic in its own debate
(decision D5, §7.4; prompt v1.2, §7.6) — after the 92-topic dry run showed that Qwen3-1.7B cannot partition a whole bin's
topics in one prompt (`../session-05_2026-10-23/02_dry-run-log.md` §3.8, finding D2; approvals in
`../session-06_2026-10-24/00_approvals.md`). The bin-1 test batch on Colab is session 06; the gates of §12, bins 2–9 and S6
follow. **Version 1.3 (2026-10-25)** records the code harmonization of session 07 — one pipeline under `code/` (the session-02
exploratory tool removed), the scripts renumbered (`05-analysis-pipeline.py`, `05b-…-colab.ipynb`, `06-tag-corpus.R`,
`07-validate-analysis.R`), one path root (`PIPELINE_ROOT`) and the flat Google Drive layout `MyDrive/fifa-communities/`
(`../session-07_2026-10-25/00_approvals.md`, decisions H1–H7); nothing analytical changed. The session documents of
October 2026 use the file names of their time. This document supersedes the *run structure* of `../session-02_2026-10-13/02_topic-modelling-design.md`
(calendar-year, ±2-month window and era models); the modelling settings approved there are kept unless stated. **Version 1.4 (2026-10-25, after the first Colab run of bin 1)** changes step 2 of the taxonomy stage: each
assignment item lists only the topic's most similar categories, most similar first, the answer names the category as well
as its id, and a topic whose agreed category ranks low by similarity is flagged (decision D6, §7.4; prompt v1.3, §7.6; flag
F41) — with all 35 categories listed, both analysts had agreed on the same wrong ids for about 18 of 65 topics
(`../session-08_2026-10-25/00_colab-run-log.md`). **Version 1.5 (2026-10-25, evening)** records two decisions of the author:
**D7** — the unit of analysis is the individual BERTopic topic; the taxonomy stage S3b (D5, D6, N13) is removed from the
pipeline and its code; every post carries its topic's corpus-wide key, agreed label and tone (`topic_uid`, `topic_name`,
`topic_tone`); the topic table carries the mean VADER score of the topic's posts (`mean_vader_sentiment`); the cross-bin
comparison is a manual crosswalk of topics (§7.7, §9, F42–F44). **D8** — no fixed topic range per bin but a density of 50–75
raw topics per 10,000 modelled posts, the guard searching the minimum cluster size from a floor of 25 (§6.2). §7.4 and the S3b
texts of §7.6 are kept as the record of the superseded protocol (`../session-08_2026-10-25/00_colab-run-log.md` §9).

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

1. **Per post** (appended to the master and yearly CSVs): the analysis period (`period_bin`), the BERTopic topic of the
   post — its bin-local `topic_id`, its corpus-wide key `topic_uid`, its agreed label and tone `topic_name` / `topic_tone`
   (D7, 2026-10-25) — and its VADER sentiment (`post_sentiment`).
2. **Per topic** (one aggregated CSV, the brief's "separate CSV"): the two-agent label of every topic cluster
   (name, purpose, emotional tone, description, inclusion cues), how many debate rounds the agents needed
   (`round_iterations`), and the period bin.
3. ~~**Per period bin**: a discourse taxonomy (categories grouping similar topics)~~ — removed by decision D7 (2026-10-25):
   the topic is the unit of analysis; the per-topic table adds the mean VADER score of the topic's posts, and a topic-level
   crosswalk template serves the manual cross-bin comparison.

Out of scope here (hooks left): the cross-period alignment of topics (a manual crosswalk, §13 F4/F43), a political
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
            ─► S4 VADER sentiment per post
            ─► S5 collect: post_variables.csv (every post, with its topic's uid, label and tone) + topic_labels_all_bins.csv
                           (with mean_vader_sentiment) + topic_crosswalk.csv (template)
            ─► S6 write-back (local, R): master + yearly files gain 10 columns ─► codebook v3.0 ─► validation
            (the S3b taxonomy stage of v1.0–v1.4 was removed by decision D7, 2026-10-25)
```

| Stage | Reads | Writes | Where it runs |
|---|---|---|---|
| S0 bins | yearly CSVs | `data/analysis/period_bins.csv` (id, period_bin, period_type) + bin summary | Colab or local |
| S1 embed | yearly CSVs | `results/topic-models/embeddings/<model>/<year>.npz` | Colab GPU |
| S2 topics | S0, S1 | `results/analysis/bin-N/` (§6.4) | Colab GPU (cuML) |
| S3a label | S2 | `bin-N/labels/topic_<id>.json`, `labels.jsonl`, `topics_labelled.csv` | Colab GPU |
| S4 sentiment | yearly CSVs | `data/analysis/sentiment.csv` (id, post_sentiment, post_sentiment_full, label) | CPU |
| S5 collect | S0–S4 (+ topic overrides) | `data/analysis/post_variables.csv` (+ `.json` sidecar), `results/analysis/topic_labels_all_bins.csv`, `topic_crosswalk.csv` | any |
| S6 write-back | S5 | `data/soccer_data.clean.csv`, 17 yearly files (v3.0), validation report | local (R) |

Dependency chain to respect: the topic overrides are applied by S5; S6 runs only after the co-author review of S3a (since
D7 there is no S3b). One command-line program runs the stages (`python code/05-analysis-pipeline.py --stage
{bins,embed,topics,grid,reassign,label,sentiment,collect,report} --bin N`);
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
| HDBSCAN `min_cluster_size` | start at the floor of 25 posts (`MIN_CLUSTER_FLOOR`), then the **topic-density guard** (§6.2, decision D8) | a topic holds at least 25 posts; the guard sets the size so that the bin reaches 50–75 raw topics per 10,000 modelled posts (v1.0–v1.4: 0.25 % of the bin, floor 25, towards 50–100 topics per bin) |
| HDBSCAN `min_samples` | 10 (`HDBSCAN_MIN_SAMPLES`) | the setting approved and tested in session 02; revisited only if the large-bin gate (§12) shows > 45 % raw outliers — since 2026-10-25 this and the three settings below may be one value or a dict by period type (`config.setting()`), the bins differing by an order of magnitude in size |
| HDBSCAN other | `metric="euclidean"`, `cluster_selection_method="eom"` (`HDBSCAN_SELECTION_METHOD`), `cluster_selection_epsilon` 0 (`HDBSCAN_EPSILON`), `prediction_data=True` | standard; the two settings exist for the gate decision; the values in force are recorded per bin in `config.json → topics.hdbscan` |
| Keywords: vectorizer | `CountVectorizer(ngram_range=(1,2), min_df=2, stop_words=English + EXTRA_STOPWORDS)`; `EXTRA_STOPWORDS` = the session-02 scaffolding list (`thread`, `match thread`, `post match`, `amp`, `http`, `reddit`, `imgur`, `youtube`, `twitter`, `streamable`, `edit`, `deleted`, `removed`, …) **plus** `vs`, `soccer` (F11) | stop-words affect only the keyword descriptions, never the documents or embeddings |
| Keywords: weighting | c-TF-IDF with `reduce_frequent_words=True`; `top_n_words=15`; representation chain KeyBERTInspired (30 candidates) → MaximalMarginalRelevance (`diversity=0.3`, 15 words) | readable, non-redundant keywords |
| Keywords: which posts | computed from **cluster members only** (posts assigned by HDBSCAN), not from reassigned outliers | the description stays pure; reassigned posts count towards prevalence only |
| Outliers | §6.3 | decision D3 |
| Representative posts | 5 nearest to the topic's c-TF-IDF representation (BERTopic) + 35 seeded random cluster members → the 40 posts the agents see | the paper used a random sample; the five anchors stabilise naming |
| Topics over time | weekly bins for tournament periods, monthly for interim periods (`topics_over_time`, no global/evolution tuning) | matches the period length |
| GPU | on Colab the stage runs under `python -m cuml.accel …` (NVIDIA RAPIDS cuML is pre-installed on Colab; its accelerator runs the unchanged `umap-learn`/`hdbscan` calls on the GPU); the program detects which backend actually ran and writes `backend: "cuml" | "cpu"` into `config.json`; the CPU path is the same code | the largest bins (≈ 330k × 768) take ≈ 1 h on CPU and minutes on an A100 |

### 6.2 Topic-density guard (decision D8, 2026-10-25; v1.0–v1.4: a topic-count guard towards 50–100 topics per bin)

Target: **50–75 raw topics per 10,000 modelled posts** (`TOPICS_PER_10K`; before any outlier handling), scaled to each bin
(`config.target_topic_range(n_modelled)`: bin 1 → 49–74, bin 4 → ≈ 1,650–2,480). Rationale (author, 2026-10-25): the topic is
the unit of analysis (D7), so the number of distinct topics should follow the volume of discussion rather than be fixed per
bin; a density keeps the granularity — posts per topic — comparable across bins that differ 33× in size. The earlier target
of 50–100 topics per bin served the taxonomy stage, which fed a bin's topics to the 1.7B model in one prompt.

Procedure: fit HDBSCAN on the frozen 5-d array at the floor `min_cluster_size` 25; record (`min_cluster_size`, `n_topics`,
`outlier_share`); if `n_topics` is outside the bin's range, move ×1.5 / ÷1.5, then bisect between the bracketing values
already tried; at most six fits (seconds each on the GPU); keep the value whose count is nearest to the range and flag the bin
(`guard.converged = False`) if the range was not reached. All trials and the HDBSCAN settings in force (`min_samples`,
selection method, epsilon — one value or per period type, §6.1) are logged in `config.json`. The guard is disabled for
debugging subsamples (`--max-docs`). A failed large-bin gate (§12.2) is diagnosed with `--stage grid` and applied with
`--stage topics --force --keep-umap`.

Consequence to keep in mind (F44): the interim bins will hold 1,000–2,500 topics each — a few posts short of a complete read
by the co-authors; the review there is the design's sample plus the flagged topics. Labelling scales linearly (≈ 2 min per 65
topics on the A100).

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
  vLLM is an optional accelerator with the same prompts. `max_new_tokens` 600 (S3a) / 4,000 (S3b step 1, definitions) /
  800 (S3b step 2, assignment); the superseded single-prompt S3b used 8,000 (R2).
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

### 7.4 Stage S3b — discourse taxonomy (per bin): the two-step protocol (v1.2, decision D5, 2026-10-24) — SUPERSEDED

> **Removed by decision D7 (2026-10-25):** the pipeline has no taxonomy stage; the individual topic is the unit of analysis.
> This section (D5, D6, N13 and §7.4.1) is kept as the record of the protocol that ran on bin 1 on 2026-10-25; its code is in
> git history up to `d120990b`.

Items are the bin's topics, one compact line each — `topic_id | agreed name | emotion_tone | 8 keywords |
one-sentence description` (≈ 60 tokens per topic). The brief's rule applies: no minimum or maximum number of
categories; topics are grouped only if sufficiently similar. The 1.7B model cannot, however, enumerate a whole bin's
topic ids inside category objects (dry-run finding D2: 92 items → answers cut off at 8,000 tokens, 443–605 ids listed
for 92 topics, 23 topics placed by the repair). The taxonomy is therefore built in two steps, both under the D2
protocol of §7.5:

* **Step 1 — definitions (one debate per bin).** Both analysts receive all topic lines and define the categories
  (`id, name, purpose, emotion_tone, description, inclusion_cues`), **without topic ids**. Validation: at least one
  named category (hard); duplicated names or an empty text field are soft problems — the answer is kept, the repair
  turn quotes them, and what the repair does not cure is settled deterministically (same-name categories merged, logged
  in `taxonomy.json → definitions_cleanup`). Ids are canonicalised to `cat_1 … cat_K` in the order given and never
  renumbered afterwards. Identity rule: same set of normalised names and, per name, the same `tone_category`. The agreed
  definitions are checkpointed (`taxonomy_definitions.json`) before step 2; `--force` removes the checkpoint
  (`--force --keep-definitions` keeps it, so that only step 2 re-runs — D6).
* **Step 2 — assignment (one debate per topic; `TAXONOMY_ASSIGN_BATCH = 1`).** Each item shows **the topic's
  `TAXONOMY_SHORTLIST` (10) most similar categories, most similar first** (decision D6, 2026-10-25: the cosine similarity
  of the sentence embeddings, `EMBEDDING_MODEL`, of the category line `category_id | name | emotion_tone | inclusion_cues |
  one-sentence description` and the topic line as they appear in the prompt; the whole list when the taxonomy has no more
  categories than the shortlist, or with `TAXONOMY_SHORTLIST = 0`) and one topic line; the analysts return `{"topic_id",
  "category_id", "category_name"}`. Category references are resolved leniently (exact id, bare number, the category's name;
  `null`/"none" count as missing; the resolution is recorded per answer); when id and name point to different categories
  the **name wins** (the id is the token a small model slips on) and the conflict is recorded (`resolved_from = name_over_id`,
  `cleanup`). Extra pairs — a topic id that is not in the item, a repeated id — are dropped and recorded **without a repair
  turn** (refinement N13, 2026-10-25: in the D6 dry run on bin 1, 16 of 65 items spent a repair turn on such extras and 5
  answers changed category in it, one from right to wrong); the repair turn remains for a topic left without a category or
  an unresolvable reference. Identity rule: the same topic → category mapping. Items are generated together per
  (round, analyst) like S3a topics. Several topics per item remain possible through the same setting (recorded per topic
  as `assignment_item`; the item then lists the union of the topics' shortlists, interleaved by rank). Before D6 (prompt
  v1.2, the first Colab run of bin 1) every item listed all 35 defined categories: both analysts ignored the categories late
  in the list and agreed on the same wrong ids — 8 categories stayed empty and about 18 of 65 topics were misplaced
  (`../session-08_2026-10-25/00_colab-run-log.md`).
* **Deterministic repair.** Topics the agents left without a category are placed in the category whose member-mean
  topic embedding is most similar (cosine); `category_assignment ∈ {agent, repaired, unplaced}`. A defined category to
  which no topic was assigned is dropped from `categories.csv` and listed in `taxonomy.json →
  categories_dropped_empty` (ids keep their gaps). Degenerate paths: no agreed definitions → step 2 skipped, all topics
  `unplaced`, bin state `labelled`; every assignment without answer → all topics `unplaced`.
* **Record.** `categories.csv` carries the definitions debate (`round_iterations`, `agreement_mode`, `agreed`,
  `needs_review`); `topic_categories.csv` carries, per topic, `assignment_item`, `assignment_round_iterations`,
  `assignment_agreement_mode`, `assignment_contested` (the analysts' final answers name different categories),
  `assignment_similarity_rank` (D6: the rank of the agreed category among all categories by similarity to the topic, 1 =
  nearest), `assignment_candidates` (the shortlist shown) and `assignment_needs_review` (item not agreed and contested, **or
  rank worse than `TAXONOMY_REVIEW_RANK` = 3** — the analysts agreed on a category the embeddings call a poor fit; the
  co-author review reads these first); `taxonomy.json` keeps both steps' transcripts, `categories_defined`, the repair log,
  `n_answers_salvaged` and `shortlist` (k, the rank distribution, `n_flagged_by_rank`).
* **Review files.** `category_overrides.csv` (rename or merge categories) and `topic_category_overrides.csv` (move one
  topic to another category, F28), both written by the taxonomy stage with the ids of the taxonomy they refer to.

#### 7.4.1 The single-prompt protocol of v1.0/v1.1 (superseded 2026-10-24; kept as `TAXONOMY_PROTOCOL = "partition"`)

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
   S3b step 1 compared as sets of category names with equal tone categories, S3b step 2 as the topic → category
   mapping, the superseded single prompt as partitions with matched categories' names and tone categories equal),
   the topic is agreed
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
accepted_by_B, stalemate, cap, no_answer}`, `needs_review`, `analyst_a_final`, `analyst_b_final`. Batch definition: round r, all
unresolved topics for Analyst A; topics A accepted leave the set; the remainder for Analyst B. In the two-step S3b
(v1.2) the definitions debate's fields describe the taxonomy as a whole (`categories.csv`), and every topic additionally
carries its assignment debate's `assignment_round_iterations`, `assignment_agreement_mode`, `assignment_contested` and
`assignment_needs_review` (`topic_categories.csv` and the label table; F22). Under two lenses and greedy decoding the
usual path to agreement is an `accept` in round 2 (S3a dry run: 76 of 92 topics), so Analyst B's wording is adopted
more often than Analyst A's.

### 7.6 Prompts (v1.4 since D7, 2026-10-25: the S3a and shared texts only; the S3b texts below are history; logged verbatim in every run's `config.json`)

v1.2 = v1.1 with the two-step S3b texts (step 1 definitions, step 2 assignment) and the revision turn naming the
stage's answer key; the S3a texts are byte-identical to v1.1 (approved as drafted, `../session-06_2026-10-24/00_approvals.md`
§3 and §7). The v1.1 single-prompt S3b texts remain in `code/topicmodel/prompts.py` (`SYSTEM_TAXONOMY`,
`USER_TAXONOMY`) for the `partition` protocol. Refinements of the brief's prompt in v1.1, each flagged in §13: F1 (duplicated `inclusion_cues` removed, `round_iterations`
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

SYSTEM — S3b step 1 (definitions; v1.2)
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

SYSTEM — S3b step 2 (assignment; text unchanged since v1.2)
You are {agent_name}, a discourse analyst studying posts from the football community r/soccer.
{lens}
Assign each topic to exactly one category of the agreed discourse taxonomy. Use the keywords and the labelled sample
of each item. Return JSON only; no prose or markdown.

USER — S3b step 2 (v1.3, decision D6; the v1.2 text listed the whole taxonomy and asked for category_id only)
Candidate categories for the topic(s) below: the {n_candidates} of the {n_categories} categories of the agreed taxonomy of
r/soccer posts between {start} and {end} that are most similar to them, most similar first, one per line:
   (or, when the whole taxonomy is listed: "Categories (the agreed taxonomy of r/soccer posts between {start} and {end}), one per line:")
category_id | name | emotion_tone | inclusion_cues | description
{category_lines}
Topics to assign ({n_batch} of the bin's {n_topics} topics), one per line:
topic_id | name | emotion_tone | keywords | description
{topic_lines}
Output schema:
{"decision": "revise",
 "assignments": [
  {"topic_id": <topic_id>, "category_id": "<category_id>", "category_name": "<name>"}
 ]}
Rules: list every topic_id above exactly once with the single best-fitting category; copy that category's category_id and
name exactly as listed, from the same line; use only the categories listed; if none fits well, choose the closest one.

(S3b single prompt of v1.1 — superseded, `partition` protocol only: SYSTEM "Build a discourse taxonomy from the
items. Use the keywords and the labelled sample of each item."; USER with the schema above plus "topics": [topic_id, ...]
per category and the rule "every topic_id must appear in exactly one category".)

USER — revision turn (all stages; the items block of the stage is repeated above this text; no chat history)
Your previous answer:
{mine}
Your peer's latest answer:
{peer}
If the peer's answer is at least as good as yours, return {"decision": "accept"} and nothing else.
Otherwise return {"decision": "revise", "{answer_key}": [...]} with a revised answer that improves agreement with the
peer while preserving quality. Same schema. JSON only.
   {answer_key} = "categories" (S3a, S3b step 1) | "assignments" (S3b step 2) — the only change to the v1.1 revision text
```

### 7.7 Outputs

* Per topic: `results/analysis/bin-N/labels/topic_<id>.json` — the agreed category, both analysts' final answers, the
  full transcript (every prompt and answer, decisions, seeds), `round_iterations`, `agreement_mode`; `labels.jsonl`
  (one line per topic, append-only, the resume file).
* **`results/analysis/topic_labels_all_bins.csv`** — the brief's single aggregated CSV, one row per topic cluster (D7):
  `period_bin, topic_id, topic_uid, name, purpose, emotion_tone, tone_category, description, inclusion_cues,
  round_iterations, agreed, agreement_mode, needs_review, analyst_a_final, analyst_b_final, n_cluster, n_total, share,
  mean_vader_sentiment, n_posts_scored, keywords, llm_model, llm_revision, prompt_version, decoding` (+ `name_agent`,
  `merged_from` from the overrides). `topic_uid` = `<period_bin>-<topic_id>`, unique across the corpus;
  `mean_vader_sentiment` = the mean title compound score of the topic's posts after reassignment and overrides (author,
  2026-10-25). (The S3a answer's `id`, always `cat_1`, is not carried: it would collide with the post `id`.)
* `results/analysis/topic_crosswalk.csv` — template for the co-authors' manual cross-bin crosswalk of topics to chapter
  themes (`period_bin, period_type, topic_id, topic_uid, name, n_total, chapter_theme, notes`; kept once it has entries; F43).
* `topic_overrides.csv` (rename / merge topics) — the one editable decision file, applied by S5.
* *(v1.0–v1.4, removed by D7: `taxonomy.json`, `categories.csv`, `topic_categories.csv`, `categories_all_bins.csv`,
  `category_overrides.csv`, `topic_category_overrides.csv`, `category_crosswalk.csv`.)*

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

`data/analysis/post_variables.csv`, one row per post (1,302,021): `id, period_bin, period_type, topic_id, topic_uid,
topic_name, topic_tone, topic_assignment, post_sentiment, post_sentiment_full, post_sentiment_label` (D7: the topic's
corpus-wide key, agreed name and controlled tone on every post that has a topic; empty for outliers and unmodelled posts),
plus the audit column `topic_id_raw` that is **not** written to the master. A sidecar `post_variables.json` records the
labels version (a hash of the labelled topics), the run ids and software versions. Checks: every corpus id present
once; ids valid for their bin; scores within range.

### 9.2 S6 write-back (local, R; gated on the S3a review)

`code/07-tag-corpus.R` left-joins `post_variables.csv` by `id` into `data/soccer_data.clean.csv` and the 17 yearly
files, appending the **10 analysis columns** after the existing 42 (which stay byte-identical), with the same
`readr::write_csv(na = "")` call as the cleaning pipeline. Side files are not analysed and keep their 43 columns.
`code/08-validate-analysis.R` checks: row counts unchanged; every post has one bin consistent with its date; bin sizes
sum to the corpus; `topic_id` exists in its bin's table; `topic_uid`, `topic_name` and `topic_tone` present iff the post has a
topic; `topic_assignment` consistent with `topic_id`;
sentiment within [−1, 1] and labels consistent with the thresholds; the original 42 columns unchanged (hash); master
and `post_variables.csv` identical on the new columns.

**Keys.** `topic_id` is bin-local; the composite key is (`period_bin`, `topic_id`), carried on every post as `topic_uid` =
`<period_bin>-<topic_id>`. The labels version is recorded in the codebook's version row and the sidecar, not per row.

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
* **Layout since session 07 (2026-10-25; decisions H2/H3).** Everything that persists lives in the flat Drive folder
  `MyDrive/fifa-communities/` — `yearly datasets/` (input), `analysis/` (join tables), `results/` (`bin-N/` and the aggregated
  tables), `embeddings/`, `hf-cache/`, `logs/`, `handback/`, `comparisons/` — through the program's root setting `PIPELINE_ROOT`
  (no symlinks). The code is cloned from GitHub on the Colab machine; the 17 CSVs are copied once per session to its local
  disk (`PIPELINE_YEARLY_DIR`). The hand-back zip rebuilds the repository paths. Locally the repository layout applies, with
  the embedding cache under `results/embeddings/`. Notebook: `code/05b-analysis-pipeline-colab.ipynb`.
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
2. **Large-bin gate (bin 4 or 6).** Raw outlier share ≤ 45 %, guard converges (to the density target of D8) within six
   trials; otherwise revisit `min_samples`, the selection method or epsilon before the other bins. Diagnostic (2026-10-25): `--stage grid --bin N` refits HDBSCAN over `HDBSCAN_GRID` on the bin's cached UMAP array
   (`hdbscan_grid.csv`: topics, outlier share, largest cluster per setting); the decision is applied with `--stage topics --force
   --keep-umap`. **First attempt, bin 4 (2026-10-25): guard 820 → 2 topics, 547 → 74 topics with 56.2 % raw outliers — failed on
   the outlier share; decision pending** (`../session-08_2026-10-25/00_colab-run-log.md` §8).
3. **Outlier threshold.** `T` from the similarity quantiles + the 100-post human check, on bin 1 and the large bin.
4. **Agents.** Round distribution and `agreement_mode` counts; `needs_review` ≤ 15 % of topics; greedy versus
   seeded-sampling comparison on bin 1 (dropped 2026-10-25, §14); 20 topics read by a co-author against their posts; optional relabel of bin 1
   with Qwen3-4B (weights cached) to see what the larger model changes (dropped 2026-10-24, §14). (The D6 reading of the assignment ranks fell away with the taxonomy stage, D7; the lesson of F41 — agreement between
   the two analysts is not a check on correctness — stands for the labels.)
5. **Sentiment.** The 300-post hand-coded sample.
6. **Automatic checks** per bin (counts, coverage, assignment coverage, ids valid) and the S6 validation script.

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
| F22 (2026-10-24) | Two S3b debates per bin instead of one; under two lenses the definitions are mostly adopted through `accept` (Analyst B's wording) | `round_iterations` in `categories.csv` = definitions debate; per-topic assignment rounds, modes and `contested` in `topic_categories.csv` and the label table; §7.4, §7.5 |
| F23 | A defined category to which no topic is assigned has no posts and is never a placement candidate | dropped from `categories.csv`, listed in `taxonomy.json → categories_dropped_empty`; ids keep gaps |
| F24 | Assignment can be batched; batches would be a procedural artefact | one topic per item (decided); batching stays a config value, item recorded per topic |
| F25 | Prompt version 1.2 applies to all records although only the S3b texts and the revision turn's answer key changed | S3a text byte-identical to v1.1; every stage logs its own prompt texts into its section of `config.json` |
| F26 | Keeping the single-prompt protocol adds a switch nobody may use | kept as `partition`: tested, replays the session-05 dry run, allows a comparison |
| F27 | The Colab environment is verified only at run time (hdbscan source build, transformers compatibility, cuML UMAP with the cosine metric) | assertions in the notebook's environment cell and `backend` in `config.json` |
| F28 | Co-authors could not move a single topic to another category | `topic_category_overrides.csv` written by the taxonomy stage, applied in S5; `category_id_agent` keeps the agents' value |
| F29 | Complete transcripts of two debates per bin ≈ 1 MB of `taxonomy.json` per bin in git | accepted: the transcripts are the audit trail |
| F30 | Assignment references are resolved leniently (names, bare numbers) and `null` counts as missing | recorded per answer (`resolved_from`); strictness is one function |
| F41 (2026-10-25) | The two analysts are one model: with all 35 categories listed in every assignment prompt they agreed on the same wrong ids (first Colab run of bin 1: 8 defined categories empty, ≈ 18 of 65 topics misplaced, `needs_review` 0 %); agreement is no check on correctness | D6: a similarity-ranked shortlist per topic, `category_id` + `category_name` in the answer (the name wins a conflict), a rank-based review flag (`assignment_similarity_rank`, `TAXONOMY_REVIEW_RANK`); gate 4 reads the rank distribution; the co-author review starts with the flagged topics |
| F42 (2026-10-25) | With the topic as the unit of analysis (D7) the chapter works with fine-grained topics — about 65 in bin 1 and 1,000–2,500 in each interim bin under D8 — instead of a dozen categories per bin | accepted by the author; the co-author review per bin is a sample plus the flagged topics; topic merges in `topic_overrides.csv` are the only consolidation |
| F43 (2026-10-25) | The cross-bin comparison (F4) moves from categories to topics: the crosswalk maps hundreds of topics to chapter themes | `topic_crosswalk.csv` template written by S5 (one row per topic, kept once filled); filled only when the chapter's analysis needs it, by hand |
| F44 (2026-10-25) | Under the density target the interim bins yield 1,000–2,500 topics: labelling ≈ 1 h per bin on the A100 (fine), but no complete human read | the design's sample (20 topics against their posts) plus every `needs_review` topic per bin; the author decides whether a larger sample is needed |

## 14. Decisions log (author: Alan Nemirovski, 2026-10-22)

| # | Decision | Chosen | Why |
|---|---|---|---|
| D1 | Labelling structure / which id tags a post | Two-level per bin: S3a one category per topic, S3b taxonomy with explicit `topics`; posts get `topic_id` + `category_id` | Honours both the brief's cluster labelling and its "group only if similar" rule; fits the model's context |
| D2 | Agreement rule and stopping | Explicit acceptance, alternating turns, cap 5 rounds; unresolved → `needs_review`, both answers kept | "Must agree on the output" taken literally; bounded runtime; nothing silently resolved |
| D3 | Outliers | Thresholded reassignment (embedding similarity ≥ T, calibrated); raw and final ids kept | Middle path between purity and coverage |
| D4 | Session scope | Design document + codebook now; implementation and test in session 05 | Author reviews F1–F21 first |
| Defaults accepted with the plan (2026-10-22) | bins supersede year/window/era runs (F6); `all-mpnet-base-v2`; Qwen3-1.7B, greedy, thinking off, lenses; `period_type` column; bin-local ids with composite keys; stop-list additions; title-only `post_sentiment` with `post_sentiment_full` (F18); topic range 50–100; `min_samples` 10 | As proposed; each is a single value in `config.py` |
| Open | `T` (outlier threshold), `min_samples` for large bins | Set at the quality gates (§12) and recorded here (greedy vs seeded: decided 2026-10-25, below) |
| 2026-10-23 F4 | Cross-bin comparability | Approved as designed: manual `category_crosswalk.csv` (template written by S5); S3c stays a hook | author's review before implementation |
| 2026-10-23 F6 | Bin boundaries | Approved exactly as in the brief (§4); no `in_tournament` column | idem |
| 2026-10-23 F18 | Sentiment measure | Approved: `post_sentiment` on the title, `post_sentiment_full` kept | idem |
| 2026-10-23 | Earlier test data | The session-02/03 run folders (`window-wc2010`, `year-2010`) removed from git, MiniLM cache deleted; `05-topic-model.py` kept as the exploratory tool | clean start for the bin-1 test |
| 2026-10-23 N1–N10, R1–R13 | Implementation refinements (approved with the session-05 plan) | §17; none changes D1–D4 or a codebook variable | `../session-05_2026-10-23/00_approvals.md` |
| **2026-10-24 D5** | S3b protocol after dry-run finding D2 | **Two-step: a definitions debate per bin, then one assignment debate per topic** (`TAXONOMY_PROTOCOL = "two_step"`, `TAXONOMY_ASSIGN_BATCH = 1`); prompt v1.2; the single prompt kept as `partition` | Qwen3-1.7B cannot enumerate 92 topic ids in one answer; alternatives declined: chunking at 30/25, Qwen3-4B for S3b, a single assignment-style prompt — `../session-06_2026-10-24/00_approvals.md` |
| 2026-10-24 N11, N12 | Tolerant JSON recovery; over-long cue lists kept | **Confirmed** | idem |
| 2026-10-24 | Qwen3-4B comparison (F9, §12.4) | **Dropped**: the pipeline stays with Qwen3-1.7B | author's decision |
| 2026-10-24 | Colab test batch | Pass 1 (every stage once, 12 labelled topics) + the full bin 1; gates and bins 2–9 afterwards | idem |
| 2026-10-24 F22–F30 | Consequences of the two-step protocol (above) | as listed | approvals §5 |
| **2026-10-25** | Greedy vs seeded sampling (F17, §12.4) | **Greedy decoding kept for every bin; the seeded-sampling comparison on bin 1 is dropped** (notebook cell 3.2 stays as an unused tool) | author's decision after Colab Pass 1 on bin 1 (65 topics, the 12 labelled topics sound): reproducible labels; the two lenses already give the second analyst a distinct view; the comparison would have cost a second labelling run of bin 1 |
| **2026-10-25 D6** | S3b step 2 after the first full run of bin 1 on Colab | **Each assignment item lists only the topic's `TAXONOMY_SHORTLIST` (10) most similar categories, most similar first (sentence embeddings of the category and topic lines); the answer carries `category_id` + `category_name` (a mismatch is resolved in favour of the name); a topic whose agreed category ranks worse than `TAXONOMY_REVIEW_RANK` (3) is flagged `assignment_needs_review`**; prompt v1.3; `--force --keep-definitions` re-runs step 2 alone | with all 35 categories listed, both analysts ignored the late categories and agreed on the same wrong ids (8 empty categories, ≈ 18 of 65 topics misplaced — `../session-08_2026-10-25/00_colab-run-log.md`); alternatives declined: embedding-only assignment with the analysts confirming, a review flag alone, hand repair per bin |
| 2026-10-25 N13 | Step-2 answers with extra pairs (a foreign or repeated topic id) or an id/name conflict | **Cleaned and recorded (`cleanup`), no repair turn**; the repair turn stays for a missing topic or an unresolvable reference | applied with D6 after its dry run on the Mac: 16 of 65 items had spent a repair turn on extras, 5 answers changed category in it (topic 18 from right to wrong) |
| **2026-10-25 D7** | The unit of analysis | **The individual BERTopic topic, labelled by the two analysts (S3a). The taxonomy stage S3b (D5, D6, N13) is removed from the pipeline and its code**; every post carries `topic_uid` (`<bin>-<topic_id>`), `topic_name` and `topic_tone`; the topic table carries `mean_vader_sentiment` and `n_posts_scored`; the cross-bin comparison is a manual topic crosswalk (`topic_crosswalk.csv`); the one review file is `topic_overrides.csv`; the analysis columns are 10 (52 in the data files); the bin-1 taxonomy outputs of the day were removed from the live folder (git history keeps them) | author's decision after the first Colab run (`../session-08_2026-10-25/00_colab-run-log.md` §9): the chapter wants the fine-grained topics, not an aggregation; alternatives declined: keeping the taxonomy code switched off; VADER-only topic sentiment (the analysts' tone stays, the VADER mean is added) |
| **2026-10-25 D8** | Topic granularity | **No fixed topic range per bin: 50–75 raw topics per 10,000 modelled posts (`TOPICS_PER_10K`), the guard searching `min_cluster_size` from the floor of 25** (§6.2); the HDBSCAN settings may be set per period type (`config.setting()`) from the grid diagnostic | author's decision: topics should emerge from the volume of discussion; the fixed 50–100 range served the removed taxonomy stage |
| **2026-10-25 H1–H7** | Code harmonization (session 07) | **H1** the session-02 exploratory tool removed entirely; **H2** flat, purpose-named Drive layout under `MyDrive/fifa-communities/`; **H3** Drive as the root for everything that persists, code cloned on the Colab machine, CSVs copied locally, no symlinks; **H4** shared R helper file `code/R/pipeline-utils.R`; **H5** scripts renumbered 06→05, 06b→05b, 07→06, 08→07; **H6** stale tracked copies under `data/` untracked; **H7** `03-data-validation.R` accepts the v3.0 spec now | `../session-07_2026-10-25/00_approvals.md` (defaults and flags F31–F38 there); no analytical change |

## 15. Brief versus design

| Brief | Design | Where |
|---|---|---|
| BERTopic on the corpus; parameters stated in the design document | One BERTopic model per period bin; all settings with rationale | §6 |
| Tag the relevant topic id ("id" of the schema) to each post in the main CSV | `topic_id` (BERTopic) **and** `category_id` (agents' `cat_N`) per post; written to the master and yearly files in S6 | §9, D1, F2 |
| Two Qwen3-1.7B agents agree on topic name and attributes through multi-turn cluster labelling; debate on disagreement; multiple rounds | S3a per topic with explicit acceptance, alternating turns, cap 5; unresolved flagged | §7.3, §7.5, D2 |
| Tag how many rounds it took in the output | `round_iterations` per topic and per category, computed by the pipeline | §7.5, F1 |
| The given prompt (may be refined; final version logged) | Prompt v1.2 (v1.1 + the two-step S3b texts, 2026-10-24): schema fixed (F1), `decision` field added, lenses, stage-specific wording; the r/soccer context sentence; "Use keywords and tagged" read as "the tagged sample posts" (alternative reading, "tagged keywords", rejected because the items carry both); the brief's category schema is answered by step 1 and the topic membership by step 2 | §7.6 |
| "No minimum or maximum number of categories; cluster topics together only if sufficiently similar" | Applies to S3b (the taxonomy). S3a must return exactly one category per cluster, otherwise posts could not be assigned | §7.3–7.4, F21 |
| Posts fed to the agents | S3a: 40 posts per topic. S3b: topic lines (name, tone, keywords, description) instead of posts, for context reasons | §7.4, F3 |
| JSON per topic cluster aggregated into one separate CSV with `period_bin` | `topic_<id>.json` per topic → `topic_labels_all_bins.csv` (one row per topic, with the category columns joined); `categories_all_bins.csv` is an additional convenience table | §7.7 |
| Nine batches with the given dates; `period_bin` 1–9 per post in the main CSV | S0; `period_bin` (+ `period_type` convenience) | §4 |
| VADER after the agent stage; `post_sentiment` in the main CSV | `post_sentiment` on the title; `post_sentiment_full` and `post_sentiment_label` added | §8, F18 |
| Colab Pro+ GPUs, Colab VS Code extension | §10; cuML acceleration; resumable stages; the extension's Drive mount tested in the first cell | §10 |
| Variables logged in the codebook | Codebook v3.0 §4.9 (8 variables), §8 | codebook |
| Flaws flagged for review; ask before assuming | F1–F21; four questions asked and answered (D1–D4) before the plan was finalised | §13–§14 |

## 16. Code layout for the implementation (session 05)

> **Note (2026-10-25, session 07).** The layout below is the one implemented in session 05. Since session 07 there is one
> pipeline under `code/`: `05-analysis-pipeline.py` (was `06-…`), `05b-analysis-pipeline-colab.ipynb` (was `06b-…`),
> `06-tag-corpus.R` (was `07-…`), `07-validate-analysis.R` (was `08-…`), the new `code/R/pipeline-utils.R`; `05-topic-model.py`,
> `05b-topic-model-colab.ipynb` and `topicmodel/label.py` were removed (the tone mapper lives in `agents.py`, its table in
> `config.TONE_STEMS`); `config.py` lost the exploratory settings and gained the path root (`PIPELINE_ROOT`, `PIPELINE_YEARLY_DIR`,
> `YEARLY_FILE`) and these renames: `EMBEDDING_MODEL` (mpnet; `EMBEDDING_MODEL_DRY_RUN` = MiniLM), `MIN_CLUSTER_FRACTION`,
> `MIN_CLUSTER_FLOOR`, `TIME_BIN`, `LLM_MAX_NEW_TOKENS_PARTITION`; `model.py` keeps one builder (`build_topic_model`, `fit_model`,
> `topic_table`, `topics_over_time`). Mapping and reasons: `../session-07_2026-10-25/01_code-harmonization-notes.md`.

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
| `code/topicmodel/agents.py` (new) | batched alternating debate with `decision`, normalisation and partition comparison, stalemate detection, S3a and S3b, chunking fallback, deterministic repair, transcripts, aggregation, override application; v1.2: definitions / assignment validators and identity rules, per-stage salvage, the two-step runner with checkpoint, the topic-level override reader |
| `code/topicmodel/tagging.py` (new) | `post_variables.csv` + sidecar, `topic_labels_all_bins.csv`, `categories_all_bins.csv`, per-bin checks |
| `code/06-analysis-pipeline.py` (new) | the stage CLI (`--stage`, `--bin`, `--device`, `--embedding-model`, `--llm`, `--max-docs`) |
| `code/06b-analysis-pipeline-colab.ipynb` (new) | orchestration on Colab (§10) |
| `code/07-tag-corpus.R`, `code/08-validate-analysis.R` (new) | S6 write-back and validation |
| `code/requirements-topicmodel.txt` | RAPIDS-compatible pins; optional `vllm` extra |

Test sequence: local dry run on bin 1 (`--max-docs 3000`, MiniLM, guard disabled) → bin 1 in full on Colab → the
large-bin gate → gates of §12 → bins 2–9 → reviews → S5 → S6 → validation → review pack for co-authors.

## 17. Implementation notes (session 05, 2026-10-23)

The code follows §3–§11 and the layout of §16; `../session-05_2026-10-23/01_implementation-notes.md` maps every
section to its function. Points where the implementation is more specific than, or deviates from, the text above:

* **`topic_id_raw` is the BERTopic id before outlier reassignment.** BERTopic renumbers the clusters it receives by
  size; the pipeline pre-renumbers the guard's HDBSCAN labels (largest → 0) and saves both `hdbscan_labels_raw.npy`
  (HDBSCAN's labels) and `hdbscan_to_topic_map.json` (R1). §6.3's "`topic_id_raw` = the HDBSCAN label" reads accordingly.
* **Outlier reassignment is a separate stage** (`--stage reassign`, N1): S2 saves every outlier's nearest topic and
  similarity, the quantile table and the 100-post judgement samples; `T` is applied afterwards in seconds, rewriting
  `documents.csv`, `topics.csv`, `topics_over_time.csv` and the report (R4). Until `T` is set, `topic_id = topic_id_raw`.
* **Frozen cluster labels** (N2): the guard's chosen labels are injected into BERTopic (no second HDBSCAN fit;
  `prediction_data` unnecessary). The guard never goes below the floor of 25; a bin that cannot reach 50 topics at
  the floor is flagged rather than split into smaller topics.
* **S3b token budget 8,000** instead of 4,000 (§7.1, R2); truncated answers are detected and reported in the repair turn.
* **`agreement_mode` gains `no_answer`** (R7): one analyst produced no usable round-1 answer after the repair turn; the
  peer's answer is provisional and the topic is `needs_review`. A failed revision turn counts as an identical revise
  (stalemate rule). An `accept` in round 1 is invalid.
* **Representative posts:** five per topic via BERTopic's extractor (it keeps three by default; R3).
* **Overrides** (R5): `merge_into` in `topic_overrides.csv` re-maps post `topic_id` in S5, drops merged-away topics from
  the S3b items and sums sizes; `category_id_agent` is the pre-override category.
* **documents.csv** also carries `num_comments` (needed by `reassign`); `topics_labelled.csv` carries no size columns.
* **S5 may run partially** (N6): all ids always present, `post_variables.json` lists `bins_complete`; `07-tag-corpus.R`
  refuses an incomplete table except for dry runs into another root. `08` verifies per-column xxhash64 fingerprints of
  the 42 original columns (R12).
* **Colab environment** (N5, R8): `requirements-colab.txt` installed with `--no-deps` on top of RAPIDS, `pip check`
  asserted; backend detection records `cuml.accel` state and the imported UMAP/HDBSCAN classes.
* **Seeded sampling** (F17 comparison) runs unbatched so the per-call seed is exact (N4).
* **Dry runs** write under `--out-root` (both `results/analysis` and `data/analysis`), never into the real folders (N10).
* **Two-step S3b (v1.2, session 06, 2026-10-24):** `agents.run_taxonomy()` dispatches on `TAXONOMY_PROTOCOL`;
  `run_taxonomy_two_step()` = definitions debate (`validate_definitions`, `cleanup_definitions`,
  `definitions_identical`, salvage of truncated definitions) → checkpoint → one assignment debate per topic
  (`validate_assignments` with lenient references, `assignments_identical`, `salvage_assignments`) → `repair_partition`
  → the shared writer; `Debate` takes the stage's token budget, salvage function and revision answer key; each stage
  logs its own prompt texts into `config.json`; review templates are regenerated unless filled in. Details:
  `../session-06_2026-10-24/01_implementation-notes.md`.

* **Session 07 (2026-10-25):** no change to the method; the code was consolidated into one pipeline, the paths re-rooted
  through `PIPELINE_ROOT`, the scripts renumbered and the Colab notebook rooted on `MyDrive/fifa-communities/`; the bin-1
  dry run after the clean-up reproduces the session-05 UMAP array and HDBSCAN labels exactly
  (`../session-07_2026-10-25/01_code-harmonization-notes.md` §8).

## 18. References

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
