# Session 05 (2026-10-23) — Implementation notes: how the design became code

**Author of the code: Claude; decisions: Alan Nemirovski** (`00_approvals.md`). Read with the design
(`../session-04_2026-10-22/01_analytical-pipeline-design.md`); this note says where each design section lives in the
code, what the code does that the design left open, and how to run and test it. The plan that was approved before
implementation is reproduced in §6.

## 1. Module map (design §16 → files)

| Design | File | What it holds |
|---|---|---|
| §0, all settings | `code/topicmodel/config.py` | existing settings unchanged except `LLM_MODEL` → Qwen3-1.7B, `LLM_TEMPERATURE` → 0, `EXTRA_STOPWORDS` + `vs`, `soccer` (N7); new block "Analytical pipeline": `PERIOD_BINS`, `PERIOD_BIN_EXPECTED_N`, `BIN_TIME_BIN`, `ANALYSIS_EMBEDDING_MODEL`, guard settings, outlier settings, LLM revision / rounds / token budgets / batch sizes, taxonomy chunking, sentiment settings |
| §4 S0 | `code/topicmodel/bins.py` | `bin_of()` (vectorised date → bin), `bin_years()`, `build_period_bins()` (writes `period_bins.csv` + `period_bins_summary.csv`, checks Σ, compares with the design table), `load_bin()` (documents of a bin through the join table, N8) |
| §5 S1 | `code/topicmodel/embed.py` (unchanged) | per-year `.npz` cache of **all** documents of a year, keyed by model name; both 05 and 06 use it |
| §6 S2 | `code/topicmodel/model.py` (appended) | `FrozenUMAP`, `FrozenCluster`, `detect_backend()`, `fit_umap()`, `guard_min_cluster_size()`, `renumber_by_size()`, `build_bin_topic_model()`, `fit_bin_model()`, `representative_ids()`, `outlier_nearest()`, `calibration_table()`, `calibration_samples()`, `resolve_threshold()`, `apply_outlier_threshold()`, `bin_topic_table()`, `bin_topics_over_time()` |
| §7.6 prompts | `code/topicmodel/prompts.py` | prompt v1.1 verbatim, lenses, repair-turn text, renderers, `all_prompts()` (logged in every `config.json`) |
| §7.1–7.5, 7.7 S3 | `code/topicmodel/agents.py` | `LLM` (batched greedy / unbatched seeded), `extract_json`, `validate_label`, `validate_taxonomy`, normalisation and partition comparison, `Debate` (the D2 engine), `run_labelling` (S3a), `run_taxonomy` (S3b incl. chunking and `repair_partition`), override readers/appliers |
| §8 S4 | `code/topicmodel/sentiment.py` (appended) | `score_year()` (cached per year), `run_sentiment()`, `validation_sample()` |
| §9.1 S5 | `code/topicmodel/tagging.py` | `collect()`: `post_variables.csv` + sidecar, `topic_labels_all_bins.csv`, `categories_all_bins.csv`, crosswalk template, checks |
| §3 CLI | `code/06-analysis-pipeline.py` | stages `bins, embed, topics, reassign, label, taxonomy, sentiment, collect, report`; `--out-root` for dry runs; per-bin `config.json` / `log.txt` |
| §10 Colab | `code/06b-analysis-pipeline-colab.ipynb`, `code/requirements-colab.txt` | orchestration only (19 cells); `--no-deps` install on top of RAPIDS |
| §9.2 S6 | `code/07-tag-corpus.R`, `code/08-validate-analysis.R` | write-back with per-column fingerprints; 12 check families |
| tests | `code/tests/test_pipeline_units.py` | 13 checks without GPU or model (`StubLLM`) |

## 2. What the code does that the design left open (all recorded in `00_approvals.md`)

* **`topic_id_raw`** is the BERTopic id *before* outlier reassignment. BERTopic renumbers the injected HDBSCAN labels by
  size; the pipeline pre-renumbers them (largest cluster → 0, ties by original label) so the renumbering is the
  identity up to ties, and saves `hdbscan_labels_raw.npy` (HDBSCAN's own labels) plus `hdbscan_to_topic_map.json`
  (HDBSCAN label → `topic_id_raw`) for replay (R1).
* **Outlier reassignment is `--stage reassign`** (N1). `--stage topics` always writes `outlier_nearest.csv` (nearest topic
  and cosine similarity for every raw outlier), `outlier_calibration/calibration_table.csv` (50th/65th/80th-percentile
  candidates) and one `sample_T<value>.csv` per candidate with 100 posts drawn from the similarity band between that
  candidate and the next higher one (so a co-author judges what *lowering* the threshold to that value would add).
  With no threshold configured, `topic_id = topic_id_raw`. `reassign` rewrites `documents.csv`, `topics.csv`
  (`n_total`, `share`, covariates), `topics_over_time.csv` and `report.html` (R4) and records the threshold in
  `config.json` under `reassign`.
* **Topic-count guard** never goes below `BIN_MIN_CLUSTER_FLOOR` (25): if the floor still yields fewer than 50 topics
  the bin is flagged (`guard.converged = false`) rather than split into tiny topics. Disabled under `--max-docs`.
* **"No answer"** (R7): `agreement_mode = "no_answer"` when one analyst produced no usable round-1 answer after the
  repair turn (the peer's answer is provisional, `needs_review = true`); a failed revision turn is treated as an
  identical revise (stalemate rule). An `accept` in round 1 is invalid (there is no peer answer yet) and triggers the
  repair turn.
* **Resume granularity** = topic: `labels.jsonl` is appended as soon as a topic resolves; an interrupted debate restarts
  from round 1 for that topic. Batches are formed per (round, analyst) over all unresolved topics in sorted id order.
* **S3b token budget** 8,000 (R2); a truncated answer is reported as such in the repair turn and in the transcript.
* **Inclusion cues (N12, added after the 92-topic run):** `validate_label` keeps the first ten cues of an over-long list and
  records `n_cues_given`; an empty list is still an error. Before this, over-long lists made whole labels `no_answer`.
* **Tolerant JSON recovery (N11, added during the dry run):** `agents.salvage_json()` recovers answers whose category
  objects are intact but whose brackets between them are not (the 1.7B model's `}], [{` pattern); it runs only after
  the strict parse failed and never on truncated answers; the transcript records `parsed_via` and the original parser
  error; `config.LLM_JSON_SALVAGE = False` disables it. A taxonomy that still yields no category writes empty
  `categories.csv` / all-`unplaced` `topic_categories.csv`, and S5 treats the bin as `labelled`, not `complete`.
* **Taxonomy ids**: the agents' `cat_N` ids are kept when they are well-formed and unique; otherwise categories are
  renumbered in the order given. `topic_categories.csv` (topic → category, `category_assignment`) is the per-topic map
  that S5 consumes; `categories.csv` carries `n_repaired`.
* **Overrides** (R5): `topic_overrides.csv` — `final_name` replaces the agreed name (`name_agent` keeps it),
  `merge_into` re-maps the posts' `topic_id` in S5 (never `topic_id_raw`), removes merged-away topics from the S3b items
  and sums sizes; `category_overrides.csv` likewise for categories (`category_id_agent` is the pre-override value).
  S3b must be re-run with `--force` after any S3a override, as the design requires.
* **documents.csv** carries `num_comments` in addition to the §6.4 columns, so `reassign` can recompute the topic
  covariates without reloading the corpus.
* **Embedding cache** holds all documents of a year (too-short ones included), the convention of the exploratory tool,
  so the two programs share one cache safely.
* **Sentiment** is cached per year under `data/analysis/sentiment/` (gitignored) and merged into `sentiment.csv`; the
  300-post validation sample (`sentiment_validation_sample.csv`) is stratified by bin × label × `is_self`, proportional
  to stratum size with at least two posts per non-empty stratum, with blank `coder_1`, `coder_2`, `notes` columns.
* **S5 partial mode** (N6): `post_variables.csv` always lists every corpus id; `post_variables.json` records
  `bins_state`, `bins_complete`, `complete`, the taxonomy version (hash of the nine `categories.csv`) and the
  software versions. `07-tag-corpus.R` refuses an incomplete table unless `TAG_ALLOW_PARTIAL=1` **and** the output
  root differs from the input root.
* **Fingerprints** (R12): `07` stores a per-column xxhash64 of the 42 original columns of every file it rewrites
  (`data/data artefacts/tagging_fingerprint.json`); `08` recomputes them and also compares the yearly files with the
  master's year subsets column by column.
* **Backend detection** (R8): `config.json → topics.backend` records `cuml_accel`, the cuML version and the classes
  of `umap.UMAP` / `hdbscan.HDBSCAN` that were imported, so a silent fall-back to CPU is visible.

## 3. Outputs per bin (`results/analysis/bin-N/`)

`umap_5d.npy`, `hdbscan_labels_raw.npy`, `hdbscan_to_topic_map.json`, `topic_embeddings.npy`, `documents.csv`,
`topics.csv`, `topics_over_time.csv`, `outlier_nearest.csv`, `outlier_calibration/`, `model/`, `config.json`
(sections `topics`, `reassign`, `label`, `taxonomy`), `log.txt`, `report.html` / `report.md`; after S3a:
`labels/topic_<id>.json` (full transcripts), `labels.jsonl`, `topics_labelled.csv`, `topic_overrides.csv`; after S3b:
`taxonomy.json`, `categories.csv`, `topic_categories.csv`, `category_overrides.csv`.
Join tables: `data/analysis/period_bins.csv`, `period_bins_summary.csv`, `sentiment.csv`, `sentiment_validation_sample.csv`,
`post_variables.csv` (+ `.json`). Aggregates: `results/analysis/topic_labels_all_bins.csv`, `categories_all_bins.csv`,
`category_crosswalk.csv` (template).

## 4. Running

See `memory/workflows.md` §7 (local commands and the dry-run recipe) and `03_runbook-session-06.md` (Colab).
Unit checks: `.venv/bin/python code/tests/test_pipeline_units.py`.

## 5. Known limits to keep in mind

* The Colab install (`requirements-colab.txt`, `--no-deps`) and `cuml.accel` could not be exercised from the Mac; the
  notebook's third cell asserts them and is the first item of the session-06 runbook.
* `representative_ids()` uses a private BERTopic method (`_extract_representative_docs`, version pinned to 0.17.4) with a
  public fallback.
* Seeded-sampling mode (`--sampling seeded`) is implemented for the bin-1 comparison (F17) but was only smoke-tested.

## 6. The approved plan (2026-10-23)

The plan file approved before implementation is kept verbatim below for the record.

## Session 05 plan — implement the analytical pipeline and set it up for the run session

**Prepared 2026-10-23 for Alan Nemirovski (decisions) · FIFA Communities chapter.** Basis: the approved design
`docs/session-04_2026-10-22/01_analytical-pipeline-design.md` (v1.0, §16 code layout), codebook v3.0 draft §4.9,
`memory/2026-10-22_session-04.md`, and the existing package `code/topicmodel/`. Status markers: **[Q]** = asked in
this planning round, **[default]** = accepted when this plan is approved unless you object, **[open]** = set later at
the quality gates (design §12), as decided in session 04.

---

### 0. Context

Session 04 designed the full analytical pipeline (nine period bins → one BERTopic model per bin → two Qwen3-1.7B
analyst agents labelling topics (S3a) and building a per-bin taxonomy (S3b) → VADER → eight analysis columns appended
to the master and yearly files) and left implementation to session 05. The goal of session 05 is to **build every
piece of §16, verify it end-to-end on a local dry run of bin 1, and leave the repository in a state where the run
session (06) only has to execute the Colab notebook, review outputs and take the gate decisions**. Your additional
instructions (2026-10-23): (i) you review and approve the previously flagged items before anything is implemented;
(ii) the data of the earlier 2010 test runs is removed; (iii) dry-run testing is done on bin 1 (1 Jan – 30 Sep 2010),
as §16 of the design specifies.

What exists and is reused (no rewrite): `topicmodel/data.py` (documents, `clean_text`, `build_documents`),
`embed.py` (per-year `.npz` cache, `embeddings_for`, `get_device`), `model.py` (`topic_table`, `topics_over_time`,
`save_model`), `label.py` (`tone_category` mapper, `parse_json` idea), `sentiment.py` (`score_documents`),
`report.py` (`write_report`), the R conventions of `02-data-cleaning.R` / `03-data-validation.R` (character-only
reads, `write_csv(na = "")`, `check()` PASS/FAIL with `stop()` at the end, `CLEAN_OUT_ROOT` dry-run override).
`code/05-topic-model.py` and `05b-topic-model-colab.ipynb` remain the exploratory tools (design §16).

Environment verified today: `.venv` Python 3.14.4 with bertopic 0.17.4, sentence-transformers 6.1.0, transformers
5.19.0, torch 2.14.1, umap-learn 0.5.12, hdbscan 0.8.44, pandas 3.0.6, vaderSentiment 3.3.2; Qwen3-1.7B snapshot
`70d244cc…` and Qwen3-4B cached in `.hf-cache/`; R 4.6.1 with readr/dplyr/stringr/jsonlite/digest; Apple M4, 16 GB,
180 GB free. BERTopic's `reduce_outliers(strategy="embeddings", embeddings=…, threshold=T)` exists as designed;
BERTopic calls `umap_model.fit_transform()` / `hdbscan_model.fit()` + `.labels_`, so frozen wrappers work.

---

### 1. Approval gate (before any code is written)

#### 1.1 Previously flagged items — asked and decided 2026-10-23 (Alan Nemirovski)

| Flag | What it is | Decision | Alternatives that were offered and declined |
|---|---|---|---|
| **F4** | Topic/category ids are bin-local; cross-World-Cup comparison needs a mapping layer. Design: a manual `category_crosswalk.csv` (co-authors map bin categories to chapter themes); an automated alignment (S3c) is designed but not run before S3b review | **Approved as designed.** Manual crosswalk; S3c stays a documented hook; S5 emits an empty `category_crosswalk.csv` template (R13) | build S3c now as an advisory stage; defer until after bin 1 |
| **F6** | The nine bins supersede the ±2-month windows; tournament bins start ≈ 4.3 months before kick-off and end ≈ 2.5 months after the final (bin 9: 12 days, corpus end) | **Approved as designed.** Boundaries exactly as in the brief / design §4; `period_type` stays the only convenience column | add an `in_tournament` flag column; revise the dates |
| **F18** | `post_sentiment` = VADER on the **title only** (comparable across bins and post types); title + body kept as `post_sentiment_full` | **Approved as designed.** | swap (full text primary); title only without the second column |
| Old test data | What to remove from the session-02/03 2010 test | **Run folders + cache:** `git rm -r results/topic-models/window-wc2010 results/topic-models/year-2010`, delete the local MiniLM cache; keep `05-topic-model.py` / `05b` as exploratory tools; keep the session-02/03 docs as history with a dated note | also archive the old code; run folders only |

These four decisions are recorded in `docs/session-05_<date>/00_approvals.md` (flag, decision, decider, date) and
as new rows of the design's decisions log (§14) in the housekeeping commit, before any pipeline code is written.

#### 1.2 Status of the other flags (F1–F21) — nothing to re-decide unless you want to reopen one

Decided in session 04 (D1–D4 / defaults accepted 2026-10-22): F1 schema fixes, F2 both ids tag a post, F3 two stages,
F5 relative minimum topic size, F7/D3 thresholded outliers, F8/D2 agreement rule, F9 validation + optional 4B
comparison, F10 VADER limits + validation sample, F11 `vs`/`soccer` stop-words, F12/F19 replay reproducibility,
F13 S6 gated on review, F14 non-English leftovers modelled, F15 political layer out of scope, F16 `decision` field,
F17 analyst lenses, F20 stateless S3b turns, F21 brief-vs-design table.
Still **[open]** by design, set at the gates in session 06 and not now: outlier threshold `T`, greedy vs seeded
sampling, `min_samples` for large bins.

#### 1.3 New implementation-level refinements found in this planning — **[default]** unless you object

These do not change any decision D1–D4 or any variable of codebook §4.9; they change *how* a stage is built.
They will be written up as N1–N10 in `00_approvals.md` with the same who/why format.

| # | Refinement | Why |
|---|---|---|
| N1 | **Outlier reassignment becomes its own cheap, replayable step** (`--stage reassign`). S2 always saves, for every raw outlier, its nearest topic and cosine similarity (`outlier_nearest.csv`), writes the calibration table (similarity quantiles) and the three 100-post judgement samples (`outlier_calibration/`). Until `OUTLIER_THRESHOLD` is set, `topic_id = topic_id_raw` and `topic_assignment ∈ {cluster, outlier, not_modelled}`; `--stage reassign --bin N` applies `T` in seconds without refitting | D3 says `T` is calibrated *after* bin 1 and a large bin exist; folding reassignment into S2 would force a refit per candidate `T` |
| N2 | **Frozen HDBSCAN labels.** The topic-count guard runs HDBSCAN directly on the saved 5-d array; the chosen trial's labels are injected into BERTopic through a `FrozenCluster` (`fit` no-op, `labels_` preset), so `hdbscan_labels_raw.npy` is *exactly* what BERTopic used (no second fit; `prediction_data` not needed since `calculate_probabilities=False`) | removes one source of GPU non-determinism within a run |
| N3 | S2 also saves `topic_embeddings.npy` (768-d, one row per topic) next to `model/` | S3b's deterministic repair and chunking fallback need them without loading BERTopic |
| N4 | Seeded-sampling comparison mode (F17 test) runs **unbatched** so the per-call SHA-256 seed is exact; greedy mode is batched across topics | HF `generate` has no per-sequence seed |
| N5 | **Requirements split:** `requirements-topicmodel.txt` (local, pinned as today) + new `requirements-colab.txt` (no torch/numpy/scikit-learn/pandas pins; installed on top of Colab's RAPIDS stack); the notebook asserts `import cuml`, CUDA and prints resolved versions | design §10: do not disturb the pre-installed RAPIDS stack |
| N6 | **S5 may run partially.** `post_variables.csv` always holds all 1,302,021 ids; topic/category columns stay empty for bins not yet modelled; the sidecar lists `bins_complete`. `07-tag-corpus.R` refuses to write back unless all nine bins are complete, except with `TAG_ALLOW_PARTIAL=1` **and** `TAG_OUT_ROOT` ≠ repository (dry runs only) | lets the bin-1 test exercise S5/S6 without touching the real files |
| N7 | Shared `config.py` defaults move to the design values: `LLM_MODEL` → Qwen3-1.7B (was 4B), `LLM_TEMPERATURE` → 0 (was 0.2), `EXTRA_STOPWORDS` + `vs`, `soccer`; a separate `BIN_MIN_CLUSTER_FLOOR = 25` is added so `05-topic-model.py` keeps its floor of 15 | one settings file (design §0); 05's old runs are being deleted anyway |
| N8 | S2 takes a bin's documents by joining `period_bins.csv` on `id`, not by recomputing dates | S0 is the single source of truth for `period_bin`; S6 checks consistency against it |
| N9 | `--stage sentiment` also writes `data/analysis/sentiment_validation_sample.csv` (300 posts stratified by bin × label × `is_self`, blank coder columns) | design §8 validation sample becomes a produced file |
| N10 | Dry-run artefacts go to a gitignored root (`--out-root results/dry-runs/<name>`), never into `results/analysis/bin-1/`; `--out-root` redirects **both** `results/analysis` and `data/analysis` (only the embedding cache is shared) | the real bin-1 run on Colab must start clean (R9) |

#### 1.4 Independent review of this draft (2026-10-23) — findings folded in

A fresh reviewer checked the draft against the design, the codebook, the existing code and the BERTopic 0.17.4
source. All 13 findings were accepted; they appear below as R1–R13 and are already incorporated into §4–§6. Two of
them (R1, R2, R7) touch definitions and are therefore **also listed for your awareness** — they will be written into
`00_approvals.md` with the N-items:

| # | Finding | What the plan now does |
|---|---|---|
| R1 (high) | BERTopic re-numbers cluster labels by size (`_sort_mappings_by_frequency`, `topic_mapper_`), so the injected HDBSCAN labels are **not** BERTopic's `topic_id` | Labels are pre-renumbered by cluster size before freezing, and `topic_mapper_.get_mappings()` is saved as `hdbscan_to_topic_map.json` next to `hdbscan_labels_raw.npy`; `topic_id_raw` in `documents.csv` / `post_variables.csv` is defined as **the BERTopic id before outlier reassignment** (codebook §4.9 audit-column sentence gains "HDBSCAN cluster re-numbered by size") |
| R2 (high) | `LLM_MAX_NEW_TOKENS_TAXONOMY = 4000` can truncate a 30–50-category S3b answer and push the bin into deterministic repair | 8,000 tokens; truncation detected (generation hit the cap / no closing brace) and recorded as `truncated: true` in the transcript; deviation from design §7.1 logged |
| R3 (medium) | BERTopic stores only 3 representative docs per topic; design wants 5 | After the fit, `_extract_representative_docs(..., nr_repr_docs=5)` is called on cluster members (private API, version pinned), with a fallback to the 5 members nearest the topic embedding |
| R4 (medium) | Sizes (`n_total`, `share`), `topics_over_time`, `similarity` and the report depend on `T`, which is calibrated after S3a may have run | `--stage reassign` rewrites `documents.csv`, `topics.csv`, `topics_over_time.csv` and `report.html`; `topics_labelled.csv` carries **no size columns**; `collect` joins sizes from `topics.csv`; runbook order S2 → calibrate → reassign → S3a, code safe either way |
| R5 (medium) | `merge_into` semantics unspecified | `apply_topic_overrides` re-maps post `topic_id` (keeping `topic_id_raw`), drops merged-away topics from the S3b items and sums sizes; `category_id_agent` = the category before category overrides |
| R6 (medium) | The D2 debate loop would only be exercised live | `StubLLM` with scripted answers keyed by (topic, analyst, round): tests for every `agreement_mode`, the repair and no-answer paths, resume from a partial `labels.jsonl`, chunking with ~130 fake topics and deterministic repair |
| R7 (medium) | "No answer" after the repair turn is undefined | Failed revision turn → the analyst keeps its previous answer (counts as an identical `revise`, so the stalemate rule applies); failed round-1 answer → the other analyst's answer is provisional, `agreed = False`, `needs_review = True`, `agreement_mode = "no_answer"` (one new value in the vocabulary; both fail → empty label, `needs_review`) |
| R8 (medium) | A plain `pip install -r` on Colab can move numpy/scikit-learn/pandas against the RAPIDS pins | `pip install --no-deps` of the five pinned packages, then `pip check` + version asserts; backend detected from the class module of the fitted UMAP/HDBSCAN objects *and* `cuml.accel.enabled()`; cuML fallback logging enabled; first item of the session-06 runbook |
| R9 (low/med) | `--out-root` redirected results but not `data/analysis` | Both redirected (N10) |
| R10 (low) | S0 would load `selftext` for all 17 files at once | Per-year read with `usecols`, flags derived, text dropped before concatenating |
| R11 (low) | Qwen3's `generation_config.json` defaults to sampling | Greedy calls pass `do_sample=False, temperature=None, top_p=None, top_k=None`, explicit `pad_token_id`, left padding; `dtype=` keyword (transformers 5) |
| R12 (low) | R memory/runtime on a 16 GB Mac | 07 processes the master first, then yearly files one at a time (`rm(); gc()`); digests on bare character vectors; 08 compares per-year column digests instead of a full subset comparison |
| R13 (low) | Scope | vLLM hook dropped; `--stage report` limited to `report.write_report`; the seeded 5-topic run is a smoke test, not an acceptance criterion; empty `category_crosswalk.csv` template emitted at collect |

---

### 2. Scope of session 05

**In:** housekeeping (§3); all code of §4; the bin-1 dry run (§5); documentation, codebook touch-ups, memory and
the handoff runbook (§6); one commit per milestone, pushed at the end.
**Out (session 06, the run session):** bin 1 in full on Colab (mpnet + cuML), the large-bin gate, calibration of
`T`, agent QA, the sentiment hand-coding, bins 2–9, co-author reviews, S6 write-back, codebook v3.0 final,
`03-data-validation.R` update to 50 columns.

---

### 3. Housekeeping (first commit) — scope decided 2026-10-23 (§1.1)

1. Remove the earlier test data: `git rm -r results/topic-models/window-wc2010 results/topic-models/year-2010`
   (tracked; recoverable from git history), delete the local MiniLM cache `results/topic-models/embeddings/`
   (gitignored). Add a dated note at the top of `docs/session-02_2026-10-13/03_topic-model-test-case.md` and in
   `README.md` §4 / `memory/workflows.md` §6 saying the run folders were removed on <date> and how to regenerate them
   with `05-topic-model.py` (the docs stay as the historical record; the code stays as the exploratory tool).
2. `.gitignore`: add `/results/dry-runs/`, `/data/analysis/sentiment/` (per-year cache; the merged file is under
   `/data/*` already). `results/analysis/` **is tracked** (CSV/JSON/arrays; largest file ≈ 25 MB for bin 4).
3. Create `docs/session-05_<date>/` with `00_approvals.md` filled from your answers (§1).

---

### 4. Implementation (build order; each file's role as in design §16)

#### 4.1 `code/topicmodel/config.py` (extend; nothing removed)
`ANALYSIS_DIR = ROOT/"results"/"analysis"`, `ANALYSIS_DATA_DIR = DATA_DIR/"analysis"`;
`PERIOD_BINS` (the nine rows of design §4: bin, start, end, `period_type`, World Cup kick-off/final);
`BIN_TIME_BIN = {"tournament": "W", "interim": "M"}`; `ANALYSIS_EMBEDDING_MODEL = EMBEDDING_MODEL_HQ` (mpnet);
`BIN_MIN_CLUSTER_FRACTION = 0.0025`, `BIN_MIN_CLUSTER_FLOOR = 25`, `TOPIC_COUNT_RANGE = (50, 100)`,
`GUARD_MAX_TRIALS = 4`, `GUARD_STEP = 1.5`; `OUTLIER_STRATEGY = "embeddings"`, `OUTLIER_THRESHOLD = {}`
(`{"default": T}` or per-bin `{1: T1, …}`; empty = not calibrated), `OUTLIER_CALIBRATION_QUANTILES = (0.50, 0.65, 0.80)`,
`OUTLIER_SAMPLE_PER_CANDIDATE = 100`; `LLM_MODEL = "Qwen/Qwen3-1.7B"`, `LLM_REVISION = "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e"`,
`LLM_TEMPERATURE = 0`, `LLM_SAMPLING_SEEDED = False`, `LLM_SAMPLING_TEMPERATURE = 0.7`,
`LLM_BATCH_SIZE = {"cuda": 16, "mps": 4, "cpu": 1}`, `MAX_ROUNDS = 5`, `LLM_MAX_NEW_TOKENS_LABEL = 600`,
`LLM_MAX_NEW_TOKENS_TAXONOMY = 8000` (R2; design said 4,000), `LLM_POST_MAX_CHARS = 300`, `TAXONOMY_MAX_ITEMS = 100`,
`TAXONOMY_CHUNK_SIZE = 80`;
`SENTIMENT_TEXT = "title"`, `SENTIMENT_VALIDATION_SAMPLE = 300`; `EXTRA_STOPWORDS += ["vs", "soccer"]`.
`effective_settings()` unchanged (it already serialises every upper-case setting).

#### 4.2 `code/topicmodel/bins.py` (new, S0)
`bin_of(dates) -> int[]` (vectorised `searchsorted` on bin start dates; asserts every date inside bin 1–9),
`period_type_of(bin)`, `bin_years(bin)` (yearly files a bin touches; bin 7 = 2022 + 2023), `bin_dates(bin)`.
`build_period_bins()` reads the 17 yearly files one year at a time with `usecols` (id, created_datetime, is_self,
selftext, post_type, lang), derives the flags and drops the text before concatenating (R10) →
`data/analysis/period_bins.csv` (`id, period_bin, period_type`) and `period_bins_summary.csv` (n, self share,
with-body share, match-thread share, `lang = en` share, titles < 3 tokens, starting `min_cluster_size`); hard check
Σ = rows read; the design's expected counts (10,002 … 37,626; Σ 1,302,021) are compared and any difference logged.
`load_bin(bin) -> (docs, full_year_docs)` = `data.load_year` + `data.build_documents` for `bin_years`, joined to
`period_bins.csv` by id, filtered to the bin (N8); `full_year_docs` feeds `embed.embeddings_for` unchanged.

#### 4.3 `code/topicmodel/model.py` (extend; existing functions untouched)
* `FrozenUMAP(array)` — `fit` no-op, `transform`/`fit_transform` return the saved 5-d array.
* `FrozenCluster(labels)` — `fit` no-op, `labels_` preset (N2).
* `detect_backend()` → `"cuml"` if `cuml.accel` reports enabled (defensive: try `cuml.accel.enabled()`, else inspect
  the module of the `umap.UMAP` class), else `"cpu"`; written to `config.json`.
* `fit_umap(emb) -> np.ndarray` with `C.UMAP_PARAMS` (under `python -m cuml.accel` this is the GPU UMAP); saved as
  `umap_5d.npy`.
* `guard_min_cluster_size(umap5d, n_modelled, enabled=True) -> (labels, trials, converged)`: start
  `max(25, round(0.0025·n))`; if `n_topics` outside 50–100 try ×1.5 / ÷1.5 then bisect between bracketing values; ≤ 4
  fits; keep the value nearest the range; `trials` = list of `(min_cluster_size, n_topics, outlier_share)`; disabled
  (single fit) under `--max-docs`. The chosen labels are saved as `hdbscan_labels_raw.npy`, then **renumbered by
  cluster size** (largest → 0, ties by original label) before freezing (R1).
* `build_bin_topic_model(umap5d, labels, embedding_model_name, device)` — same vectorizer / c-TF-IDF /
  KeyBERTInspired → MMR chain as `build_topic_model`, with the two frozen components; after `fit_transform(docs, emb)`
  the mapping `model.topic_mapper_.get_mappings()` is saved as `hdbscan_to_topic_map.json` and asserted to be the
  identity up to ties (R1); keywords come from cluster members only (§6.1); representative posts: 5 per topic via
  `model._extract_representative_docs(c_tf_idf, docs_df, topic_representations_, nr_samples=500, nr_repr_docs=5)`
  (BERTopic stores only 3 by default; private API, pinned version; fallback = the 5 members nearest the topic
  embedding) (R3). `topic_id_raw` = BERTopic id before reassignment.
* `outlier_nearest(model, emb, labels_raw) -> DataFrame(id, nearest_topic, similarity)` using
  `model.topic_embeddings_` (cosine); `calibration_table(nearest)` (quantiles) and `calibration_samples(…)` (three
  100-post CSVs with id, title, candidate `T`, nearest topic keywords, blank `belongs` column) (N1).
* `apply_outlier_threshold(labels_raw, nearest, T) -> (topic_id, topic_assignment, similarity)`; `T` resolved per bin
  from `OUTLIER_THRESHOLD` or `--outlier-threshold`. `--stage reassign` then **rewrites** `documents.csv`,
  `topics.csv` (`n_total`, `share`), `topics_over_time.csv` and `report.html` (R4).
* `bin_topic_table(model, docs, topic_id_raw, topic_id)` → §6.4 `topics.csv` columns (`n_cluster`, `n_total`,
  `share`, keywords, `representative_ids/titles`, `share_match_thread`, `share_lang_en`, `mean_comments`), built on
  the existing `topic_table`.
* `sample_posts_for_topic(…)` → the 5 representative posts + 35 seeded-random cluster members (seed 42,
  excluding the five; all members if fewer than 40), each truncated to 300 chars, title first.
* `bin_topics_over_time(model, docs, topics, period_type)` — weekly / monthly via `BIN_TIME_BIN`.

#### 4.4 `code/topicmodel/prompts.py` (new)
Prompt v1.1 verbatim from design §7.6: `SYSTEM_LABEL`, `USER_LABEL`, `SYSTEM_TAXONOMY`, `USER_TAXONOMY`,
`USER_REVISION`, `LENSES = {"Analyst A": …keyword-first, "Analyst B": …post-first}`, `PROMPT_VERSION = "1.1"`;
`render_label(agent, keywords, posts)`, `render_taxonomy(agent, n_topics, start, end, topic_lines)`,
`render_revision(items_block, mine, peer)` (stateless: items + two latest answers, no history); `all_prompts()` for
`config.json`. Topic lines for S3b: `topic_id | agreed name | emotion_tone | 8 keywords | one-sentence description`.

#### 4.5 `code/topicmodel/agents.py` (new; the largest piece)
* `LLM`: loads `LLM_MODEL` at `LLM_REVISION` (`dtype=` bf16 on CUDA/MPS, fp16 on T4 detected by capability, fp32
  CPU; tokenizer `padding_side="left"`, `enable_thinking=False`); `generate(prompts, max_new_tokens, seeds=None) ->
  list[str]` batched greedy — explicitly `do_sample=False, temperature=None, top_p=None, top_k=None`, explicit
  `pad_token_id`, because Qwen3's `generation_config.json` defaults to sampling (R11) — or unbatched seeded sampling
  (N4); strips `<think>` blocks; reports per answer whether generation hit `max_new_tokens` (`truncated`, R2); logs
  tokens and seconds per batch. No vLLM in session 05 (R13).
* Parsing/validation: `extract_json(text)` (`JSONDecoder().raw_decode` from the first `{`), `validate_label(obj)`
  (`decision`; exactly one category with `name, purpose, emotion_tone, description, inclusion_cues[list 1–10]`),
  `validate_taxonomy(obj, topic_ids)` (categories with `id, name, purpose, emotion_tone, description, inclusion_cues,
  topics`; partition check → list of problems); one repair turn quoting the error (or the truncation); second
  failure = **no answer** (R7): in a revision turn the analyst keeps its previous answer (an identical `revise`, so the
  stalemate rule applies); in round 1 the peer's answer becomes provisional with `agreed = False`,
  `needs_review = True`, `agreement_mode = "no_answer"`; both without answer → empty label, `needs_review`.
* Normalisation/comparison: `norm_label(cat)` (lower-case, whitespace/punctuation collapsed, cue order ignored),
  `labels_identical(a, b)`; `partition(tax)` (set of frozensets), `taxonomies_identical(a, b)` (same partition and,
  for matched categories, equal normalised names and `tone_category`).
* `Debate` engine (one class for both stages, D2 verbatim): round 1 both analysts; identical → `identical`;
  rounds 2–5 alternating A then B, each receiving its own latest + peer's latest answer; `accept` adopts the peer's
  parsed answer (`accepted_by_A/B`); `revise` identical to the analyst's own previous answer → `stalemate`; cap →
  `cap`, `agreed=False`, `needs_review=True`, Analyst A's last answer provisional. Records `round_iterations`,
  `agreement_mode`, both final answers, the full transcript (prompts, raw answers, decisions, seeds, timings).
  S3a is batched across all unresolved topics per (round, analyst) in sorted topic-id order; resume at topic
  granularity from `labels.jsonl` (unfinished debates restart from round 1).
* `run_labelling(bin_dir, docs, topics_df, topic_id_raw, llm, max_topics=None)` → `labels/topic_<id>.json`,
  `labels.jsonl`, `topics_labelled.csv` (`topic_id`, keywords, name, purpose, emotion_tone, `tone_category` via
  `label.tone_category`, description, inclusion_cues, round_iterations, agreed, agreement_mode, needs_review,
  analyst_a_final, analyst_b_final, llm_model, llm_revision, prompt_version, decoding — **no size columns**, which
  stay in `topics.csv` and are joined at collect, R4); writes the `topic_overrides.csv` template
  (`topic_id, name, needs_review, final_name, merge_into, notes`) if absent.
* `run_taxonomy(bin_dir, llm)` → reads `topics_labelled.csv` + `topic_overrides.csv`; builds items; if > 100 topics,
  k-means chunking on `topic_embeddings.npy` into `ceil(n/80)` chunks, per-chunk taxonomy, second pass over chunk
  categories, expansion back to topic ids (§7.4); validation → repair turn → deterministic repair (unknown ids
  dropped, duplicates kept in the first category, missing topics placed by cosine to the member-mean embedding;
  `category_assignment ∈ {agent, repaired, unplaced}`); writes `taxonomy.json`, `categories.csv`
  (`category_id, name, purpose, emotion_tone, tone_category, description, inclusion_cues, topic_ids, n_topics,
  n_posts, share, round_iterations, agreed, agreement_mode, needs_review`) and the `category_overrides.csv` template.
* Overrides (R5): `apply_topic_overrides(topics_labelled, documents, overrides)` — `final_name` replaces the agreed
  name; `merge_into` (chains resolved as in `label.apply_merges`) re-maps the posts' `topic_id` (`topic_id_raw`
  kept), removes merged-away topics from the S3b items and sums `n_cluster`/`n_total`; `notes` carried.
  `apply_category_overrides(categories, overrides)` — `final_name`, `merge_into` between categories;
  `category_id_agent` in `post_variables.csv` = the category before category overrides.

#### 4.6 `code/topicmodel/sentiment.py` (extend, S4)
`score_year(year) -> DataFrame(id, post_sentiment, post_sentiment_full, post_sentiment_label)` — title via
`clean_text`, full = the `document` text; 4 decimals; cached per year in `data/analysis/sentiment/<year>.csv`
(resumable), merged to `data/analysis/sentiment.csv`. `validation_sample(sent, period_bins, n=300)` (N9).
Existing `score_documents/by_topic/by_period` untouched (05 uses them).

#### 4.7 `code/topicmodel/tagging.py` (new, S5)
`collect(out_root, bins_available)`: `period_bins.csv` ⟕ per-bin `documents.csv` (`topic_id_raw, topic_id,
topic_assignment`) ⟕ topic→category map after overrides (`category_id`, `category_id_agent`) ⟕ `sentiment.csv` →
`data/analysis/post_variables.csv` (11 columns, every id once) + `post_variables.json` sidecar (run ids per bin,
taxonomy version = hash of `categories.csv` files, software versions, `bins_complete`); checks (ids unique and
complete; `topic_id` ∈ bin's `topics.csv`; `category_id` ∈ bin's `categories.csv`; assignment consistent with
`topic_id`; scores in [−1, 1]; labels consistent with thresholds). Aggregates `results/analysis/topic_labels_all_bins.csv`
(§7.7 columns, one row per topic, `period_bin` first; sizes joined from each bin's `topics.csv`, R4) and
`categories_all_bins.csv`; writes an empty `results/analysis/category_crosswalk.csv` template
(`period_bin, category_id, category_name, chapter_theme, notes`) if absent (F4, R13).

#### 4.8 `code/06-analysis-pipeline.py` (new; the stage CLI)
`--stage {bins,embed,topics,reassign,label,taxonomy,sentiment,collect,report}`, `--bin N` (or `--bins 1-9`),
`--device`, `--embedding-model` (default mpnet), `--llm`, `--sampling greedy|seeded`, `--max-docs N` (guard off,
subsample seeded 42), `--max-topics N` (label only the N largest; smoke tests), `--outlier-threshold T`,
`--years` (sentiment subset), `--out-root PATH` (default repo; redirects `results/analysis` **and** `data/analysis`,
N10/R9), `--allow-partial` (collect), `--force` (re-run despite existing output). Same header/logging style as
`05-topic-model.py`; per bin `log.txt` and `config.json` (effective settings, guard trials, backend, CUDA/cuML
versions, `pip freeze`, prompts, prompt version, elapsed). Every stage idempotent (sentinel file check) and
resumable. `--stage report` renders `report.html` per bin through the existing `report.write_report` only (R13),
with kick-off/final markers for tournament bins and the label/category columns.

#### 4.9 `code/06b-analysis-pipeline-colab.ipynb` (new; orchestration only) + `code/requirements-colab.txt` (N5)
Cells, following design §10: (1) Drive mount test with `gdown` fallback, clone/pull the public repo, copy
`data/yearly datasets/` from Drive to `/content/data`; (2) `pip install --no-deps` of the five pinned packages
(bertopic 0.17.4, umap-learn 0.5.12, hdbscan 0.8.44, sentence-transformers 6.1.0, vaderSentiment 3.3.2) listed in
`requirements-colab.txt`, then `pip check`, assert `import cuml`, `torch.cuda.is_available()`, print GPU + resolved
versions, enable cuML's fallback logging (R8); (3) `--stage bins`; (4) loop bins:
`--stage embed`, `python -m cuml.accel code/06-analysis-pipeline.py --stage topics --bin N`; (5) `--stage label`
per bin → **review pause** markdown (what to read, where `topic_overrides.csv` is); (6) `--stage taxonomy` → review
pause; (7) `--stage sentiment`; (8) `--stage collect`; (9) copy `results/analysis/` + `data/analysis/` to Drive /
back to the repository. Results, embedding cache and `HF_HOME` live on Drive
(`MyDrive/fifa-reddit-communities/…`). Cannot be executed from this Mac; it is validated in session 06 (first item
of the runbook). The notebook is built with `nbformat` so it opens in VS Code's Colab extension.

#### 4.10 `code/07-tag-corpus.R` and `code/08-validate-analysis.R` (new, S6)
* **07**: header block in the 02/03 style; reads `data/analysis/post_variables.csv` once, then the master, then the
  17 yearly files one at a time (`rm(); gc()` between files, R12), all as character (`read_chr`); pre-flight: sidecar
  `bins_complete` = 9 (unless `TAG_ALLOW_PARTIAL=1` and `TAG_OUT_ROOT` set), every master id present once in
  `post_variables`, 42 expected columns; records per file and per column a `digest::digest(as.character(col),
  algo = "xxhash64")` of the 42 original columns into `data/data artefacts/tagging_fingerprint.json`; `left_join` by
  `id`; appends the 8 columns in codebook §4.9 order; writes with `write_csv(na = "")` (master + yearly files; side
  files untouched); `TAG_IN_ROOT` / `TAG_OUT_ROOT` env overrides (dry runs).
* **08**: checks A1–A12 with the `check()` PASS/FAIL idiom and `stop()` on failure: row counts unchanged; 50 columns
  in master and yearly, 43 in side files; `period_bin` consistent with the UTC date for every row; bin sizes sum to
  the corpus; `topic_id`/`category_id` exist in their bin's `topics.csv`/`categories.csv`; `topic_assignment`
  consistent with `topic_id` (outlier ⇔ −1, not_modelled ⇔ empty); sentiment in [−1, 1], label thresholds; the
  42-column digests (re-read with identical readr settings) equal the pre-write fingerprint, file by file; master and
  `post_variables.csv` identical on the 8 columns; per-year column digests of the yearly files equal those of the
  master's year subset (R12).

#### 4.11 Repository plumbing
`code/topicmodel/__init__.py` docstring (new modules, entry points 05 and 06); `.gitignore` (§3); `README.md` §1
tree + §4 + §5 commands; `memory/workflows.md` §6 (deleted runs) and §7 (real commands, dry-run recipe, out-root);
`docs/README.md` session-05 row.

---

### 5. Test plan — bin 1 dry run (local, this session)

Design §16: *local dry run on bin 1 (`--max-docs 3000`, MiniLM, guard disabled) → bin 1 in full on Colab*. The
first half is done here; the second is step 1 of session 06. All dry-run outputs go to `results/dry-runs/bin1-3000/`
(N10); the LLM part runs unattended.

**5.1 Fast unit checks (no GPU, no LLM, minutes)** — kept as `code/tests/test_pipeline_units.py` (pytest-free, run
with `python -m`): `bin_of` on boundary dates (2010-09-30 → 1, 2010-10-01 → 2, 2023-02-28 → 7, 2026-07-31 → 9);
`build_period_bins` on the real files reproduces the nine counts and Σ = 1,302,021; `FrozenUMAP`/`FrozenCluster`
through BERTopic on 1,000 synthetic embeddings (labels returned unchanged up to the size renumbering; saved mapping
is the identity, R1); guard bisection with a stub counter; `extract_json`/`validate_*`/`labels_identical`/
`taxonomies_identical`/deterministic repair on fixture answers (a partition with duplicates, a missing and an unknown
id); override chains incl. `merge_into` re-mapping of post ids (R5); `apply_outlier_threshold` at `T = 0`, `0.5`,
`1.0`; **the debate engine with a `StubLLM`** (scripted answers keyed by topic, analyst, round) covering every
`agreement_mode` — `identical`, `accepted_by_A`, `accepted_by_B`, `stalemate`, `cap`, `no_answer` — the repair turn,
truncation handling, resume from a partial `labels.jsonl`, and the S3b chunking fallback with ~130 fake topics (R6).

**5.2 End-to-end on the bin-1 subsample**
```bash
export HF_HOME=$PWD/.hf-cache TOKENIZERS_PARALLELISM=false
P=.venv/bin/python; S=code/06-analysis-pipeline.py; OUT=results/dry-runs/bin1-3000
MINI=sentence-transformers/all-MiniLM-L6-v2
$P $S --stage bins                                             # the real S0 once (all 17 files, ~3 min) -> data/analysis/period_bins*.csv, for the docs
$P $S --stage bins --out-root $OUT                             # the dry run's own copy (N10/R9)
$P $S --stage embed --bin 1 --embedding-model $MINI             # 2010.npz, 13,638 docs (~2 min on MPS); shared cache
$P $S --stage topics --bin 1 --max-docs 3000 --embedding-model $MINI --out-root $OUT
$P $S --stage reassign --bin 1 --outlier-threshold 0.5 --out-root $OUT     # mechanics only; T is calibrated in 06
$P $S --stage label --bin 1 --llm Qwen/Qwen3-1.7B --out-root $OUT         # ~20–30 min unattended, batch 4 on MPS
$P $S --stage taxonomy --bin 1 --out-root $OUT
$P $S --stage sentiment --years 2010 --out-root $OUT
$P $S --stage collect --allow-partial --out-root $OUT
$P $S --stage report --bin 1 --out-root $OUT
## S6 on a mini corpus (2010 yearly file as master + yearly; post_variables restricted to it)
TAG_IN_ROOT=$OUT/r-mini TAG_OUT_ROOT=$OUT/r-out TAG_ALLOW_PARTIAL=1 Rscript code/07-tag-corpus.R
TAG_OUT_ROOT=$OUT/r-out Rscript code/08-validate-analysis.R
```
Then one resume test (`Ctrl-C` during `--stage label`, re-run, verify it continues) and one `--sampling seeded`
smoke run on 5 topics (`--max-topics 5`) to exercise N4 (not an acceptance criterion, R13).

**5.3 Acceptance criteria for "session 05 done"**
* every stage runs end-to-end on the subsample; re-running a finished stage is a no-op; `--force` re-runs;
* output files and columns match design §6.4, §7.7, §9.1 and codebook §4.9 exactly (checked by a schema assertion
  in `tagging.collect` and by 08);
* `period_bins.csv` reproduces the nine design counts; `config.json` holds settings, guard trials, backend
  (`cpu` locally), versions, prompts v1.1;
* all S3a answers parse or are flagged; `round_iterations`, `agreement_mode` and transcripts present for every
  topic; S3b partition valid (or repaired with `category_assignment` recorded);
* 2010 themes visible in the subsample keywords (vuvuzela, Jabulani, Spain, Lampard, Suárez, octopus) — a sanity
  check only, not a gate;
* 08 passes on the mini corpus; the 42-column fingerprint check works;
* nothing written under `results/analysis/`; under `data/` only `data/analysis/period_bins.csv` and
  `period_bins_summary.csv` (the real S0); the MiniLM cache under `results/topic-models/embeddings/` (gitignored).

---

### 6. Documentation and memory (last commit)

* `docs/session-05_<date>/00_approvals.md` (your decisions on F4/F6/F18 + deletion scope; N1–N10 and R1–R13 with
  why), `01_implementation-notes.md` (module map, how each design section maps to code, deviations),
  `02_dry-run-log.md` (commands, counts, timings, issues, resume test), `03_runbook-session-06.md` (step-by-step:
  Drive upload, open 06b in VS Code → Colab kernel, the install/cuML check first (R8), bin 1 full, order
  S2 → calibrate → reassign → S3a (R4), gates §12 with the exact files to look at, what co-authors review, then
  bins 2–9 → S5 → S6, and the `03-data-validation.R` 50-column update that S6 requires).
* Design doc: status line + §14 rows for the new decisions + a short §18 "Implementation notes (session 05)"
  covering the deviations (S3b token budget 8,000 — R2; `agreement_mode` value `no_answer` — R7; reassignment as
  its own stage — N1; frozen labels and the saved mapping — N2/R1); version → 1.1 **[default]**.
* Codebook: no new variables (F6/F18 approved as designed). Two wording touch-ups in §4.9: the audit-column
  sentence for `topic_id_raw` ("the HDBSCAN cluster, re-numbered by size — i.e. the BERTopic id before outlier
  reassignment", R1) and `agreement_mode` gains `no_answer` (R7). Version stays 3.0 draft.
* `memory/2026-10-<dd>_session-05.md`, `memory/workflows.md` §6–§7, `README.md`, `docs/README.md`; the auto-memory
  project status file updated (implementation done, run pending).

---

### 7. Verification summary (how you can check the result)

1. `git log` shows three commits: housekeeping, implementation, dry run + docs; `git status` clean.
2. `python code/tests/test_pipeline_units.py` prints all PASS.
3. `results/dry-runs/bin1-3000/bin-1/` contains the §6.4/§7.7 files; open `report.html` and
   `topics_labelled.csv`; `labels/topic_<id>.json` shows prompts v1.1, both analysts, decisions and rounds.
4. `data/analysis/period_bins_summary.csv` equals the design §4 table.
5. `Rscript code/08-validate-analysis.R` on the mini corpus prints `ALL CHECKS PASSED`.
6. `docs/session-05_<date>/03_runbook-session-06.md` is sufficient to run bin 1 on Colab without this conversation.

### 8. Rough effort

Housekeeping 0.5 h · config/bins/model 1.5 h · prompts/agents 2.5 h · sentiment/tagging/CLI 1.5 h · notebook +
requirements 0.5 h · R scripts 1 h · unit checks + dry run 2 h (≈ 45 min unattended compute) · docs/memory 1 h.
If the session has to be split, the natural break is after the implementation commit (before §5).

### 9. Decisions taken in this planning round

| Item | Decision | By | Date |
|---|---|---|---|
| F4 | Approved as designed: manual category crosswalk; S3c stays a hook; empty crosswalk template emitted at collect | Alan Nemirovski | 2026-10-23 |
| F6 | Approved as designed: bin boundaries exactly as in the brief; no `in_tournament` column | Alan Nemirovski | 2026-10-23 |
| F18 | Approved as designed: `post_sentiment` on the title; `post_sentiment_full` kept | Alan Nemirovski | 2026-10-23 |
| Deletion scope | Remove the two 2010 run folders (git) and the MiniLM embedding cache; keep the exploratory code and the historical docs | Alan Nemirovski | 2026-10-23 |
| N1–N10, R1–R13 | Implementation refinements, accepted with this plan (no change to D1–D4 or to the eight codebook variables) | Alan Nemirovski (plan approval) | 2026-10-23 |
| Test depth in session 05 | Local dry run on the bin-1 subsample (design §16 first half) + unit checks; bin 1 in full on Colab is step 1 of session 06 | plan default | 2026-10-23 |
