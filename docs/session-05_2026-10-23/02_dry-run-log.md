# Session 05 (2026-10-23) — Dry-run log: bin 1 (1 Jan – 30 Sep 2010) on the local machine

**Purpose.** Exercise every stage of `code/06-analysis-pipeline.py` end to end on bin 1 before the Colab run
(design §16: "local dry run on bin 1 → bin 1 in full on Colab"). Machine: Apple M4, 16 GB, `.venv` (Python 3.14.4,
bertopic 0.17.4, transformers 5.19.0, torch 2.14.1 on MPS). Embeddings: `all-MiniLM-L6-v2` (the local test model; the
Colab run uses mpnet). Language model: `Qwen/Qwen3-1.7B` at revision `70d244cc…`, greedy, batch 4 on MPS. All outputs
under `results/dry-runs/` (gitignored); nothing was written to `results/analysis/`; `data/analysis/` received only the
real S0 output (`period_bins.csv`, `period_bins_summary.csv`).

## 0. Unit checks (no GPU, no model)

`.venv/bin/python code/tests/test_pipeline_units.py` → **13/13 PASS** (bin boundaries; topic-count guard; frozen UMAP /
cluster components inside BERTopic 0.17.4 with the size renumbering; JSON extraction and validation; normalisation and
partition comparison; deterministic repair; overrides incl. merge chains; outlier thresholds; the full D2 debate engine
with a scripted stub model — every `agreement_mode`, repair and no-answer paths, resume; the S3b chunking fallback with
130 fake topics; a partial S5 collect; sentiment labels). Two defects found and fixed on the way: a NaN-handling bug in
`tagging.load_bin_outputs` and a wrong test expectation.

## 1. Stage S0 on the real corpus (12:00)

`--stage bins` read the 17 yearly files in ≈ 3 min and reproduced the design table exactly — all nine bin sizes equal
the counts computed on 2026-10-22 (10,002 / 186,988 / 83,147 / 331,216 / 88,648 / 305,653 / 41,438 / 217,303 / 37,626;
Σ = 1,302,021). `data/analysis/period_bins_summary.csv` also holds the shares of self posts, posts with body, match
threads, `lang = en`, short titles and the starting `min_cluster_size` per bin.

## 2. The 3,000-post subsample (design §16's `--max-docs 3000`) — 12:05–12:07

| Step | Result |
|---|---|
| `--stage embed --bin 1` (MiniLM, MPS) | 13,638 documents of 2010 encoded in 22 s → `results/topic-models/embeddings/…/2010.npz` (cache shared with the exploratory tool) |
| `--stage topics --bin 1 --max-docs 3000` | 3,000 posts sampled (seed 42), 2,945 modelled, 55 too short; UMAP 0.3 min; guard disabled; **HDBSCAN at `min_cluster_size` 25 / `min_samples` 10 returned 2 topics** (one of 2,859 posts, one vuvuzela topic of 40) with 1.6 % outliers; 0.6 min in total; all files of design §6.4 written (plus `hdbscan_to_topic_map.json`, `topic_embeddings.npy`, `outlier_nearest.csv`, `outlier_calibration/` with the quantile table and three samples) |
| `--stage reassign` | refused: the stage re-derived the subsample and found a different document set. **Fixed:** `reassign` now takes its document set from `documents.csv` (no `--max-docs` needed) |

**Finding D1 (for the author).** The 2-topic result is HDBSCAN's excess-of-mass (EOM) cluster selection choosing the
root cluster, not a code defect: on the same 5-d array, `min_cluster_size` 15 → 4 topics, **10 → 65 topics with 40 %
outliers** (the 2010-test picture), `leaf` selection at 25/10 → 29 topics with 52 % outliers. A 3,000-post sample is
too small for the design's relative minimum (25 posts = 0.85 % of the sample instead of 0.25 %), so the subsample is
not a meaningful test of the modelling stage; the dry run was therefore repeated on the **full bin 1** (§3), where the
guard runs as designed. Two consequences worth deciding at the quality gates rather than now: (i) EOM collapse is not
monotonic in `min_cluster_size`, so a bin that collapses at the floor of 25 cannot be rescued by the guard (it only
moves upward from the floor) — if this happens on a real bin, lowering `BIN_MIN_CLUSTER_FLOOR` for tournament bins or
switching to `cluster_selection_method = "leaf"` are the options; (ii) the collapse test (largest topic share) should
be read alongside the topic count when judging gate 1. Both are recorded here for the session-06 decisions log.

## 3. The full bin 1 (10,002 posts) — 12:10 onwards, `results/dry-runs/bin1-full/`

### 3.1 S2 topics (`--stage topics --bin 1`, MiniLM, CPU UMAP/HDBSCAN) — 1.9 min

| Quantity | Value |
|---|---|
| documents / modelled / too short | 10,002 / 9,809 / 193 |
| UMAP 9,809 × 384 → 5 | 0.4 min |
| guard | start `min_cluster_size` = 25 → **92 topics, 35.9 % raw outliers; converged at the first trial** (range 50–100) |
| BERTopic renumbering | 35 ids swapped among equal-size topics (recorded in `hdbscan_to_topic_map.json`; `topic_id_raw` is the BERTopic id) |
| largest topic | 284 posts (2.9 % of modelled) — no EOM collapse at this size |
| 2010 themes in the top-15 keywords | vuvuzela (topic 6), Lampard (32, 62, 63), Suárez / handball (66, 74), Spain (4, 12, 68), Paul the octopus (31, 79); *Jabulani* does not appear in any keyword list (it did in the session-02 window run with 93 topics; the bin covers nine months instead of five, so the ball sits inside broader World Cup topics) |
| outlier calibration table | candidates at the 50th / 65th / 80th percentile of outlier similarity: **T = 0.535 / 0.575 / 0.621** → 1,764 / 1,234 / 705 of the 3,523 outliers would be reassigned, leaving 17.9 / 23.3 / 28.7 % outliers; three 100-post judgement samples written |
| files | all of design §6.4 plus `hdbscan_to_topic_map.json`, `topic_embeddings.npy`, `outlier_nearest.csv`, `outlier_calibration/`, `config.json` (settings, guard trials, backend `cpu`, versions, prompts v1.1), `report.html` |

Comparison with the session-02 window run (7,678 docs, `min_cluster_size` 19): 93 topics, 36.6 % raw outliers — the bin
model is on the same scale, as the design expected.

### 3.2 `--stage reassign --bin 1 --outlier-threshold 0.5` — 20 s (mechanics only; `T` is calibrated in session 06)

2,151 of 3,523 outliers reassigned; 14.0 % outliers remain; `documents.csv` (10,002 rows: 6,286 `cluster`, 2,151
`reassigned`, 1,372 `outlier`, 193 `not_modelled`), `topics.csv` (`n_cluster` unchanged, `n_total` updated),
`topics_over_time.csv` (40 weekly periods) and `report.html` rewritten; `config.json → reassign` records the value.

### 3.3 S3a labelling of the 12 largest topics (`--max-topics 12`) — 6.3 min

Qwen3-1.7B (revision `70d244cc…`), bfloat16 on MPS, batch 4, greedy. 42 prompts, 367 s → **8.7 s per prompt**
(session 02, unbatched: ≈ 15 s). No truncated answer, every answer parsed. Outcomes: `identical` 2 (topics 8, 9),
`accepted_by_A` 8, `accepted_by_B` 1, `stalemate` 1 (topic 1: both analysts name it "World Cup Predictions" but differ
in cues/description, and Analyst A repeated its own answer in round 2 — the design's stalemate rule, D2);
`round_iterations` 1 / 2 / 3 for 1 / 10 / 1 topics; `needs_review` 1 of 12 (8 %). Names: Arsenal vs Liverpool ·
World Cup Predictions · EPL Support · World Cup Match Predictions · FIFA Livestream Matches · World Cup Streaming ·
Vuvuzela Complaints · American Football Interest · Referee Criticism · World Cup Pics · Real Madrid Transfer Rumors ·
jersey. Extrapolation for all 92 topics on this machine: ≈ 1 h (the A100 estimate in design §10 stands).
`labels/topic_<id>.json` holds the 40 sample posts' ids, both analysts' final answers and the full transcript;
`topics_labelled.csv` and the `topic_overrides.csv` template were written.

**Observation for gate 4.** "Identical" requires all five fields to match after normalisation; with greedy decoding
the typical path is *A accepts B's answer in round 2*. The stalemate rule can fire even when the two names agree
(topic 1). If `needs_review` exceeds 15 % in the Colab run, the author may want to decide whether agreement on name
+ tone category should count as agreement (a D2 refinement), rather than relax the cap.

### 3.4 S3b taxonomy — first attempt failed, recovery added (N11)

The first run (`--stage taxonomy --bin 1`, 5.2 min) ended with **`no_answer`: 0 categories, 12 topics `unplaced`**.
Transcript: both analysts, in round 1 and in the repair turn, returned the same 4-category partition (World Cup
Predictions and Streaming · World Cup Match Predictions · Vuvuzela Complaints · … ) with every category wrapped in its
own list (`}], [{` between categories, `]]]` endings) — syntactically invalid JSON, semantically a valid partition;
≈ 1,000 tokens, 64–80 s per answer on MPS, never truncated. `collect` then crashed on the empty `categories.csv`.
Fixes: tolerant recovery of such answers behind `config.LLM_JSON_SALVAGE` (strict parse first; recorded as
`parsed_via = "salvage"` in the transcript; **N11 in `00_approvals.md`, for the author's confirmation**), and an empty
taxonomy is now non-fatal (bin state `labelled`, all topics `unplaced`). Two unit checks added (15/15 pass).

### 3.5 S3b taxonomy, second run (`--stage taxonomy --bin 1 --force`, with recovery) — 6.5 min

Round 1: both analysts' answers were recovered (`parsed_via = salvage`) and failed the partition check (duplicated
ids) → repair turns (recovered again; A's repair now missed one id, B's still duplicated); round 2: Analyst A's revise
parsed as plain JSON and was identical to B's answer → **agreed (`identical`, 2 rounds)**; 5 of the 6 answers needed the
recovery step. Deterministic repair: duplicated ids 9, 10, 11 kept in the first category listed, topic 0 (Arsenal vs
Liverpool) placed by embedding similarity into cat_4 (`category_assignment = repaired`, `n_repaired = 1`); nothing
unplaced. Result, six categories: World Cup Predictions and Streaming (topics 1, 4, 5, 9; 906 posts) · World Cup
Match Predictions (3) · Referee Criticism (8) · EPL Support and Club Engagement (2, 10, 11, 0; 881) · Vuvuzela
Complaints (6) · American Football Interest (7). `taxonomy.json` holds the full transcript and the repair log;
`categories.csv`, `topic_categories.csv` and the `category_overrides.csv` template were written.

### 3.6 S4 sentiment (`--years 2010`, 3 s) and S5 collect (`--allow-partial`, 7 s)

Sentiment: 13,638 posts of 2010 scored (2,281 with a body); bin 1 labels: neutral 4,620 / positive 3,527 / negative
1,855; validation sample of 299 posts over 12 strata (1 bin × 3 labels × `is_self`) with blank coder columns.
Collect: `post_variables.csv` with **1,302,021 rows and the 11 columns** of design §9.1; bin 1 fully populated
(6,286 `cluster`, 2,151 `reassigned`, 1,372 `outlier`, 193 `not_modelled`; 2,583 posts with a category, i.e. the
posts of the 12 labelled topics; sentiment for all 10,002), bins 2–9 empty; sidecar: `bins_complete = [1]`,
`complete = false`, taxonomy version `a910fbe68c95`, no check failures. `topic_labels_all_bins.csv` (12 rows, 27
columns, sizes joined from `topics.csv`), `categories_all_bins.csv` (6 rows) and the empty `category_crosswalk.csv`
template written. The report (`report.html`) re-rendered with labels and categories.

### 3.7 Stage S6 on a mini corpus (`07-tag-corpus.R`, `08-validate-analysis.R`) — 12:38

Mini root `r-mini/` = the 2010 yearly file (13,638 rows; it also contains the 3,636 posts of October–December 2010,
i.e. bin 2, which has no model run) + the dry run's `post_variables.csv`/`.json`; output root `r-out/`.
* `TAG_ALLOW_PARTIAL=1 Rscript code/07-tag-corpus.R`: the partial table was accepted only because the output root
  differs from the input root; 13,638 rows tagged in 4 s (9,809 with a topic, 2,583 with a category, 13,638 with
  sentiment); 50 columns in codebook order; per-column fingerprints and `tagging_summary.csv` written.
* `Rscript code/08-validate-analysis.R`: first run **failed A7** because the untagged bin-2 rows have an empty
  `topic_assignment` (correct for a partial table). Fix: A7 applies to tagged rows; new A7b asserts that untagged rows
  occur only in bins without a model run. Second run: **ALL CHECKS PASSED** (A1b, A2a, A2b, A3, A4a–c, A5, A6, A7,
  A7b, A8a, A8b, A9b, A10; A2c skipped — no side files in the mini root; A11 skipped — no master; A12 noted as
  incomplete). The 42 original columns of the tagged file are byte-identical to the input (A9b).

### 3.8 Full labelling of all 92 topics and the taxonomy at scale — started 12:39, `results/dry-runs/bin1-full92/`

A copy of the 12-topic run resumed with `--stage label --bin 1` (80 topics to go; ≈ 1 h on this machine), then
`--stage taxonomy --bin 1 --force` on all 92 items (the realistic S3b test: ≤ 100 items, 8,000-token budget),
`collect`, `report`. Resumable at any time (`labels.jsonl`). Results are appended below when available.

## 4. Acceptance criteria of the plan (§5.3) — status

| Criterion | Status |
|---|---|
| every stage runs end to end; re-running a finished stage is a no-op; `--force` re-runs | **met** on the full bin 1 (all nine stages + S6 in R); no-op verified for `bins`, `embed`, `topics`; `--force` used for `taxonomy` |
| output files and columns match design §6.4, §7.7, §9.1 and codebook §4.9 | **met** (`post_variables.csv` 11 columns; `topic_labels_all_bins.csv` 27 columns incl. the §7.7 set; `categories_all_bins.csv`; per-bin files of §6.4 plus the documented additions) |
| `period_bins.csv` reproduces the nine design counts; `config.json` holds settings, guard trials, backend, versions, prompts v1.1 | **met** |
| all S3a answers parse or are flagged; rounds, modes and transcripts present; S3b partition valid or repaired with `category_assignment` | **met** (12 topics: all parsed; 1 `needs_review`; S3b valid after one embedding-based placement) — S3b needed the N11 recovery |
| 2010 themes visible in the keywords (sanity) | **met** except *Jabulani* (see §3.1) |
| 08 passes on the mini corpus; fingerprint check works | **met** after the A7 fix |
| nothing written under `results/analysis/`; under `data/` only the real S0 output and the MiniLM cache | **met** (`results/analysis/` exists only as an empty folder created by the program's path setup during the real S0 run — no files; `data/analysis/` holds `period_bins.csv`, `period_bins_summary.csv`, `pipeline-log.txt`) |
| the design's `--max-docs 3000` subsample run | run, but degenerate (§2, finding D1); the full-bin run replaces it as the reference dry run |

## 5. Defects found and fixed during the dry run

1. `tagging.load_bin_outputs`: NaN handling when mapping overridden topic ids (unit test).
2. `--stage reassign` re-derived the document set instead of reading `documents.csv` (failed on the subsample run).
3. Qwen3-1.7B's bracket pattern in S3b answers → tolerant recovery behind `LLM_JSON_SALVAGE` (**N11, to confirm**).
4. An empty taxonomy crashed S5; now non-fatal (bin state `labelled`).
5. `08-validate-analysis.R` A7 too strict for partial tables (A7 restricted to tagged rows, A7b added).
6. (Operational) zsh does not word-split quoted variables; the chain scripts run under bash.

## 6. Timings on the M4 (MiniLM, MPS/CPU)

S0 all 17 files ≈ 3 min · embed 2010 (13,638 docs) 22 s · S2 bin 1 1.9 min (UMAP 0.4 min) · reassign 20 s ·
S3a 12 topics 6.3 min (8.7 s per prompt, batch 4) · S3b 12 items 6.5 min (65–80 s per answer) · S4 2010 3 s ·
S5 7 s · 07 4 s · 08 4 s · unit checks ≈ 1 min.

