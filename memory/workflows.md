# Workflows (how to re-run things)

All commands run from the repository root. The repository is under git (GitHub `alannemi/fifa-reddit-communities`,
since 2026-10-14); the author's working copy lives in Dropbox, and the `data/` folder is shared through OneDrive
(`data/README.md`), so large rewrites of the data files take a few minutes to sync.

## 1. Rebuild the cleaned corpus from the raw export

```bash
Rscript code/02-data-cleaning.R          # ~15 min, ~6-8 GB RAM; overwrites outputs in place
```
Inputs: `data/soccer_posts.csv` (never modified). Outputs: `data/soccer_data.clean.csv`,
`data/yearly datasets/soccer.data.cleaned-YYYY.csv`, six side files (`deleted_`, `removed_`, `bot_threads`,
`promotional_`, `encoding_issue_`, `non_english_posts.csv`) and audit tables in `data/data artefacts/`
(`filter_summary.csv`, `spam_accounts.csv`, `piracy_accounts.csv`, `encoding_repairs.csv`).

Dry run on a subset without touching the real outputs:
```bash
CLEAN_NMAX=60000 CLEAN_OUT_ROOT=/tmp/testout Rscript code/02-data-cleaning.R
CLEAN_OUT_ROOT=/tmp/testout Rscript code/03-data-validation.R    # C1/C2/C5c will fail on a subset, by design
```

To change a rule: edit the CONFIG section (§1) of `code/02-data-cleaning.R` only, re-run, re-validate,
and record the change in a new `docs/session-NN_<date>/` folder and in the codebook's version history.

## 2. Validate

```bash
Rscript code/03-data-validation.R        # 28 checks, prints PASS/FAIL each; exits with error on any FAIL (needs encoding_repairs.csv from step 1)
```

## 3. Load the data for analysis (R)

```r
library(readr)
d <- read_csv("data/soccer_data.clean.csv", col_types = cols(.default = col_character()), na = character())
# then convert what you need, e.g.
d$num_comments <- as.integer(d$num_comments)
d$created_datetime <- as.POSIXct(d$created_datetime, tz = "UTC")
```
Never load these files with `data.table::fread()` default settings (it doubled every quotation mark in v1).
Python: `pandas.read_csv(path, dtype=str, keep_default_na=False)`.

## 4. Regenerate the codebook PDF

Source of truth: `data/codebook_soccer_posts.md`. Render:
```bash
Rscript -e 'html <- commonmark::markdown_html(readLines("data/codebook_soccer_posts.md"), extensions = TRUE); writeLines(c("<html><head><meta charset=\"utf-8\"><style>", readLines("docs/codebook.css"), "</style></head><body>", html, "</body></html>"), "/tmp/codebook.html")'
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" --headless --disable-gpu --no-pdf-header-footer --print-to-pdf="data/codebook_soccer_posts.pdf" /tmp/codebook.html
```
(The helper `code/04-render-codebook.R` wraps exactly these steps.)

## 5. Conventions for every script in `code/`

* Header block: purpose, why, design principles, how to run, outputs.
* Section banners and line-level comments explaining *why*, not just what.
* All CSV reads as character (`col_types = cols(.default = col_character()), na = character()`);
  all writes with `readr::write_csv(na = "")`.
* No row is ever discarded silently: filtered rows go to a side file with `filter_reason`.
* Literature cited only when real and verifiable; thresholds justified from the data.
* The R scripts 02, 03, 06 and 07 source `code/R/pipeline-utils.R` (logging, the PASS/FAIL check, the character-only
  reader, the per-column fingerprint, `find_root`, the 42-column and 8-column specifications, the bin table); run them
  from the repository root (`Rscript code/<script>.R`) or with the `.Rproj` open.

## 6. The session-02 exploratory tool (removed 2026-10-25)

`code/05-topic-model.py` (calendar-year, ±2-month tournament-window and era runs) and `05b-topic-model-colab.ipynb` were
the exploratory tool of session 02; the period-bin pipeline of §7 superseded them and they were removed from `code/` in
session 07 (decision H1; code in git history up to commit `1ad6e489`). Their design and the 2010 test case remain in
`docs/session-02_2026-10-13/` and `docs/session-03_2026-10-14/`.

## 7. The analytical pipeline (period bins → BERTopic → two-agent topic labels → VADER → tagging)

Design (settings, prompts, outputs, flags, decisions): `docs/session-04_2026-10-22/01_analytical-pipeline-design.md`.
Approvals, implementation notes, dry-run log: `docs/session-05_2026-10-23/`; the two-step taxonomy protocol (D5, superseded):
`docs/session-06_2026-10-24/`; the consolidated code, the path root and the Drive layout: `docs/session-07_2026-10-25/` (runbook
for the Colab run: `02_runbook-colab-run.md`); the first Colab run and decisions D7 (the topic is the unit of analysis, no
category layer) and D8 (topic density 50–75 per 10k posts): `docs/session-08_2026-10-25/`. Variables it adds: codebook §4.9.
Program: `code/05-analysis-pipeline.py` (one `--stage` per call; every stage skips when its output exists, `--force`
re-runs; per bin `config.json` + `log.txt`; every stage logs the effective paths). Package: `code/topicmodel/` (every
setting in `config.py`).

**Paths.** By default the repository layout: `data/yearly datasets` (input), `data/analysis` (join tables),
`results/analysis` (bin folders + aggregated tables), `results/embeddings` (embedding cache), `.hf-cache` (model weights).
`PIPELINE_ROOT=<folder>` re-roots everything that persists on the flat layout of the shared Drive folder
`MyDrive/fifa-communities/` (`yearly datasets/`, `analysis/`, `results/`, `embeddings/`, `hf-cache/`); `PIPELINE_YEARLY_DIR`
overrides the input folder alone (the Colab notebook points it at a local copy of the CSVs). `--out-root PATH` is for dry
runs: it writes `PATH/results/analysis` and `PATH/data/analysis` (repository-shaped, so the R scripts read them) and leaves
the input and the caches alone. `HF_HOME` is set by the program to the model cache when unset.

```bash
export TOKENIZERS_PARALLELISM=false
P=.venv/bin/python; S=code/05-analysis-pipeline.py
$P code/tests/test_pipeline_units.py                 # 12 unit checks (no GPU, no LLM), ~1 min
$P $S --stage bins                                   # S0: data/analysis/period_bins.csv (+ summary; checks Σ = 1,302,021)
$P $S --stage embed --bin 1                          # S1: mpnet embeddings, cached per year (GPU)
python -m cuml.accel $S --stage topics --bin 1       # S2: BERTopic per bin (GPU on Colab; same command without the prefix on CPU)
$P $S --stage label --bin 1                          # S3a: two-agent cluster labelling (resumable via labels.jsonl)
#   -> co-author review: results/analysis/bin-1/topics_labelled.csv, report.html -> fill topic_overrides.csv
$P $S --stage reassign --bin 1                       # D9: re-apply the outlier tiers from the persisted arrays (seconds; bins clustered before D9, or after a quantile change)
$P $S --stage sentiment                              # S4: data/analysis/sentiment.csv (+ sentiment_validation_sample.csv)
$P $S --stage collect                                # S5: data/analysis/post_variables.csv (+ .json; topic_uid/topic_name/topic_tone per post),
#                                                       topic_labels_all_bins.csv (with mean_vader_sentiment), topic_crosswalk.csv (template)
Rscript code/06-tag-corpus.R && Rscript code/07-validate-analysis.R   # S6 (local): append the 15 columns (v3.0) + 13 check families
$P $S --stage status                                 # one row per bin: state, topics, labelled, review_ready (= bin-N/REVIEW_READY.txt), reassigned
$P code/08-review-pack.py pack --bin 1   # review_pack.md for the co-authors (after each hand-back); `check --bin 1` validates the
#                                          filled override files before they go back to Drive (guide: docs/session-08_2026-10-25/01_...)
```
Options: `--bins 1-9` (loop), `--device auto|cuda|mps|cpu`, `--embedding-model <id>` (default `config.EMBEDDING_MODEL` = mpnet),
`--llm <id>`, `--sampling greedy|seeded`, `--max-topics N` (smoke test), `--years 2010 2011` (sentiment subset),
`--allow-partial` (collect before all bins exist), `--max-docs N` (mechanics test on a subsample; guard off), `--keep-umap`
(topics with --force: refit HDBSCAN only), `--grid '<json>'` (grid stage).
Order matters: S3a review (topic_overrides.csv) → S5 → S6; S6 refuses to write unless `post_variables.json` says `complete`.
On Colab: `code/05b-analysis-pipeline-colab.ipynb` runs the same commands (A100 preferred; the code cloned from GitHub on
the Colab machine, the CSVs copied to its local disk, every output written to `MyDrive/fifa-communities/` through
`PIPELINE_ROOT`; the hand-back zip rebuilds the repository paths). Step by step: `docs/session-07_2026-10-25/02_runbook-colab-run.md`.

**Local dry run on bin 1** (never touches `results/analysis/` or the real `data/analysis/`). Run it on the **full bin**
(10,002 posts, ≈ 2 min for S2 with MiniLM): a 3,000-post subsample (`--max-docs 3000`) collapses to 2 topics under
HDBSCAN's excess-of-mass selection at the floor of 25 (session-05 dry-run log, finding D1), so it only tests mechanics.
```bash
OUT=results/dry-runs/<name>; MINI=sentence-transformers/all-MiniLM-L6-v2        # = config.EMBEDDING_MODEL_DRY_RUN
mkdir -p $OUT/data/analysis && cp data/analysis/period_bins*.csv $OUT/data/analysis/   # or: --stage bins --out-root $OUT
$P $S --stage embed --bin 1 --embedding-model $MINI --out-root $OUT
$P $S --stage topics --bin 1 --embedding-model $MINI --out-root $OUT                     # 92 topics, 35.9 % raw outliers (2026-10-23 and 2026-10-25)
$P $S --stage reassign --bin 1 --outlier-threshold 0.5 --embedding-model $MINI --out-root $OUT
$P $S --stage label --bin 1 --out-root $OUT            # Qwen3-1.7B on MPS, batch 4; --max-topics 5 for a quick smoke test
$P $S --stage sentiment --years 2010 --out-root $OUT
$P $S --stage collect --allow-partial --out-root $OUT
# S6 on a mini corpus: the 2010 yearly file + the dry run's post_variables, written to a separate root
mkdir -p $OUT/r-mini/data/yearly\ datasets $OUT/r-mini/data/analysis
cp "data/yearly datasets/soccer.data.cleaned-2010.csv" "$OUT/r-mini/data/yearly datasets/"; cp $OUT/data/analysis/post_variables.* $OUT/r-mini/data/analysis/
TAG_IN_ROOT=$OUT/r-mini TAG_OUT_ROOT=$OUT/r-out TAG_ALLOW_PARTIAL=1 Rscript code/06-tag-corpus.R
TAG_IN_ROOT=$OUT/r-mini TAG_OUT_ROOT=$OUT/r-out TAG_ANALYSIS_DIR=$OUT/results/analysis Rscript code/07-validate-analysis.R
```
The dry runs of sessions 05–07 are under `results/dry-runs/` (their pipeline outputs are tracked since 2026-10-25; the
join tables, the R mini-corpus copies and the caches are not — `results/README.md`): `bin1-full` (12 labelled topics), `bin1-full92`
(all 92 topics; the single-prompt taxonomy that led to D5), `bin1-twostep` (stopped after round 1 of the definitions
debate), `h7-bin1` (session 07: S1 + S2 after the harmonization, identical UMAP array and HDBSCAN labels), `d6-bin1` (the D6
shortlist re-run of the taxonomy on the Colab outputs of bin 1). The taxonomy outputs in these folders are records of the
superseded protocol (D7, 2026-10-25).

## 8. Tone vocabulary, sentiment validation and descriptive figures (session 09, 2026-10-27)

```bash
# after any change of config.TONE_SET / TONE_STEMS / TONE_STEMS_ORDERED (no model call; only "other" rows may change)
.venv/bin/python code/05-analysis-pipeline.py --stage retone --bins 1-9
for b in 1 2 3 4 5 6 7 8 9; do .venv/bin/python code/08-review-pack.py pack --bin $b; done
.venv/bin/python code/05-analysis-pipeline.py --stage collect              # new labels_version in post_variables.json

Rscript code/09-sentiment-coding-pack.R      # blinded coder_A.csv / coder_B.csv in results/analysis/manual review/sentiment_coding/
Rscript code/10-sentiment-validation.R       # after both sheets are back (+ optional adjudicated.csv): agreement, kappa, confusion
Rscript code/11-descriptives.R               # ~1 min: figures + tables in results/descriptives/ (reads the yearly files + post_variables.csv)
Rscript code/12-topic-network.R              # ~20 s: topic network D1/D1b/D2 + family tables (topic embeddings via the .venv Python, reticulate)
```
`09` refuses to overwrite a sheet that already holds codes. `10` takes `SENT_CODING_DIR` / `SENT_OUT_DIR` for tests.
`11` does not depend on S6; re-run it after any change of the join table. Decisions and readings: `docs/session-09_2026-10-27/`.

## 9. The co-authors' QUALTAG tags (session 10, 2026-10-28)

```bash
git pull --no-rebase origin main             # the co-authors upload their review files on GitHub
Rscript code/13-recode-qualtag.R             # seconds: 11 qt_ columns after QUALTAG in every BIN<N>_topics_labelled_*.csv, + QUALTAG_to_settle.csv
```
Re-run after any change to a bin file in `results/analysis/manual review/`; it recomputes its own columns and leaves the rest of
each file byte for byte as it was. It stops if a bin has two files (remove the superseded one with `git rm`) or a tag lies
outside the scheme. Scheme, rules and decisions: that folder's `README.md` and `docs/session-10_2026-10-28/00_approvals.md`.
