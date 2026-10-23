# Workflows (how to re-run things)

All commands run from the repository root
(`…/SHARED FOLDER - FIFA Chapter/fifa-reddit-communities`). The repository is not under git; it is
synchronised through OneDrive, so large rewrites take a few minutes to sync.

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

## 6. Topic modelling (BERTopic; Python)

Environment (machine-local, rebuild once per machine):
```bash
python3 -m venv .venv && .venv/bin/pip install -r code/requirements-topicmodel.txt
```
Runs (from the repository root; weights cache in `.hf-cache/`, results in `results/topic-models/<run>/`):
```bash
export HF_HOME=$PWD/.hf-cache TOKENIZERS_PARALLELISM=false
.venv/bin/python code/05-topic-model.py --run window --worldcup 2010          # tournament window (±2 months)
.venv/bin/python code/05-topic-model.py --run year --year 2010                # calendar year
.venv/bin/python code/05-topic-model.py --run era --era 2010                  # align the era's year models (run years first)
.venv/bin/python code/05-topic-model.py --label-only results/topic-models/window-wc2010 --llm Qwen/Qwen3-1.7B   # LLM labels, resumable
.venv/bin/python code/05-topic-model.py --run year --year 2010 --apply-merges # after editing topic_merge_table.csv
```
All settings: `code/topicmodel/config.py` (each run also writes its effective settings to `config.json`).
Google Colab (GPU, full corpus): `code/05b-topic-model-colab.ipynb` — upload `code/` and `data/yearly datasets/` to Drive.
Design and plain-language explanation: `docs/session-02_2026-10-13/02_topic-modelling-design.md`.
These year/window/era runs are now **exploratory**; the chapter's analysis uses the period-bin pipeline of §7.
The 2010 test outputs (`window-wc2010/`, `year-2010/`) and the MiniLM embedding cache were deleted on 2026-10-23
(author's decision, session 05); re-run the commands above to regenerate them if ever needed.

## 7. Analytical pipeline (period bins → BERTopic → two-agent labels → VADER → tagging) — implemented in session 05

Design (settings, prompts, outputs, flags, decisions): `docs/session-04_2026-10-22/01_analytical-pipeline-design.md`.
Approvals, implementation notes, dry-run log and the runbook for the run session: `docs/session-05_2026-10-23/`.
Variables it adds: codebook §4.9. Program: `code/06-analysis-pipeline.py` (one `--stage` per call; every stage skips
when its output exists, `--force` re-runs; per bin `results/analysis/bin-N/config.json` + `log.txt`).

```bash
export HF_HOME=$PWD/.hf-cache TOKENIZERS_PARALLELISM=false
P=.venv/bin/python; S=code/06-analysis-pipeline.py
$P code/tests/test_pipeline_units.py                 # 13 unit checks (no GPU, no LLM), ~1 min
$P $S --stage bins                                   # S0: data/analysis/period_bins.csv (+ summary; checks Σ = 1,302,021)
$P $S --stage embed --bin 1                          # S1: mpnet embeddings, cached per year (GPU)
python -m cuml.accel $S --stage topics --bin 1       # S2: BERTopic per bin (GPU on Colab; same command without the prefix on CPU)
$P $S --stage label --bin 1                          # S3a: two-agent cluster labelling (resumable via labels.jsonl)
#   -> co-author review: results/analysis/bin-1/topics_labelled.csv, report.html -> fill topic_overrides.csv
$P $S --stage taxonomy --bin 1                       # S3b: per-bin taxonomy (re-run with --force after any S3a override)
#   -> co-author review: categories.csv -> fill category_overrides.csv
$P $S --stage reassign --bin 1 --outlier-threshold T # D3: apply the calibrated T (seconds; see outlier_calibration/)
$P $S --stage sentiment                              # S4: data/analysis/sentiment.csv (+ sentiment_validation_sample.csv)
$P $S --stage collect                                # S5: data/analysis/post_variables.csv (+ .json), topic_labels_all_bins.csv, categories_all_bins.csv
Rscript code/07-tag-corpus.R && Rscript code/08-validate-analysis.R   # S6 (local): append the 8 columns (v3.0) + 12 check families
```
Options: `--bins 1-9` (loop), `--device auto|cuda|mps|cpu`, `--embedding-model <id>`, `--llm <id>`, `--sampling greedy|seeded`,
`--max-topics N` (smoke test), `--years 2010 2011` (sentiment subset), `--allow-partial` (collect before all bins exist).
Order matters: S3a review → S3b → overrides → S5 → S6; S6 refuses to write unless `post_variables.json` says `complete`.
On Colab: `code/06b-analysis-pipeline-colab.ipynb` runs the same commands (A100 preferred; data copied from Drive to
`/content`; results, embedding cache and `.hf-cache` symlinked to Drive). Step-by-step: `docs/session-05_2026-10-23/03_runbook-session-06.md`.

**Local dry run on bin 1** (never touches `results/analysis/` or the real `data/analysis/`). Run it on the **full bin**
(10,002 posts, ≈ 2 min for S2 with MiniLM): a 3,000-post subsample (`--max-docs 3000`) collapses to 2 topics under
HDBSCAN's excess-of-mass selection at the floor of 25 (session-05 dry-run log, finding D1), so it only tests mechanics.
```bash
OUT=results/dry-runs/bin1-full; MINI=sentence-transformers/all-MiniLM-L6-v2
mkdir -p $OUT/data/analysis && cp data/analysis/period_bins*.csv $OUT/data/analysis/   # or: --stage bins --out-root $OUT
$P $S --stage embed --bin 1 --embedding-model $MINI --out-root $OUT
$P $S --stage topics --bin 1 --embedding-model $MINI --out-root $OUT                     # add --max-docs N only for mechanics tests (guard off)
$P $S --stage reassign --bin 1 --outlier-threshold 0.5 --embedding-model $MINI --out-root $OUT
$P $S --stage label --bin 1 --out-root $OUT            # Qwen3-1.7B on MPS, batch 4; --max-topics 5 for a quick smoke test
$P $S --stage taxonomy --bin 1 --out-root $OUT
$P $S --stage sentiment --years 2010 --out-root $OUT
$P $S --stage collect --allow-partial --out-root $OUT
# S6 on a mini corpus: the 2010 yearly file + the dry run's post_variables, written to a separate root
mkdir -p $OUT/r-mini/data/yearly\ datasets $OUT/r-mini/data/analysis
cp "data/yearly datasets/soccer.data.cleaned-2010.csv" "$OUT/r-mini/data/yearly datasets/"; cp $OUT/data/analysis/post_variables.* $OUT/r-mini/data/analysis/
TAG_IN_ROOT=$OUT/r-mini TAG_OUT_ROOT=$OUT/r-out TAG_ALLOW_PARTIAL=1 Rscript code/07-tag-corpus.R
TAG_IN_ROOT=$OUT/r-mini TAG_OUT_ROOT=$OUT/r-out ANALYSIS_DIR=$OUT/results/analysis Rscript code/08-validate-analysis.R
```
