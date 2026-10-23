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

## 7. Analytical pipeline (period bins → BERTopic → two-agent labels → VADER → tagging) — designed, not yet implemented

Design (settings, prompts, outputs, flags, decisions): `docs/session-04_2026-10-22/01_analytical-pipeline-design.md`.
Variables it adds: codebook §4.9. Implementation is scheduled for session 05; the intended commands are:
```bash
python code/06-analysis-pipeline.py --stage bins                      # period_bin for every post (checks Σ = corpus)
python code/06-analysis-pipeline.py --stage embed --bin 1             # mpnet embeddings, cached per year
python -m cuml.accel code/06-analysis-pipeline.py --stage topics --bin 1   # BERTopic per bin (GPU on Colab)
python code/06-analysis-pipeline.py --stage label --bin 1             # two-agent cluster labelling (resumable)
#   -> co-author review: results/analysis/bin-1/topic_overrides.csv
python code/06-analysis-pipeline.py --stage taxonomy --bin 1          # two-agent taxonomy of the bin
#   -> co-author review: category_overrides.csv
python code/06-analysis-pipeline.py --stage sentiment                 # VADER (title + full) for every post
python code/06-analysis-pipeline.py --stage collect                   # post_variables.csv + aggregated CSVs
Rscript code/07-tag-corpus.R && Rscript code/08-validate-analysis.R   # local: append the 8 columns (v3.0) + checks
```
On Colab: `code/06b-analysis-pipeline-colab.ipynb` runs the same commands (A100 preferred; data copied from Drive to
`/content`; results and caches on Drive). Order matters: S3a review → S3b → overrides → collect → write-back.
