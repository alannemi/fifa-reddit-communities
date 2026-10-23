# fifa-reddit-communities

Working repository of the **FIFA Communities chapter** (McGill University): a mixed-methods study of
discourse on r/soccer from 1 January 2010 to 31 July 2026, organised around the 2010, 2014, 2018, 2022 and
2026 FIFA World Cups. It holds the cleaned post corpus, its codebook, the cleaning and topic-modelling
code, and session-by-session documentation of every decision.

This README is the entry point for co-authors. Read it top to bottom once; afterwards the two places to
return to are `memory/workflows.md` (how to run things) and the latest `memory/<date>_session-NN.md`
(where we are).

---

## 1. What is in the workspace

```
fifa-reddit-communities/
├── README.md                     ← this file
├── data/                         ← the corpus (large; shared via OneDrive, NOT in GitHub)
│   ├── soccer_posts.csv              raw export, 2,047,322 posts, 64 columns (never modified)
│   ├── soccer_data.clean.csv         clean master, 1,302,021 posts, 42 columns (v2.1)
│   ├── yearly datasets/              the clean master split by year, 2010–2026 (same format)
│   ├── deleted_posts.csv, removed_posts.csv, bot_threads.csv, promotional_posts.csv,
│   │   encoding_issue_posts.csv, non_english_posts.csv     side files: every filtered post + its reason
│   ├── codebook_soccer_posts.md / .pdf   codebook v2.1 (read this before using any file)
│   └── data artefacts/               raw JSONL batches, conversion script, audit tables
│                                      (filter_summary, spam_accounts, piracy_accounts, encoding_repairs)
├── code/                         ← all scripts, numbered in pipeline order
│   ├── 01-data-inspection.R          load the raw file (readr; never fread — see codebook §5)
│   ├── 02-data-cleaning.R            rebuild every cleaned file from the raw export (7 filter stages)
│   ├── 03-data-validation.R          28 consistency and fidelity checks
│   ├── 04-render-codebook.R          codebook Markdown → PDF
│   ├── 05-topic-model.py             exploratory BERTopic runs (year / World-Cup window / era); superseded by 06
│   ├── 05b-topic-model-colab.ipynb   the exploratory tool on a Colab GPU
│   ├── 06-analysis-pipeline.py       THE ANALYTICAL PIPELINE: period bins → BERTopic per bin → two Qwen3 agents
│   │                                  → VADER → post_variables.csv (stages: bins, embed, topics, reassign, label,
│   │                                  taxonomy, sentiment, collect, report)
│   ├── 06b-analysis-pipeline-colab.ipynb   orchestration of 06 on a Colab GPU (A100; cuML)
│   ├── 07-tag-corpus.R               stage S6: append the 8 analysis columns to the master and yearly files
│   ├── 08-validate-analysis.R        12 check families on the tagged files (fingerprints of the original columns)
│   ├── topicmodel/                   the Python package behind 05 and 06 (config.py holds every setting;
│   │                                  bins, embed, model, prompts, agents, sentiment, tagging, report)
│   ├── tests/test_pipeline_units.py  13 unit checks that need no GPU and no language model
│   ├── requirements-topicmodel.txt   pinned Python environment (local)
│   └── requirements-colab.txt        the few packages installed on top of Colab's RAPIDS stack
├── results/analysis/             ← the analytical pipeline's outputs: bin-1 … bin-9/ (topics, labels, taxonomy,
│                                    transcripts, config.json, report.html) + topic_labels_all_bins.csv etc.
├── results/topic-models/         ← exploratory runs of 05 (the 2010 test outputs were removed on 2026-10-23)
├── results/dry-runs/             ← local dry runs of 06 (gitignored)
├── docs/                         ← documentation by working session (what was found, decided, run)
│   ├── README.md                     index of sessions
│   ├── codebook.css                  styling for the codebook PDF
│   └── session-NN_YYYY-MM-DD/        audit findings, rules and rationale, run logs, codebook changes,
│                                      topic-model design and test-case notes, progress reports
├── memory/                       ← short, dated summaries of each session + reusable workflows
│   ├── README.md, workflows.md
│   └── YYYY-MM-DD_session-NN.md
├── reddit-soccer-collector/      ← the data-collection tool (Arctic Shift + live scanner); see its README
├── fifa-reddit-communities.Rproj ← RStudio project (open this to get the right working directory)
├── .venv/, .hf-cache/            ← machine-local Python environment and model weights (not shared)
└── .gitignore                    ← keeps data, environments and model weights out of git
```

## 2. Where the data lives and how to get it

**Note for co-authors cloning from GitHub (https://github.com/alannemi/fifa-reddit-communities):** the
`data/` folder is about 14 GB and is **shared through the OneDrive folder, not through GitHub** (GitHub
limits files to 100 MB). Only the codebook and a short note (`data/README.md`) are in git. If you cloned from GitHub, copy `data/` from the shared OneDrive
folder into the repository root; every script expects it there.

Which file to use:

| Task | File |
|---|---|
| Any analysis of the corpus | `data/soccer_data.clean.csv` (or a yearly file for one year) |
| Checking what was filtered and why | the six side files; column `filter_reason` names the rule |
| Understanding a variable, a rule or a limitation | `data/codebook_soccer_posts.md` (same content as the PDF) |
| Reproducing the cleaning from scratch | `data/soccer_posts.csv` + `code/02-data-cleaning.R` |

Loading the CSVs: every column is text; missing = empty cell; booleans are `True`/`False`; timestamps are
UTC. In R use `readr::read_csv(path, col_types = cols(.default = col_character()), na = character())`;
in Python `pandas.read_csv(path, dtype=str, keep_default_na=False)`. **Do not use `data.table::fread()`
with default settings** — it doubled every quotation mark in version 1 of the cleaned files.

## 3. How the corpus was built (short version)

1. **Collection.** All r/soccer submissions 2010-01-01 → 2026-07-31 retrieved from the Arctic Shift archive
   (9 Aug 2026) in three batches, de-duplicated by post id. Posts only — no comment text exists in the corpus.
2. **Cleaning (v2.1, 13 Oct 2026).** Seven ordered filter stages, each writing its rows to a side file:
   empty content (0) · deleted by author (283,433) · removed by moderators/Reddit (375,613) · bot-authored
   template threads (48,156) · promotional/spam (34,974) · unrepairable encoding damage (335; 126 titles were
   repaired by strict byte-level re-decoding) · non-English titles (2,790). Retained: **1,302,021 (63.6 %)**.
   Raw = clean + side files, verified by 28 automated checks. Rules, thresholds and literature anchors:
   `docs/session-01_2026-10-11/02_cleaning-rules-and-rationale.md`.
3. **Documentation.** Codebook v2.1 in `data/`; every session's findings and decisions in `docs/` and `memory/`.

Things to keep in mind when analysing: 88 % of posts are link posts whose only text is the title;
engagement figures are crawl-time snapshots; Reddit's removal metadata exists only from mid-2019, so the
retained share of posts drops from 2020 onwards (a metadata effect, not a collection gap).

## 4. The analytical pipeline (current work)

`code/06-analysis-pipeline.py` (designed in session 04, implemented in session 05) turns the clean corpus into the
chapter's analysis variables. One post = one document (title + up to 1,000 characters of body). The corpus is cut
into **nine period bins** (five World Cup tournament periods, four interim periods; `period_bin`); for each bin a
BERTopic model (mpnet sentence embeddings → UMAP → HDBSCAN with a 50–100-topic guard → c-TF-IDF keywords) assigns
every post a `topic_id`; two Qwen3-1.7B "analyst" agents that must agree (multi-turn debate, at most five rounds)
name and describe every topic (S3a) and then group each bin's topics into a discourse taxonomy (`category_id`, S3b);
VADER scores every title (`post_sentiment`). Co-authors review the labels and taxonomies through editable override
files before stage S6 (`code/07-tag-corpus.R`) appends the eight columns (codebook §4.9) to the master and yearly
files. Design with every setting and its reason: `docs/session-04_2026-10-22/01_analytical-pipeline-design.md`;
approvals and implementation notes: `docs/session-05_2026-10-23/`; how to run it: `memory/workflows.md` §7 and the
session-06 runbook (`docs/session-05_2026-10-23/03_runbook-session-06.md`).

`code/05-topic-model.py` (year / ±2-month window / era runs) remains as the exploratory tool from session 02; its 2010
test outputs were removed on 2026-10-23 (git history up to `6eeb0fee`; see
`docs/session-02_2026-10-13/03_topic-model-test-case.md`).
Design and plain-language explanation of every setting: `docs/session-02_2026-10-13/02_topic-modelling-design.md`.

## 5. Running things

All commands from the repository root. R 4.6 with tidyverse/readr/jsonlite (+ `cld2`, `cld3` for the language
step); Python 3.14 for topic modelling.

```bash
# cleaning and validation (≈10 + 5 min)
Rscript code/02-data-cleaning.R
Rscript code/03-data-validation.R

# Python environment (one-time)
python3 -m venv .venv && .venv/bin/pip install -r code/requirements-topicmodel.txt
export HF_HOME=$PWD/.hf-cache TOKENIZERS_PARALLELISM=false
.venv/bin/python code/tests/test_pipeline_units.py                       # 13 checks, no GPU needed

# the analytical pipeline (full runs on Colab: code/06b-analysis-pipeline-colab.ipynb)
.venv/bin/python code/06-analysis-pipeline.py --stage bins               # period_bin for every post (≈ 3 min)
.venv/bin/python code/06-analysis-pipeline.py --stage embed --bin 1
.venv/bin/python code/06-analysis-pipeline.py --stage topics --bin 1     # on Colab: python -m cuml.accel …
.venv/bin/python code/06-analysis-pipeline.py --stage label --bin 1      # → review topic_overrides.csv
.venv/bin/python code/06-analysis-pipeline.py --stage taxonomy --bin 1   # → review category_overrides.csv
.venv/bin/python code/06-analysis-pipeline.py --stage sentiment
.venv/bin/python code/06-analysis-pipeline.py --stage collect
Rscript code/07-tag-corpus.R && Rscript code/08-validate-analysis.R      # stage S6 (gated on the reviews)
```
Full details, the local dry-run recipe and the Colab route: `memory/workflows.md` §7.

## 6. Conventions

* Nothing is deleted silently: filtered rows go to a side file with a `filter_reason`; repairs are logged.
* Every script is heavily commented and starts with a header (purpose, why, how to run, outputs).
* Settings live in one place per pipeline (`code/02-data-cleaning.R` §1, `code/topicmodel/config.py`) and
  each run records the settings it used.
* Each working session gets a folder in `docs/` and a dated file in `memory/`; decisions are recorded with
  who took them and why. Literature is cited only when real and verifiable.
* Decisions that affect validity (filter rules, thresholds, model choices) are taken by the authors, not by
  the tooling; open decisions are listed at the end of the latest `memory/` session file.

## 7. Citation and provenance

Data source: Arctic Shift Reddit archive (https://arctic-shift.photon-reddit.com/). Codebook version 2.1,
McGill University, 2026. Method template for the topic-modelling stage: BERTopic (Grootendorst 2022) with
LLM-assisted topic labelling; see the codebook's reference list and `docs/` for the full anchors.

## License
Private - © Alan Nemirovski, Teresa Pian, Borna Atrichan, Aiden McIlvaney (2026). Please do not use or reproduce any material from this repository, as it remains a work in progress. Replication materials will be made available as soon as possible.
