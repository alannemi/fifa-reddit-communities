# results/

Outputs of the analytical pipeline (`code/05-analysis-pipeline.py`; design `docs/session-04_2026-10-22/01_analytical-pipeline-design.md`).

| Folder | Content | In git? |
|---|---|---|
| `analysis/` | **The real run**: `bin-1` … `bin-9/` (per bin: `documents.csv`, `topics.csv`, `topics_over_time.csv`, `labels/` transcripts, `topics_labelled.csv`, `taxonomy.json`, `categories.csv`, `topic_categories.csv`, the review templates `*_overrides.csv`, `outlier_calibration/`, `config.json`, `log.txt`, `report.html`) and the aggregated tables `topic_labels_all_bins.csv`, `categories_all_bins.csv`, `category_crosswalk.csv`. Written on Google Drive (`MyDrive/fifa-communities/results/`) and copied here through the hand-back zip; empty until the first Colab run | yes (once it exists) |
| `embeddings/` | the sentence-embedding cache, one `.npz` per year and model (regenerable; ≈ 4 GB for the corpus with mpnet) | no |
| `dry-runs/<name>/` | local dry runs of the pipeline on bin 1 with MiniLM embeddings (never part of the analysis). Each has the same shape as a real run: `results/analysis/bin-1/` and `data/analysis/`. Tracked: the pipeline outputs under `results/analysis/`, `data/analysis/post_variables.json`, `period_bins_summary.csv`, `sentiment_validation_sample.csv`, the logs. Not tracked: the join tables `period_bins.csv`, `sentiment.csv`, `post_variables.csv` (23–33 MB each, regenerable with `--stage bins/sentiment/collect --out-root`) and the R mini-corpus folders `r-mini/`, `r-out*/` (copies of the 2010 yearly file: data stays off GitHub, `data/README.md`) | partly |

## The dry runs

| Run | Session | What it shows |
|---|---|---|
| `bin1-3000` | 05 (2026-10-23) | the design's 3,000-post subsample: collapses to 2 topics under HDBSCAN's cluster selection at the floor of 25 (finding D1) |
| `bin1-full` | 05 (2026-10-23) | the full bin 1: 92 topics, 35.9 % raw outliers; the 12 largest topics labelled (1 needs review); the 6-category taxonomy of the single-prompt protocol; sentiment for 2010; a partial collect; the S6 scripts tested on a mini corpus (`r-mini/`, `r-out/`, `r-out2/`, not tracked) |
| `bin1-full92` | 05/06 (2026-10-23/24) | all 92 topics labelled (`needs_review` 2.2 %); the single-prompt taxonomy over 92 items that failed (finding D2 → decision D5) |
| `bin1-twostep` | 06 (2026-10-24) | a copy of the 92-topic run on which the two-step taxonomy was started with the real model and stopped after round 1 of the definitions debate (`run-twostep.log`) |
| `h7-bin1` | 07 (2026-10-25) | embeddings and topics only, after the code harmonization: `umap_5d.npy` and `hdbscan_labels_raw.npy` bit-identical to `bin1-full` |

Records: `docs/session-05_2026-10-23/02_dry-run-log.md`, `docs/session-06_2026-10-24/`, `docs/session-07_2026-10-25/01_code-harmonization-notes.md` §8.
Read every CSV as text (`readr::read_csv(…, col_types = cols(.default = col_character()), na = character())`).
