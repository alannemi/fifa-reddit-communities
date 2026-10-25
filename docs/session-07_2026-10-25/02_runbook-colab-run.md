# Runbook for the Colab run — bin 1, the gates, bins 2–9, stage S6 (pipeline of decisions D7/D8)

**For:** whoever runs the pipeline next (Alan Nemirovski; a co-author could follow it). Design references are to
`../session-04_2026-10-22/01_analytical-pipeline-design.md` (v1.5). **Rewritten on 2026-10-25 (evening)** after the author's
decisions D7 and D8 (`../session-08_2026-10-25/00_colab-run-log.md` §9): the unit of analysis is the individual BERTopic
topic — there is no taxonomy stage any more — every post carries its topic's corpus-wide key, label and tone, and the
topic count per bin follows a density target (50–75 raw topics per 10,000 modelled posts) instead of a fixed range.
The earlier versions of this runbook (sessions 06–07, with the two-step taxonomy) are in git history.

> **Where things stand (2026-10-25, evening).** Bin 1 has run end to end on Colab (Pass 1 and the full labelling of its
> 65 topics; the taxonomy outputs of that day are superseded and were removed from the bin folder). The large-bin gate on
> bin 4 failed on the outlier share under the old settings and is to be re-run under D8 (§7). The co-author review of
> bin 1 (one file, §6) can start now.

## 0. Before Colab (local, 10 min)
1. `git pull`; `.venv/bin/python code/tests/test_pipeline_units.py` → 12/12 PASS.
2. `data/analysis/period_bins.csv` exists locally (Σ = 1,302,021). If the data changed, re-run
   `.venv/bin/python code/05-analysis-pipeline.py --stage bins --force`.
3. The 17 yearly files are on Google Drive at `MyDrive/fifa-communities/yearly datasets/`. The notebook creates the other
   folders of the flat layout (`analysis/`, `results/`, `embeddings/`, `logs/`, `handback/`, `comparisons/`) on its first run.

## 1. Colab environment — gate 0 (review R8)
Open `code/05b-analysis-pipeline-colab.ipynb` (VS Code → the **Colab** kernel → GPU/A100, high RAM; or
colab.research.google.com from the GitHub repository) and run cells 0.1–0.4. A local Python kernel fails in cell 0.1
with "No module named 'google'": select the Colab runtime.
* **0.1** mounts Drive, asserts the 17 files, creates the output folders and opens the session log
  (`MyDrive/fifa-communities/logs/colab_<date>.log`); every later cell writes into it.
* **0.2** clones the public repository on the Colab machine, copies the CSVs to `/content/data/yearly datasets`, sets
  `PIPELINE_ROOT` to the Drive folder, `PIPELINE_YEARLY_DIR` to the local copy and `PIPELINE_HF_CACHE_DIR` to
  `/content/hf-cache`, and **prints the program's effective paths** — check that `ANALYSIS_DIR` ends in
  `fifa-communities/results`, `ANALYSIS_DATA_DIR` in `fifa-communities/analysis`, `YEARLY_DIR` in `/content/data/yearly
  datasets`, `LLM_CACHE_DIR` in `/content/hf-cache` (flag F34). The model weights are re-downloaded every session (≈ 4 GB,
  a few minutes); they cannot live on Drive (safetensors memory-maps them; F39). Re-run 0.2 after every push.
* **0.3** must print the GPU name, `cuml <version>`, the pins (bertopic 0.17.4, umap 0.5.12, hdbscan 0.8.44,
  sentence-transformers 6.1.0) and `pip check (pipeline stack): OK`. A `pip check` complaint about Colab's own image
  ("ipython requires jedi") is benign; a complaint naming a package of the pipeline stack stops the cell: install **that one
  package** explicitly and re-run. Record the resolved versions in the run log (first run: A100-SXM4-80GB, torch
  2.11.0+cu130, CUDA 13.0, cuML 26.06.00, plotly 5.24.1).
* **0.4** runs the 12 unit checks on the Colab machine and defines the helpers (`stage`, `cfg`, `report_ok`).

## 2. S0 on Colab (cell 1)
Writes `fifa-communities/analysis/period_bins.csv`; the counts must equal the design table (bin 1 = 10,002 … bin 9 = 37,626;
Σ = 1,302,021). The summary table also shows each bin's density target (`target_topics_min` / `target_topics_max`, D8:
bin 1 ≈ 49–74, bin 4 ≈ 1,650–2,480).

## 3. Pass 1 — bin 1 smoke test (cells 2.x) — gate 1 (design §12.1)
`fifa-communities/results/bin-1/config.json → topics`: `backend.backend = "cuml"` (otherwise the accelerator did not engage:
check `cuml_accel`, `umap_class`, and that the command was prefixed with `python -m cuml.accel`), `guard.converged` (the
density target of D8), `n_topics_raw`, `outlier_share_raw`, `hdbscan` (the settings in force). First Colab run of bin 1
(2026-10-25): 65 topics at `min_cluster_size` 25, 34.8 % raw outliers — inside 49–74, so bin 1 stands under D8. Open
`results/bin-1/report.html`: the 2010 themes must be visible (vuvuzelas, Jabulani, Lampard's goal, Suárez's handball, Spain's
win, Paul the octopus). Pass 1 also labels the 12 largest topics, scores 2010 and collects (partial). The report is
re-rendered after topics, reassign and label; Drive's web view does not render HTML, so download it or show it in the
notebook (`IPython.display.HTML(open(path).read())`); the notebook asserts its presence (`report_ok()`).

## 4. Pass 2, cell 3.1 — bin 1 in full — gate 4 (§12.4) — **stop point of the test batch**
Labels every topic (resumable: re-run the cell after a disconnect). Gate 4: `needs_review` ≤ 15 % of topics (first run: 0 of
65); the `agreement_mode` and `round_iterations` distributions (`config.json → label`). Then the **hand-back cell (section
5)**: the zip `fifa-communities/handback/bins-1_<date>.zip` holds `results/analysis/bin-1/` (without `model/`), the aggregated
tables, `data/analysis/*` and the log, laid out as repository paths — unzip it at the repository root, commit
`results/analysis/bin-1/`, move `data/analysis/` to the shared data folder. Then `python code/08-review-pack.py pack --bin 1`
renders `results/analysis/bin-1/review_pack.md` for the reviewers. The reviewer can also read the outputs on Drive through
the Google Drive connector (authorised 2026-10-25).
**Decisions recorded (design §14):** greedy decoding kept, the seeded comparison dropped (2026-10-25); the D2 name + tone
refinement is moot (0 % flagged). **Done for bin 1 on 2026-10-25** (`../session-08_2026-10-25/00_colab-run-log.md` §3).

## 5. Outlier threshold `T` — gate 3 (§12.3)
`results/bin-1/outlier_calibration/calibration_table.csv` lists the candidate thresholds (50th / 65th / 80th percentile of
outlier-to-nearest-topic similarity); `sample_T<value>.csv` gives 100 posts per candidate with the nearest topic's keywords;
a co-author fills `belongs` (yes/no). Repeat on the large bin of step 7 before fixing `T`. Then set
`OUTLIER_THRESHOLD = {"default": T}` (or per bin) in `code/topicmodel/config.py`, commit, and run
`--stage reassign --bin N --outlier-threshold T` (cell 3.4; seconds per bin; rewrites sizes, prevalence and the report,
never `topic_id_raw`, keywords or labels).
**Decision to record:** `T` (one value or per bin), the co-author agreement on the samples.

## 6. Co-author review per bin (one file since D7)
`results/analysis/bin-N/topic_overrides.csv`: `final_name` renames a topic, `merge_into` folds it into another topic, `notes`
says why. The pack `review_pack.md` (`python code/08-review-pack.py pack --bin N`) shows every topic with its agreed label,
both analysts' answers, the keywords, the representative titles and the 40 sample posts; `check --bin N` validates the
filled file (ids exist, no merge cycles). Edit the repository copy, commit, upload the file to Drive
(`fifa-communities/results/bin-N/`), then `--stage collect --allow-partial` applies it: merged topics disappear from the topic
table (their posts carry the target's `topic_id`, `topic_uid`, `topic_name`, `topic_tone`), renamed topics show the final name
(`name_agent` keeps the analysts'). Nothing else re-runs. Step-by-step guide for the reviewers:
`../session-08_2026-10-25/01_review-guide-bin-1.md`. **Under D8 the interim bins will hold 1,000–2,500 topics each** (flag
F44): a full read is not feasible there — the design's sample (20 topics against their posts) plus every `needs_review` topic
is the review; the author decides whether a larger sample is needed.

## 7. Large-bin gate — gate 2 (§12.2): bin 4 (331k posts) or bin 6 (306k)
Cell 3.3 with `GATE_BIN = 4`: raw outlier share ≤ 45 %, guard converged (to the density target, within six trials), UMAP +
HDBSCAN minutes on the GPU (≈ 1 h on CPU would mean the accelerator is off); then the labelling (≈ 1 h for ≈ 2,000 topics on the
A100, resumable).
**First attempt (2026-10-25, under the old 50–100 range):** 331,216 posts, guard 820 → 2 topics, 547 → 74 topics, **56.2 %
raw outliers** — failed on the outlier share. Under D8 the guard searches from `min_cluster_size` 25 towards ≈ 1,650–2,480
topics, so the outlier share will be different; the first attempt's numbers are not comparable. If the gate fails again:
cell 3.3b (`--stage grid --bin 4`: HDBSCAN over `HDBSCAN_GRID` — `min_cluster_size` 25/40/60/100 × `min_samples` 10/5/3 ×
eom/leaf — on the cached UMAP array, minutes) → the author chooses `HDBSCAN_MIN_SAMPLES` / `HDBSCAN_SELECTION_METHOD` /
`HDBSCAN_EPSILON`, one value or a dict by period type, records it in design §14, commits → cell 3.3c (`topics --force
--keep-umap`, the gate checks, label). A setting that applies to every bin means re-clustering and relabelling bin 1 before
its review is used.

## 8. Bins 2–9 — one bin at a time, review as you go
Cell 3.5 finishes each bin before starting the next: embed → topics (a bin clustered before D8 is refitted on its cached UMAP
array) → the gate (raw outliers ≤ 45 %, guard converged; **the loop stops at the first bin that fails** — cell 3.3b's grid with
`GATE_BIN = N`, the author's setting, then re-run 3.5: finished bins are skipped) → label (resumable) → `review_pack.md` →
a per-bin hand-back zip in `handback/` → the flag **`results/bin-N/REVIEW_READY.txt`** on Drive (written by the pipeline when
every topic of the bin is labelled; `status.json` beside it holds the numbers) and the log line "bin N READY FOR REVIEW".
`--stage status` (cell 3.6, or locally) prints one row per bin. The co-authors review a finished bin (§6) while the GPU works
on the next; after the reviews, the reassignment with the fixed `T` (§5) runs for every bin. Each bin's `config.json` is the
record; keep the review files in git with the results.

## 9. Sentiment and collect (section 4) — gate 5 (§12.5)
`--stage sentiment` (≈ 15 min, CPU) writes `fifa-communities/analysis/sentiment.csv` and `sentiment_validation_sample.csv`
(300 posts, blank `coder_1`/`coder_2`). Two co-authors code the sample (positive / neutral / negative); report percent
agreement and Cohen's κ with VADER and between coders in the run log (and codebook §8.2).
`--stage collect` (without `--allow-partial`) must end with `post_variables.json → complete: true`. It writes every post's
`topic_id`, `topic_uid` (`<bin>-<topic_id>`), `topic_name`, `topic_tone`, `topic_assignment` and the three sentiment columns;
`results/analysis/topic_labels_all_bins.csv` (one row per topic with `mean_vader_sentiment` and `n_posts_scored`); and the
template `results/analysis/topic_crosswalk.csv` (one row per topic, `chapter_theme` blank) for the co-authors' manual
cross-bin comparison (kept once it has entries).

## 10. Back to the repository
The hand-back cell (section 5, `HANDBACK_BINS = [1, …, 9]`) zips `results/bin-N/` (without `model/`), the aggregated tables,
`analysis/*` and the session log as repository paths into `fifa-communities/handback/`; unzip at the repository root.
`results/analysis/` is tracked in git (commit it with the review files); `data/analysis/` goes to the shared data folder.

## 11. Stage S6 and the codebook (local, ≈ 30 min)
1. `Rscript code/06-tag-corpus.R` (in place; ≈ 10–15 min; rewrites the master and the 17 yearly files with the **10
   analysis columns** — 52 columns in total; stores the fingerprints in `data/data artefacts/tagging_fingerprint.json`), then
   `Rscript code/07-validate-analysis.R` (ALL CHECKS PASSED; dry-run on the mini 2010 corpus 2026-10-25) and
   `Rscript code/03-data-validation.R` (28/28; its C4 check accepts the v3.0 spec of 42 + 10 columns).
2. Codebook: remove the *draft* marker, update the file table and the version-3.0 row (run date, labels version from
   `post_variables.json`), §8.2 with the validation-sample agreement; render the PDF (`Rscript code/04-render-codebook.R`).
3. The run log (`docs/session-08_2026-10-25/00_colab-run-log.md` or its successor) with every gate's numbers and decisions,
   `memory/`, design §14 rows for the decisions taken (`T`, the clustering settings of the gate), commit and push.

## Estimates (A100)
Embeddings 15–20 min for the corpus · UMAP + HDBSCAN 1–7 min per bin (up to six guard fits) · labelling ≈ 2 min per 65 topics
(first run), i.e. ≈ 1 h for a 2,000-topic interim bin · VADER 15 min · roughly 6 h of compute for all nine bins under D8,
spread over the review pauses.
