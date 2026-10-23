# Runbook for session 06 — running the analytical pipeline (bin 1 first, then the gates, then bins 2–9)

**For:** whoever runs the pipeline next (Alan Nemirovski; a co-author could follow it). **Assumes:** the repository at
the state of the session-05 commits, the data folder in place, Colab Pro+ (A100). Every step names the file to look at
and the decision (if any) to record. Design references are to `../session-04_2026-10-22/01_analytical-pipeline-design.md`.

## 0. Before Colab (local, 10 min)
1. `git pull`; `.venv/bin/python code/tests/test_pipeline_units.py` → 13/13 PASS.
2. `data/analysis/period_bins.csv` exists locally (written 2026-10-23; Σ = 1,302,021). If the data changed, re-run
   `--stage bins --force`.
3. Once: upload `data/yearly datasets/` (17 files, 2.4 GB) to Google Drive → `MyDrive/fifa-reddit-communities/data/yearly datasets/`.

## 1. Colab environment — **gate 0 (R8)**
Open `code/06b-analysis-pipeline-colab.ipynb` (VS Code → Colab kernel → GPU/A100, or colab.research.google.com), run
cells 1–3.
* Cell 1 tests `drive.mount()` from this kernel. If it fails, set `SHARED_DATA_URL` (a shared Drive folder link) and
  re-run: the data is fetched with `gdown`. Record which path worked (the design left this open, §10).
* Cell 3 must print `pip check: OK`, the GPU name, `cuml <version>` and the pins (bertopic 0.17.4, umap 0.5.12,
  hdbscan 0.8.44, sentence-transformers 6.1.0). If `pip check` lists a broken requirement, install **that one package**
  explicitly and re-run cell 3; do not run a plain `pip install -r requirements-topicmodel.txt` (it would move numpy /
  scikit-learn / pandas against RAPIDS). Note the resolved versions in the session-06 run log.

## 2. S0 on Colab (cell 5)
Counts must equal the design table (bin 1 = 10,002 … bin 9 = 37,626; Σ = 1,302,021).

## 3. Bin 1 — **gate 1 (design §12.1)**
Cell 6 with `BINS = [1]` (embed ≈ 1 min, topics ≈ 1–2 min on an A100). Then check `results/analysis/bin-1/config.json`:
* `topics.backend.backend` = `"cuml"` (otherwise the accelerator did not engage: check `cuml_accel`, `umap_class`,
  and that the command was prefixed with `python -m cuml.accel`); `cuml.accel --verbose` output in the cell shows any
  silent CPU fall-back of UMAP (cosine metric) or HDBSCAN.
* `topics.guard.trials` / `converged`, `n_topics_raw` within 50–100, `outlier_share_raw` (2010 test: 37–40 %).
* Open `report.html`: the 2010 themes must be visible (vuvuzelas, Jabulani, Lampard's goal, Suárez's handball, Spain's
  win, Paul the octopus) — compare with the session-02/03 observations recorded in
  `../session-02_2026-10-13/03_topic-model-test-case.md`.

## 4. Bin 1 — S3a labelling (cell 8) — **gate 4 (§12.4)**
≈ 45 min estimated for ~700 topics over all bins; bin 1 alone takes minutes on an A100. Check the printed summary:
`needs_review` ≤ 15 % of topics; the `agreement_mode` and `round_iterations` distributions (`config.json → label`).
Read 20 topics against their posts (`labels/topic_<id>.json` holds the 40 sample posts and both analysts' answers).
Optional comparisons, each into its own folder so bin-1's real outputs stay clean:
* seeded sampling (F17): `--stage label --bin 1 --sampling seeded --out-root /content/cmp-seeded` after copying
  `results/analysis/bin-1` there (the stage needs `documents.csv`, `topics.csv`); compare names and `needs_review`.
* Qwen3-4B (F9): same with `--llm Qwen/Qwen3-4B --out-root /content/cmp-4b` (≈ 20 min on an A100).
**Decision to record (design §14):** greedy vs seeded sampling for the full run.

## 5. Outlier threshold `T` — **gate 3 (§12.3)**
`results/analysis/bin-1/outlier_calibration/calibration_table.csv` lists the candidate thresholds (50th / 65th / 80th
percentile of outlier-to-nearest-topic similarity) and `sample_T<value>.csv` gives 100 posts per candidate (the band
between that candidate and the next one) with the nearest topic's keywords; a co-author fills `belongs` (yes/no).
Repeat on the large bin of step 7 before fixing `T`. Then: set `OUTLIER_THRESHOLD = {"default": T}` (or per bin) in
`code/topicmodel/config.py`, commit, and run `--stage reassign --bin N --outlier-threshold T` (cell 11; seconds per
bin; rewrites sizes, prevalence and the report, never `topic_id_raw`, keywords or labels).
**Decision to record:** `T` (one value or per bin), the co-author agreement on the samples.

## 6. Bin 1 — S3b taxonomy (cell 10)
After the S3a review (`topic_overrides.csv` filled). Check `categories.csv`: `agreement_mode`, `round_iterations`,
`n_repaired` (topics the deterministic repair had to place) and read `taxonomy.json → repair_log`. Co-authors review
→ `category_overrides.csv`. Any later change to `topic_overrides.csv` requires `--stage taxonomy --bin 1 --force`.

## 7. Large-bin gate — **gate 2 (§12.2)**: bin 4 (331k posts) or bin 6 (306k)
Cell 6 with `BINS = [4]`: raw outlier share ≤ 45 %, guard converged within four trials, UMAP + HDBSCAN minutes on the
GPU (≈ 1 h on CPU would mean the accelerator is off). Then S3a (`--stage label --bin 4`) and S3b; the S3b partition
should be valid without deterministic repair for most topics (`n_repaired` small).
**Decision to record if the gate fails:** `min_samples` (`HDBSCAN_MIN_SAMPLES`) and/or `TOPIC_COUNT_RANGE`, with the
evidence (trials, outlier share), before any other bin is run.

## 8. Bins 2–9
`BINS = [2, 3, 5, 6, 7, 8, 9]` (plus whichever of 4/6 was not the gate): embed + topics → label → review → taxonomy →
review → reassign with the fixed `T`. Each bin's `config.json` is the record; keep the review files (`*_overrides.csv`)
in git with the results.

## 9. Sentiment and collect (cell 12) — **gate 5 (§12.5)**
`--stage sentiment` (≈ 15 min, CPU) writes `data/analysis/sentiment.csv` and `sentiment_validation_sample.csv`
(300 posts, blank `coder_1`/`coder_2`). Two co-authors code the sample (positive / neutral / negative); report percent
agreement and Cohen's κ with VADER and between coders in the session-06 run log (and codebook §8.2).
`--stage collect` (without `--allow-partial`) must end with `post_variables.json → complete: true`.

## 10. Back to the repository
`results/analysis/` (tracked in git) and `data/analysis/` (shared data folder, not git) are on Drive; download them
into the repository. Commit `results/analysis/` with the review files.

## 11. Stage S6 and the codebook (local, ≈ 30 min)
1. `code/03-data-validation.R` checks C4a/C4c expect 42 / 43 columns: extend them to accept the v3.0 spec (50 / 43) —
   do this in the S6 commit, not before.
2. `Rscript code/07-tag-corpus.R` (in place; ≈ 10–15 min; rewrites the master and the 17 yearly files; stores the
   fingerprints in `data/data artefacts/tagging_fingerprint.json`), then `Rscript code/08-validate-analysis.R`
   (ALL CHECKS PASSED) and `Rscript code/03-data-validation.R`.
3. Codebook: remove the *draft* marker, update the file table (50 columns; `post_variables.csv` row), the version-3.0
   row (run date, taxonomy version from `post_variables.json`), §8.2 with the validation-sample agreement; render the
   PDF (`Rscript code/04-render-codebook.R`).
4. `docs/session-06_<date>/` (run log with every gate's numbers and decisions), `memory/`, design §14 rows for the
   decisions taken (T, sampling mode, min_samples), commit and push. Let the shared folder finish syncing (≈ 4.3 GB of
   rewritten CSVs).

## Estimates (design §10, A100)
Embeddings 15–20 min for the corpus · UMAP + HDBSCAN 1–5 min per bin · S3a ≈ 45 min for all bins (re-estimate from
bin 1) · S3b minutes per bin · VADER 15 min · ≈ 3 h compute in total (≈ 35–50 compute units), spread over the review
pauses.
