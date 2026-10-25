# Runbook for the Colab run — the bin-1 test batch, the gates, bins 2–9, stage S6

**For:** whoever runs the pipeline next (Alan Nemirovski; a co-author could follow it). **Carried forward from**
`../session-06_2026-10-24/03_runbook-session-07.md` with the file names, the Drive layout and the settings of session 07
(`00_approvals.md`). Design references are to `../session-04_2026-10-22/01_analytical-pipeline-design.md` (v1.2); the
taxonomy protocol is D5 (`../session-06_2026-10-24/00_approvals.md`).

> **Where things stand (2026-10-25).** Session 06 ended before its Colab test batch ran: the local dry run of the two-step
> taxonomy was stopped on 2026-10-24 (7 minutes for round 1 of the definitions debate on the Mac) and the run moved to
> Colab. There is therefore **no Colab run log and no session-06 memory file**; the test batch (Pass 1 + the full bin 1)
> is the first thing this runbook does. The two-step protocol has never run with the real model.

## 0. Before Colab (local, 10 min)
1. `git pull`; `.venv/bin/python code/tests/test_pipeline_units.py` → 20/20 PASS.
2. `data/analysis/period_bins.csv` exists locally (Σ = 1,302,021). If the data changed, re-run
   `.venv/bin/python code/05-analysis-pipeline.py --stage bins --force`.
3. The 17 yearly files are on Google Drive at `MyDrive/fifa-communities/yearly datasets/` (confirmed 2026-10-25; the only
   folder there today). The notebook creates the other folders of the flat layout (`analysis/`, `results/`, `embeddings/`,
   `hf-cache/`, `logs/`, `handback/`, `comparisons/`) on its first run.

## 1. Colab environment — gate 0 (review R8)
Open `code/05b-analysis-pipeline-colab.ipynb` (VS Code → Colab kernel → GPU/A100, high RAM; or colab.research.google.com from
the GitHub repository) and run cells 0.1–0.4.
* **0.1** mounts Drive, asserts the 17 files, creates the output folders and opens the session log
  (`MyDrive/fifa-communities/logs/colab_<date>.log`); every later cell writes into it. If `drive.mount` fails from the
  VS Code kernel (design §10 left this open), open the notebook at colab.research.google.com instead.
* **0.2** clones the public repository on the Colab machine, copies the CSVs to `/content/data/yearly datasets`, sets
  `PIPELINE_ROOT` to the Drive folder, `PIPELINE_YEARLY_DIR` to the local copy and `PIPELINE_HF_CACHE_DIR` to
  `/content/hf-cache`, and **prints the program's effective paths** — check that `ANALYSIS_DIR` ends in
  `fifa-communities/results`, `ANALYSIS_DATA_DIR` in `fifa-communities/analysis`, `YEARLY_DIR` in `/content/data/yearly
  datasets`, `LLM_CACHE_DIR` in `/content/hf-cache` (flag F34). The model weights are re-downloaded every session (≈ 4 GB,
  a few minutes): on the first Colab run (2026-10-25) the topics stage failed with `SafetensorError: header too large`
  when it read the mpnet weights from `hf-cache/` on Drive — safetensors memory-maps the file and Drive's FUSE mount
  cannot serve that — so the weights live on the local disk and the Drive folder `hf-cache/` is unused (delete it).
* **0.3** must print the GPU name, `cuml <version>`, the pins (bertopic 0.17.4, umap 0.5.12, hdbscan 0.8.44,
  sentence-transformers 6.1.0) and `pip check (pipeline stack): OK`. A `pip check` complaint about Colab's own image — on
  2026-10-25 "ipython 7.34.0 requires jedi, which is not installed" — is benign (fix with `pip install jedi` or ignore);
  a complaint naming a package of the pipeline stack stops the cell: install **that one package** explicitly and re-run
  (never a plain `pip install -r requirements-topicmodel.txt`). Record the resolved versions in the run log (first run:
  A100-SXM4-80GB, torch 2.11.0+cu130, CUDA 13.0, cuML 26.06.00).
* **0.4** runs the 20 unit checks on the Colab machine.

## 2. S0 on Colab (cell 1)
Writes `fifa-communities/analysis/period_bins.csv`; the counts must equal the design table (bin 1 = 10,002 … bin 9 = 37,626;
Σ = 1,302,021).

## 3. Pass 1 — bin 1 smoke test (cells 2.x) — gate 1 (design §12.1)
`fifa-communities/results/bin-1/config.json → topics`: `backend.backend = "cuml"` (otherwise the accelerator did not engage:
check `cuml_accel`, `umap_class`, and that the command was prefixed with `python -m cuml.accel`), `guard.converged`,
`n_topics_raw` within 50–100, `outlier_share_raw` (session-05 dry run: 35.9 % with MiniLM). Open `results/bin-1/report.html`:
the 2010 themes must be visible (vuvuzelas, Jabulani, Lampard's goal, Suárez's handball, Spain's win, Paul the octopus).
Pass 1 also labels the 12 largest topics, runs the two-step taxonomy over them, scores 2010 and collects (partial).
The report (`report.html` + `report.md`) is re-rendered after topics, reassign, label and taxonomy; Drive's web view does not
render HTML, so download the file or show it in the notebook (`IPython.display.HTML(open(path).read())`). The writer never
fails a stage: if the file is missing, `results/bin-1/log.txt` holds `report for bin 1 skipped: <reason>` (first run,
2026-10-25: a plotly 5.24.1 defect in `add_vline`, fixed the same day); after a fix, `stage('--stage report --bin 1')`
re-renders in seconds. The notebook now asserts the report after every rendering stage (`report_ok()`).

## 4. Pass 2, cell 3.1 — bin 1 in full — gate 4 (§12.4) — **stop point of the test batch**
`needs_review` ≤ 15 % of topics (local dry run: 2.2 %); the `agreement_mode` and `round_iterations` distributions
(`config.json → label`). Read 20 topics against their posts (`labels/topic_<id>.json` holds the 40 sample posts and both
analysts' answers). The taxonomy summary printed by the notebook (and `config.json → taxonomy` / `taxonomy.json`) shows:
* the definitions debate: `agreement_mode`, `round_iterations`, `needs_review`; `n_categories_defined` → `n_categories`,
  `categories_dropped_empty` (defined but never assigned a topic, F23), `definitions_cleanup` (same-name merges);
* the assignment debates: `assignment_agreement_modes` over the items, `n_items_contested`; per topic in
  `topic_categories.csv`: `assignment_agreement_mode`, `assignment_round_iterations`, `assignment_contested`,
  `assignment_needs_review`;
* `n_repaired` (topics placed by embedding similarity), `n_unplaced` (must be 0 unless the definitions failed),
  `n_answers_salvaged`, `llm_stats.truncated`.
Then run the **hand-back cell (section 5)**: the zip `fifa-communities/handback/bins-1_<date>.zip` holds `results/analysis/bin-1/`
(without `model/`), the aggregated tables, `data/analysis/*` and the log, laid out as repository paths — unzip it at the
repository root, commit `results/analysis/bin-1/` with the review templates, move `data/analysis/` to the shared data folder.
The reviewer reads the outputs through the Google Drive connector (to be authorised in claude.ai) or from the zip.
**Decisions to record (design §14):** ~~greedy vs seeded sampling~~ — **decided 2026-10-25 (Alan): greedy kept, the seeded
comparison of cell 3.2 dropped**; whether the D2 "identical" rule needs the name + tone refinement (only if `needs_review`
exceeds 15 %; the rule compares all five label fields, the refinement would count a match on name + tone category) — **moot on
2026-10-25: 0 of 65 topics flagged**.
**D6 (Alan, 2026-10-25), after the first full bin 1** (`../session-08_2026-10-25/00_colab-run-log.md`): the assignment step
lists only each topic's 10 most similar categories, asks for id + name, and flags a topic whose agreed category ranks worse
than 3rd by similarity. Re-run step 2 alone — the definitions are kept — with notebook cell 3.1b (`--stage taxonomy --bin 1
--force --keep-definitions`), then `collect --allow-partial` and the hand-back again. Read `taxonomy.json → shortlist`
(`rank_distribution`, `n_flagged_by_rank`), `topic_categories.csv → assignment_similarity_rank`, and the flagged topics in
`topic_category_overrides.csv` (columns `assignment_needs_review`, `similarity_rank`); `resolved_from = name_over_id` in the
transcripts counts the id slips the name caught.

## 5. Outlier threshold `T` — gate 3 (§12.3)
`results/bin-1/outlier_calibration/calibration_table.csv` lists the candidate thresholds (50th / 65th / 80th percentile of
outlier-to-nearest-topic similarity); `sample_T<value>.csv` gives 100 posts per candidate with the nearest topic's keywords;
a co-author fills `belongs` (yes/no). Repeat on the large bin of step 7 before fixing `T`. Then set
`OUTLIER_THRESHOLD = {"default": T}` (or per bin) in `code/topicmodel/config.py`, commit, and run
`--stage reassign --bin N --outlier-threshold T` (cell 3.4; seconds per bin; rewrites sizes, prevalence and the report,
never `topic_id_raw`, keywords or labels).
**Decision to record:** `T` (one value or per bin), the co-author agreement on the samples.

## 6. Co-author review per bin
Three files per bin under `results/bin-N/`: `topic_overrides.csv` (rename / merge topics; **any change here requires
`--stage taxonomy --bin N --force`**, which removes the step-1 checkpoint and re-runs both steps), `category_overrides.csv`
(rename / merge categories) and `topic_category_overrides.csv` (move one topic to another category, F28). Templates are
regenerated by every taxonomy run unless they contain entries; after a re-run, check that the ids in filled templates still
exist. Stage S5 applies them in this order: topic merges, topic-level moves, category merges.

## 7. Large-bin gate — gate 2 (§12.2): bin 4 (331k posts) or bin 6 (306k)
Cell 3.3 with `GATE_BIN = 4`: raw outlier share ≤ 45 %, guard converged within four trials, UMAP + HDBSCAN minutes on the
GPU (≈ 1 h on CPU would mean the accelerator is off). Then S3a and the two-step S3b; most topics should be assigned by the
agents (`n_repaired` small, `n_unplaced` 0).
**Decision to record if the gate fails:** `HDBSCAN_MIN_SAMPLES` and/or `TOPIC_COUNT_RANGE`, with the evidence (trials, outlier
share), before any other bin is run. If a bin collapses at the floor of 25 (dry-run finding D1): a lower `MIN_CLUSTER_FLOOR`
for tournament bins or `cluster_selection_method = "leaf"` — the author decides.

## 8. Bins 2–9
Cell 3.5: embed + topics → label → review → taxonomy → review → reassign with the fixed `T`. Each bin's `config.json` is the
record; keep the review files (`*_overrides.csv`) in git with the results.

## 9. Sentiment and collect (section 4) — gate 5 (§12.5)
`--stage sentiment` (≈ 15 min, CPU) writes `fifa-communities/analysis/sentiment.csv` and `sentiment_validation_sample.csv`
(300 posts, blank `coder_1`/`coder_2`). Two co-authors code the sample (positive / neutral / negative); report percent
agreement and Cohen's κ with VADER and between coders in the run log (and codebook §8.2).
`--stage collect` (without `--allow-partial`) must end with `post_variables.json → complete: true`.

## 10. Back to the repository
The hand-back cell (section 5, `HANDBACK_BINS = [1, …, 9]`) zips `results/bin-N/` (without `model/`), the aggregated tables,
`analysis/*` and the session log as repository paths into `fifa-communities/handback/`; unzip at the repository root.
`results/analysis/` is tracked in git (commit it with the review files); `data/analysis/` goes to the shared data folder.

## 11. Stage S6 and the codebook (local, ≈ 30 min)
1. `Rscript code/06-tag-corpus.R` (in place; ≈ 10–15 min; rewrites the master and the 17 yearly files; stores the
   fingerprints in `data/data artefacts/tagging_fingerprint.json`), then `Rscript code/07-validate-analysis.R`
   (ALL CHECKS PASSED) and `Rscript code/03-data-validation.R` (28/28; since session 07 its C4 check accepts the v3.0
   spec of 42 + 8 columns and reports which spec it found — no edit needed at this step).
2. Codebook: remove the *draft* marker, update the file table (50 columns; `post_variables.csv` row), the version-3.0 row
   (run date, taxonomy version from `post_variables.json`), §8.2 with the validation-sample agreement; render the PDF
   (`Rscript code/04-render-codebook.R`).
3. `docs/session-08_<date>/` (run log with every gate's numbers and decisions), `memory/`, design §14 rows for the
   decisions taken (`T`, sampling mode, `min_samples`), commit and push. Let the shared folder finish syncing.

## Estimates (A100)
Embeddings 15–20 min for the corpus · UMAP + HDBSCAN 1–5 min per bin · S3a ≈ 5–8 min per bin of 50–100 topics (batch 16;
re-estimate from the first run log) · S3b two-step: the definitions debate 1–3 min, the assignment debates ≈ 1 min per
analyst-round · VADER 15 min · ≈ 3 h compute in total, spread over the review pauses.
