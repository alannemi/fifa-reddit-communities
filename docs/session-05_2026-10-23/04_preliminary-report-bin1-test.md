<!-- Repository copy of the Claude Doc "Analytical pipeline: bin-1 test report" (https://claude.ai/artifact/TnXkBUW2B96BYXihnT4SGW), exported 2026-10-23 after the local dry run; the doc is the review copy (comments welcome there), this file is the record. The 92-topic background run's numbers are appended to the doc and to 02_dry-run-log.md when available. -->

# Analytical pipeline: bin-1 test report

23 October 2026 · Alan Nemirovski (decisions) · Claude (code, tests, dry run, report)

## Summary

The analytical pipeline designed in session 04 is built, unit-tested and ran end to end on bin 1 on your Mac on 23 October 2026. The modelling stage behaves as the design predicted, the two-agent stages work, the write-back validates, and one refinement (N11) needs your decision before the Colab run.

- **Built and pushed (4 commits):** `code/06-analysis-pipeline.py` with nine stages, four new modules in `code/topicmodel/`, the Colab notebook, the R scripts for stage S6, and 15 unit checks that need no GPU or model.
- **Bin 1 (10,002 posts, 1 Jan – 30 Sep 2010), locally with MiniLM embeddings and Qwen3-1.7B:** 92 topics at the first guard trial with 35.9 % raw outliers (the session-02 window test gave 93 and 36.6 %); the 2010 themes are in the keywords; 12 topics labelled in 6 minutes with 1 flagged for review; a 6-category taxonomy; sentiment; `post_variables.csv` with all 1,302,021 posts; the R write-back passed all 16 applicable checks.
- **Your decision (N11):** the 1.7B model wrote the taxonomy JSON with broken brackets in every attempt while the content was a valid partition. I added a tolerant recovery step behind a config switch; without it stage S3b produces nothing. Section 6 states the choice.
- **Two findings for the quality gates:** a 3,000-post subsample collapses into one cluster under HDBSCAN's cluster selection (D1), and the strict all-fields agreement rule can declare a stalemate between two analysts who agree on the name.
- **Not yet verified:** the Colab environment. The notebook is now a two-pass flow (smoke test, then the full run) so that the first Colab session checks exactly that.
- **In progress:** labelling of all 92 bin-1 topics plus the taxonomy at full scale, running in the background; its numbers will be appended to section 4 when it finishes.

## Process: from the approved design to code

Every decision was put to you before the corresponding code existed, and every deviation from the design is written down with its reason. The steps, in order:

1. **Planning (plan mode).** All memory files and design documents were re-read; the session plan was drafted against design §16 and the stage contracts of §3.
2. **Your decisions, asked before anything was built.** F4 (manual cross-bin crosswalk), F6 (bin boundaries exactly as in the brief) and F18 (title-only `post_sentiment`) were approved as designed; the session-02/03 test outputs were removed from git and the local embedding cache deleted, with the exploratory tool `05-topic-model.py` kept.
3. **Independent review of the plan.** A fresh reviewer returned 13 findings; all were folded in. The two that mattered most: BERTopic renumbers cluster labels by size, so `topic_id_raw` had to be defined as the BERTopic id before reassignment; and the taxonomy answer budget of 4,000 tokens would have truncated large bins (now 8,000).
4. **Approvals recorded** in `docs/session-05_2026-10-23/00_approvals.md` (your four decisions, the refinements N1–N10, the review findings R1–R13) in the first commit.
5. **Implementation** of design §16: the stage program, `topicmodel/bins.py` (S0), `prompts.py` (prompt v1.1 verbatim), `agents.py` (the batched model wrapper, validation, the D2 debate engine, S3a, S3b with chunking and deterministic repair, overrides), `tagging.py` (S5), extensions of `config.py`, `model.py` and `sentiment.py`, the Colab notebook with `requirements-colab.txt`, and `07-tag-corpus.R` / `08-validate-analysis.R` for S6.
6. **Unit checks without a model.** A scripted stand-in for the language model exercises every agreement mode of the debate protocol, the repair and no-answer paths, resume from a partial run, the chunking fallback with 130 fake topics, the partition repair, override chains, the frozen UMAP/cluster components inside BERTopic 0.17.4 and a partial collect: 15 checks, all passing.
7. **Dry run on bin 1** (section 4), including the real stage S0 on all 17 yearly files, the design's 3,000-post subsample, the full bin, and the R write-back on a mini corpus. Six defects were found and fixed on the way (section 5).
8. **Documentation.** Implementation notes with the module map, the dry-run log with every number, the runbook for the Colab session, design document v1.1, codebook wording, README, workflows and the session memory file.

Commits on `main`: `8061b4cc` housekeeping · `b740e308` implementation · `46ba904b` dry run and fixes · `5b24895c` two-pass Colab notebook.

## Test setup

The dry run exercised every stage on bin 1 on your Mac with the code the Colab run will use; two substitutions make it cheap locally, and both are config values, not code paths.

| Element | Dry run (local, 23 Oct 2026) | Colab run (session 06) |
| --- | --- | --- |
| Documents | bin 1: 10,002 posts, 9,809 modelled, 193 with fewer than three word tokens | same |
| Embeddings | `all-MiniLM-L6-v2` (384 dimensions) on the M4's MPS device | `all-mpnet-base-v2` (768 dimensions) on the GPU |
| UMAP and HDBSCAN | CPU (`umap-learn`, `hdbscan`) | NVIDIA cuML through `python -m cuml.accel` |
| Language model | `Qwen/Qwen3-1.7B`, revision `70d244cc`, bfloat16, greedy, batch 4 | same model and revision, batch 16 |
| Labelling scope | the 12 largest topics; all 92 in the background run | all topics of every bin |
| Outlier threshold | 0.5, to exercise the mechanics only | calibrated at gate 3 from the judgement samples |
| Sentiment | the 2010 file only | all 17 years |
| Outputs | `results/dry-runs/bin1-full/` (not in git) | `results/analysis/` and `data/analysis/` |

Stage S0 was also run for real (`data/analysis/period_bins.csv`), and the design's 3,000-post subsample was run first; its outcome is finding D1 in section 5.

## Outcomes stage by stage

All nine Python stages and the R write-back completed; the modelling numbers sit where the session-02 window test put them (93 topics, 36.6 % raw outliers on 7,678 posts).

| Stage | Outcome | Time |
| --- | --- | --- |
| S0 period bins | all 17 yearly files read; the nine bin sizes equal the design table; Σ = 1,302,021 | 3 min |
| S1 embeddings | 13,638 posts of 2010 encoded and cached | 22 s |
| S2 BERTopic | UMAP 9,809 × 384 → 5 in 0.4 min; guard: `min_cluster_size` 25 → **92 topics, converged at the first trial**; 35.9 % raw outliers; largest topic 284 posts (2.9 %); 35 equal-size ids swapped by BERTopic's renumbering, all recorded | 1.9 min |
| Reassignment (T = 0.5, trial) | 2,151 of 3,523 outliers reassigned; 14.0 % remain; sizes, prevalence and report rewritten | 20 s |
| S3a labelling (12 topics) | 42 prompts, 8.7 s each with batch 4; every answer parsed; agreement: identical 2, accepted by A 8, accepted by B 1, stalemate 1; rounds 1 / 2 / 3 for 1 / 10 / 1 topics; `needs_review` 1 of 12 | 6.3 min |
| S3b taxonomy | first attempt: no usable answer (bracket syntax, finding N11); second attempt with recovery: **6 categories, identical in round 2**, one topic placed by the deterministic repair, 5 of 6 answers recovered | 5.2 + 6.5 min |
| S4 sentiment (2010) | 13,638 posts scored; bin 1: neutral 4,620, positive 3,527, negative 1,855; validation sample of 299 posts with blank coder columns | 3 s |
| S5 collect | `post_variables.csv`: 1,302,021 rows, 11 columns; bin 1 populated (6,286 cluster, 2,151 reassigned, 1,372 outlier, 193 not modelled; 2,583 with a category); other bins empty; sidecar `complete = false`, no check failures; aggregated label and category tables and the crosswalk template written | 7 s |
| S6 write-back (R) | `07` tagged the 2010 file (13,638 rows, 50 columns) with per-column fingerprints; `08` passed all 16 applicable checks after the A7 fix, including byte identity of the 42 original columns | 4 + 4 s |

**2010 themes in the top-15 keywords.** Vuvuzela (topic 6), Lampard's goal (32, 62, 63), Suárez's handball (66, 74), Spain (4, 12, 68), Paul the octopus (31, 79). *Jabulani* appears in no keyword list; the bin spans nine months, so the ball sits inside broader World Cup topics.

**The 12 labelled topics** (sizes before and after the trial reassignment):

| Topic | Cluster | Total at T = 0.5 | Name | Tone | Rounds | Agreement | Review |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | 284 | 354 | Arsenal vs Liverpool | other | 3 | accepted by A | — |
| 1 | 217 | 327 | World Cup Predictions | enthusiastic | 2 | stalemate | yes |
| 2 | 199 | 228 | EPL Support | positive | 2 | accepted by A | — |
| 3 | 169 | 237 | World Cup Match Predictions | enthusiastic | 2 | accepted by A | — |
| 4 | 167 | 231 | FIFA Livestream Matches | enthusiastic | 2 | accepted by A | — |
| 5 | 162 | 207 | World Cup Streaming | frustrated | 2 | accepted by A | — |
| 6 | 141 | 141 | Vuvuzela Complaints | negative | 2 | accepted by A | — |
| 7 | 133 | 216 | American Football Interest | critical | 2 | accepted by A | — |
| 8 | 130 | 202 | Referee Criticism | frustrated | 2 | identical | — |
| 9 | 117 | 141 | World Cup Pics | enthusiastic | 1 | identical | — |
| 10 | 116 | 175 | Real Madrid Transfer Rumors | excited | 2 | accepted by A | — |
| 11 | 114 | 124 | jersey | enthusiastic | 2 | accepted by B | — |

**The taxonomy of those 12 topics** (agreed in round 2):

| Category | Name | Topics | Posts | Share |
| --- | --- | --- | --- | --- |
| cat\_1 | World Cup Predictions and Streaming | 1, 4, 5, 9 | 906 | 35.1 % |
| cat\_2 | World Cup Match Predictions | 3 | 237 | 9.2 % |
| cat\_3 | Referee Criticism | 8 | 202 | 7.8 % |
| cat\_4 | EPL Support and Club Engagement | 2, 10, 11, 0 (0 placed by repair) | 881 | 34.1 % |
| cat\_5 | Vuvuzela Complaints | 6 | 141 | 5.5 % |
| cat\_6 | American Football Interest | 7 | 216 | 8.4 % |

**Outlier calibration candidates for bin 1** (quantiles of outlier-to-nearest-topic similarity; 100-post judgement samples written for each):

| Quantile | Threshold T | Outliers reassigned (of 3,523) | Outliers remaining |
| --- | --- | --- | --- |
| 50th | 0.535 | 1,764 | 17.9 % |
| 65th | 0.575 | 1,234 | 23.3 % |
| 80th | 0.621 | 705 | 28.7 % |

## Findings and issues

Two findings bear on the method, two on the model, and six defects were found and fixed during the run.

1. **D1 — a 3,000-post subsample collapses to 2 topics.** At the design's floor (`min_cluster_size` 25, `min_samples` 10) HDBSCAN's excess-of-mass selection returned one cluster of 2,859 posts plus a vuvuzela cluster; on the same array, 15 gives 4 topics, 10 gives 65 topics with 40 % outliers, and leaf selection at 25 gives 29 topics with 52 % outliers. This is a property of the small sample, not a code defect; the full bin did not collapse. Two consequences for the gates: collapse is not monotonic in the minimum size, so the guard (which only moves up from the floor) cannot rescue a bin that collapses at 25; and the largest topic's share should be read alongside the topic count at gate 1. Options if a real bin collapses: a lower floor for tournament bins, or leaf selection.
2. **N11 — the 1.7B model's JSON syntax.** In all four taxonomy attempts (both analysts, both repair turns) the model wrapped every category in its own list (`}], [{` between categories, `]]]` endings). The content was a valid 4-category partition each time; the strict parser rejected it, the repair turn reproduced it, and the stage ended with no categories. The recovery step reads each category's fields individually, only after the strict parse fails and never on truncated answers; every transcript records `parsed_via` and the original error. With it, the second run agreed in round 2 and 5 of 6 answers were recovered.
3. **The agreement rule is strict.** Identity requires all five fields to match after normalisation. Topic 1 ended as a stalemate although both analysts named it "World Cup Predictions"; they differed in cues and description and Analyst A repeated its own answer in round 2 (the design's stalemate rule). Under greedy decoding the usual path is "A accepts B's answer in round 2" (8 of 12 topics).
4. **Jabulani** is not in any keyword list of the 92 topics; in the five-month window run it was.
5. **Timing.** 8.7 s per prompt with batch 4 on MPS (15 s unbatched in session 02); all 92 topics of bin 1 take about an hour locally; the design's A100 estimates stand.

**Defects found and fixed during the dry run:** a NaN-handling bug when mapping overridden topic ids in S5; `reassign` re-derived the document set instead of reading `documents.csv`; the taxonomy syntax problem above; an empty taxonomy crashed S5 (now non-fatal, the bin counts as labelled, not complete); validation check A7 was too strict for a partial table (restricted to tagged rows, a new A7b added); and a zsh quoting quirk in the run scripts (operational only). Unit checks went from 13 to 15.

## Decisions requested

One decision is needed before the Colab run; the others stay at the quality gates as the design planned.

**Now — N11, the tolerant JSON recovery (`LLM_JSON_SALVAGE` in `config.py`).**

- [ ] **Keep it on (my recommendation).** Strict parsing is always tried first, truncated answers are never recovered, and every recovered answer is marked in the transcript, so the audit trail shows exactly how often the model's syntax failed. The content validation (one category per topic; a partition of the topic ids) is unchanged.
- [ ] **Turn it off.** Design §7.1 is then applied literally: a second syntax failure counts as no answer. On the evidence of bin 1, stage S3b would then produce no taxonomy for most bins with the 1.7B model.
- [ ] **Change the prompt instead** (for example an explicit "one JSON object; `categories` is a single array"). Prompt v1.1 is an approved, logged text, so this would become v1.2 and bin 1 would be re-labelled; it may or may not cure a 1.7B model's habit.

**At the gates (session 06), unchanged from the design:** the outlier threshold `T` (from the judgement samples on bin 1 and a large bin), greedy versus seeded sampling (the F17 comparison), and `min_samples` for the large bins.

**New items for the gates, from this run:** whether to lower the floor or use leaf selection if a real bin collapses (D1); and, if `needs_review` exceeds 15 % on Colab, whether agreement on name and tone category should count as agreement rather than requiring all five fields (a refinement of D2, not a change of the cap).

## Running it on Colab next

The notebook `code/06b-analysis-pipeline-colab.ipynb` (pushed as commit `5b24895c`) is now a two-pass flow: a smoke test on bin 1 that asserts after every stage, then the real run. Nothing analytical lives in the notebook; every cell calls the same program that ran locally.

**Once, before the first session:** upload the folder `data/yearly datasets/` (17 files, 2.4 GB) to Google Drive as `MyDrive/fifa-reddit-communities/data/yearly datasets/`. Results, the embedding cache and the model weights stay on Drive afterwards, so a disconnect costs nothing finished.

**Steps:**

1. Open the notebook in VS Code with the Colab extension (New Colab Server, GPU, A100 40 GB preferred) or at colab.research.google.com from the GitHub repository.
2. Cells 0.1 to 0.4: the Drive mount is tested (the fallback downloads a shared folder link with `gdown`); the repository is cloned and the data copied to the machine's SSD; the five pinned packages are installed with `--no-deps` on top of Colab's RAPIDS stack and `pip check` must be clean; the 15 unit checks run on the Colab machine. If `pip check` names a broken requirement, fix that one package explicitly and re-run the cell.
3. Cell 1: stage S0; the cell asserts the nine counts.
4. **Pass 1 (15 to 20 minutes of GPU time):** embeddings and topics for bin 1 (asserts backend `cuml` and a converged guard), labelling of the 12 largest topics, the taxonomy, sentiment for 2010, a partial collect. Compare the printed numbers with section 4; with mpnet embeddings the topic count and outlier share will differ somewhat, and the point is that every stage runs on the GPU backend.
5. **Pass 2:** labelling of all bin-1 topics (resumes from pass 1), the taxonomy over all topics with `--force`, the optional comparisons (seeded sampling, Qwen3-4B) into separate folders, the large-bin gate on bin 4, calibration of `T` and `reassign`, bins 2 to 9 with the review pauses, sentiment for all years, a complete collect, copy back to the repository and stage S6 locally in R.

**What to check at each step:** `config.json` of the bin (`backend`, `guard.trials`, `guard.converged`, `n_topics_raw`, `outlier_share_raw`); `report.html`; `needs_review` at or below 15 % after labelling; `n_repaired` and `n_answers_salvaged` after the taxonomy; `post_variables.json` with `complete = true` at the end.

**What could not be verified from the Mac:** whether `drive.mount` works from the VS Code Colab kernel (hence the fallback), the result of `pip check` on the current Colab image, and whether `cuml.accel` takes over UMAP with the cosine metric (a silent fall-back shows as backend `cpu` and a long UMAP time). The detailed runbook is `docs/session-05_2026-10-23/03_runbook-session-06.md`.

**Estimates on an A100 (design §10):** pass 1 about 20 minutes; the whole corpus about 3 hours of compute (35 to 50 compute units), spread across the review pauses.

## Appendix: where everything is

| What | Where |
| --- | --- |
| Approvals and refinements (your decisions, N1–N11, R1–R13) | `docs/session-05_2026-10-23/00_approvals.md` |
| Implementation notes, module map, the approved plan | `docs/session-05_2026-10-23/01_implementation-notes.md` |
| Dry-run log with every number and defect | `docs/session-05_2026-10-23/02_dry-run-log.md` |
| Runbook for the Colab session | `docs/session-05_2026-10-23/03_runbook-session-06.md` |
| Design document (v1.1: decisions of 23 Oct, §17 implementation notes) | `docs/session-04_2026-10-22/01_analytical-pipeline-design.md` |
| Codebook v3.0 draft (§4.9 wording for `topic_id_raw`, `agreement_mode`) | `data/codebook_soccer_posts.md` and `.pdf` |
| Stage program and package | `code/06-analysis-pipeline.py`, `code/topicmodel/` |
| Colab notebook and its requirements | `code/06b-analysis-pipeline-colab.ipynb`, `code/requirements-colab.txt` |
| Stage S6 | `code/07-tag-corpus.R`, `code/08-validate-analysis.R` |
| Unit checks | `code/tests/test_pipeline_units.py` |
| Real stage-S0 output | `data/analysis/period_bins.csv`, `period_bins_summary.csv` (shared data folder, not git) |
| Dry-run outputs (not in git) | `results/dry-runs/bin1-full/` (12-topic run: `report.html`, `topics_labelled.csv`, `labels/topic_<id>.json` transcripts, `taxonomy.json`, `outlier_calibration/`), `results/dry-runs/bin1-full92/` (the 92-topic run in progress), `results/dry-runs/bin1-3000/` (the subsample) |
| Session memory | `memory/2026-10-23_session-05.md`, `memory/workflows.md` §7 |

**Commits on `main` (23 Oct 2026):** `8061b4cc` housekeeping and approvals · `b740e308` implementation · `46ba904b` dry run, fixes, N11 · `5b24895c` two-pass Colab notebook.

**Timings on the M4 (MiniLM):** S0 3 min · embeddings 2010 22 s · S2 bin 1 1.9 min · reassign 20 s · S3a 12 topics 6.3 min · S3b 12 items 6.5 min · S4 2010 3 s · S5 7 s · R scripts 4 s each · unit checks 1 min.
