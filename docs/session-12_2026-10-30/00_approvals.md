# Session 12 (2026-10-30): the reference model on the whole corpus — decisions W1–W4

*Participants: Alan Nemirovski (decisions), Claude (reading, plan, code, documents). Planned in plan mode on 2026-10-30, after
session 11 had closed; Alan answered one round of questions and approved the plan before any file in the repository was changed.
Companion: `01_reference-model-notes.md` (settings, the stop point, how to read the mapping tables). Design §6.5.*

## 1. The request

Alan's words: "As an interim step, I want to re-run the topic modelling on the whole cleaned corpus (not separated by bins) to
replace the topic cross walking that would have had to be manually validated. Please enter planning mode to implement
changes/adaptations to existing coding files to set up a full-cleaned-corpus topic modelling run and 2-agent labelling. The goal
is to have a separate (reference) list of topics with labels, from which the QUALTAG attributes can be mapped onto based on the
manual coding done by TP and BA across the period bins so far."

Read as: the same pipeline (BERTopic on the mpnet embeddings, D9 tiers, the two Qwen3-1.7B analysts) fitted once on all
1,302,021 posts; its labelled topics kept apart from the nine bins' outputs as a corpus-wide reference list; the period
topics — and so the QUALTAG coding attached to them — mapped onto it. It replaces the manual crosswalk of topics to chapter
themes (`topic_crosswalk.csv → chapter_theme`, design F4/F43), a blank template to this day.

## 2. What the exploration established before the questions

- The mpnet embedding cache is keyed by calendar year (17 `.npz` files on Drive covering every post): a whole-corpus run
  re-embeds nothing.
- Every per-bin stage takes an integer bin; `bin-N` and the range 1–9 were hard-wired in about a dozen places (the CLI, the bin
  lookup, the path builder, the status table, the collect loop, the report's markers, the review pack, the notebook, the R
  loops). `collect` loops over bins 1–9 only, so a run in another folder cannot enter `post_variables.csv`.
- The density guard already had a disabled path (one HDBSCAN fit at the floor of 25), used by `--max-docs`.
- Timings on the A100: bin 8 alone took 38 min for 217k posts and 967 topics (UMAP 0.1 min, HDBSCAN 2 s, BERTopic fit 4 min,
  33 min of representatives, topics over time, model save and report); labelling 2.8–4.0 s per topic. The floor-bound interim
  bins gave 44.6–48.9 raw topics per 10,000 posts at `min_cluster_size` 25.
- The QUALTAG coding: TP's five tournament bins (1,385 topics, `QUALTAG` + 11 `qt_*` columns; 41 topics to settle), BA's bins
  2 and 4 (`qt_*` typed in directly, `1`/`0`/blank, 122 + 129 rows filled, blank undocumented), bins 6 and 8 uncoded; all keyed
  by bin-local `topic_id`.
- Two things a review of the change list found: the dense outlier-similarity matrix of the pipeline (`cosine_similarity` of all
  raw outliers × all topics, then a full `argsort`) would need an estimated 14 GB + 28 GB for the corpus (≈ 580k outliers ×
  thousands of topics) — more than Colab's RAM; and the debate engine answers round 1 for every topic before the first record
  is written, so a disconnect two hours into a 6,000-topic labelling would lose everything.

## 3. Decisions (Alan Nemirovski, 2026-10-30)

| # | Question | Decision | Why / note |
|---|---|---|---|
| **W1** | Granularity of the reference model: the design's density rule D8 (50–75 topics per 10,000 posts) would target 6,450–9,670 topics (6–8 h of labelling); a coarser fixed range of 1,000–1,500 topics would sit at the 'theme' level of the manual crosswalk (≈ 1 h) | **"No preset density target or topic target."** HDBSCAN at the design's fixed settings (`min_cluster_size` 25, `min_samples` 10, eom, ε 0) with the guard off; the topic count is whatever results | Possible as asked: the guard's disabled path does exactly this; recorded as guard mode `fixed` (distinct from `disabled`, a debug subsample). Consequence: the count is unknown until the topics stage has run, so the notebook has a **stop point** that shows the count and the labelling estimate before the labelling cell (§5 of the notes) |
| **W2** | Scope beyond the modelling and the labelling | **Model + labels + the mapping tables.** An R script (`code/16-reference-map.R`) writes the period-topic ↔ reference-topic mapping by posts and by centroid similarity, and every reference topic's QUALTAG composition as shares; **the rule that turns shares into a tag on a reference topic is decided later**, once the tables exist | Declined for now: a provisional transfer rule (e.g. political when more than half of the coded posts come from political period topics); the alternative "model + labels only" |
| **W3** | The reference topic as a post variable | **Separate file only.** The reference assignment lives in the run's own `documents.csv`; `post_variables.csv`, `labels_version`, `collect`, S6 and the codebook are untouched | The period topic stays the unit of analysis (D7); declined: new analysis columns (`ref_topic_id`, …) in the collect output and the codebook |
| **W4** | The run's large outputs: the post table ≈ 85 MB, the transcripts ≈ 150 MB (GitHub refuses files over 100 MB and the web upload over 25 MB; the bins' equivalents are tracked, bin 4's `documents.csv` 22 MB) | **Kept out of git.** `documents.csv` and `labels/` live on Drive and in the hand-back zip (`.gitignore`); the small tables are committed | Declined: committing everything but `model/` as for the bins (the repository would grow by ≈ 250 MB); a compressed post table |

## 4. Defaults stated with the plan (not objected to)

- Same embeddings, UMAP, D9 tiers (from the run's own similarity quantiles), K1 counting, prompts v1.4, Qwen3-1.7B revision,
  greedy decoding, 40 sample posts per topic. The reference run is **bin 0** inside the code (`period_bin` 0 in its label
  records; seeds `0-<topic>`); its folder is `results/analysis/reference/`; its period type is `corpus`; topics over time monthly.
- `reference/documents.csv` keeps each post's **true** `period_bin` (run 0 skips only the bin filter).
- The §12.2 gate is a bin concept and is not asserted for run 0: the notebook prints the raw outlier share beside the bins'
  0.50 at the stop point.
- The outlier judgement samples (4 × 100 posts) are written as for every run; nobody is asked to code them.
- `topics_over_time.csv` of run 0 is computed **without the model** (monthly counts of the counted posts, `keywords` empty) —
  K1 counts, whereas the bins' files keep the D9 counts (K7). The report's bar charts show the 200 largest topics above 2,000
  topics (every bin renders as before). No `review_pack.md` for run 0; `topic_overrides.csv` as a template; `REVIEW_READY.txt`
  worded "reference run".
- The labelling of run 0 runs in **chunks of 500 topics** inside one process (`--chunk-topics`, new; 0 = the bins' behaviour),
  so `labels.jsonl` grows every chunk.
- Git: besides `documents.csv` and `labels/` (W4), `umap_5d.npy` (≈ 26 MB), `hdbscan_labels_raw.npy` (≈ 10 MB) and
  `outlier_nearest.csv` (≈ 17 MB, est.) are ignored too; `labels.jsonl` (≈ 30 MB, est.; both analysts' final answers — the
  committed record of the labelling), `topics.csv`, `topics_labelled.csv`, `topics_over_time.csv`, `topic_embeddings.npy`,
  `outlier_calibration/`, `config.json`, `status.json`, the report, the log and the mapping tables are tracked.
- The mapping counts a post on either side only when it is **counted** there (K1); dual, outlier and not-modelled posts are
  reported as uncounted shares. TP's and BA's codings are reported in separate columns (BA's blank = uncoded, as found); the 41
  to-settle topics appear as "to settle". Figure G1 and its table go to `results/descriptives/`; the mapping tables to
  `results/analysis/reference/`.
- Records: this folder, design v2.0 (§6.5, §7.7, §9.3, F47, §14 W1–W4), `memory/workflows.md` §11, `memory/2026-10-30_session-12.md`,
  `docs/README.md`, `results/README.md`, `results/analysis/reference/README.md`, `results/descriptives/README.md`,
  `manual review/README.md`. The codebook is unchanged (W3).

## 5. What was changed and run (2026-10-30, after the approval)

**Code (all unit-checked; 17/17 checks, three new ones).**

- `code/topicmodel/config.py`: `REFERENCE_BIN` (bin 0, 2010-01-01 – 2026-07-31, period type `corpus`), `REFERENCE_GUARD_ENABLED
  = False` (W1), `TIME_BIN["corpus"] = "M"`, `LLM_CHUNK_TOPICS = 0`, `REPORT_CAP_ABOVE = 2000` / `REPORT_MAX_BARS = 200`,
  `run_dir_name()` / `bin_from_run_dir()`; a note that a per-type HDBSCAN dict needs a `default` key.
- `code/topicmodel/bins.py`: `bin_info(0)`; `load_bin(0)` keeps every post with its true `period_bin`.
- `code/topicmodel/model.py`: `outlier_nearest_from()` computes the two nearest topics in row chunks of 50,000 outliers
  (`argpartition`), identical to the full argsort it replaces (unit-checked, chunk sizes 1 and 3 against the one-shot result);
  `topics_over_time_counts()` (model-free monthly counts, the four columns of `topics_over_time`).
- `code/topicmodel/agents.py`: `run_labelling(..., chunk_topics)` debates the topics in consecutive chunks (the seeds are per
  topic, so no prompt or seed changes).
- `code/topicmodel/tagging.py`: `guard_state()` tests `enabled` first → `fixed` / `disabled`; `bin_status()` reads the bin from
  the folder name through `bin_from_run_dir`; `status_table()` appends the reference row when the folder exists;
  `write_status()` words run 0's flag; a comment at the collect loop (W3).
- `code/topicmodel/report.py`: the two bar charts capped above `REPORT_CAP_ABOVE` topics with the cap in the title; the
  heading reads "Reference model report" for run 0; every bin renders as before.
- `code/05-analysis-pipeline.py`: `--bin 0` (and `parse_bins` fixed: `0` had been falsy); `Paths.bin_dir` through
  `run_dir_name`; the guard off and `mode = "fixed"` for run 0; a check that run 0 holds every post of `period_bins.csv` (a
  warning when that count differs from the design table — a mini corpus); the model-free topics over time for period type
  `corpus`; the five kick-offs as the report's markers; `--chunk-topics` (recorded in `config.json → label`); the header text.
- `code/08-review-pack.py`: `--bin 0` resolves to `reference/` (no pack by default). `.gitignore`: the five paths of W4.
- `code/05b-analysis-pipeline-colab.ipynb`: cell 0.4's helpers generalised (`run_dir(b)`, `handback(bins, exclude, name)`,
  `handback_reference()`); the new section 6 (cells 6.1–6.5: cache check + embed, optional smoke run, topics + the stop point,
  labelling behind `RUN_REFERENCE_LABEL`, hand-back). Edited as JSON with the round trip asserted byte-identical first.
- `code/16-reference-map.R` (new): the mapping tables and figure G1 (§3.3 of the plan; `results/analysis/reference/README.md`).
- `code/tests/test_pipeline_units.py`: bin 0 in `test_bin_boundaries`; new `test_load_bin_reference`, `test_labelling_chunks`,
  `test_report_cap`; the fixed guard in `test_guard`; the model-free topics over time and the chunked nearest-topic search in
  `test_frozen_components_in_bertopic`; the reference folder beside bin 1 in `test_collect_partial` (listed by `status_table`,
  ignored by `collect`).

**Verification.**

- **Script 16 on a synthetic reference with a known answer** (reference topic i = the i-th period topic; every post keeps its
  period assignment with the ids remapped; the centroids are the bins' own), on the real `post_variables.csv`, labels table,
  bin embeddings and review files: **37 checks PASS**; `best_share` = 1 and `agree` = TRUE for all 6,245 period topics; the
  QUALTAG composition recovers TP's 215 yes / 1,074 no / 55 review / 41 to settle and BA's 159 `1` / 20 `0` / 2,339 blank
  topic counts; `unmapped` = 0; figure G1 rendered (every box at 100 %, as expected). ≈ 1 min.
- **The whole chain on a mini corpus** (17 yearly files of 600 sampled posts, 10,200 posts, under its own `PIPELINE_ROOT` in the
  scratchpad, MiniLM embeddings; nothing under the repository touched): every stage ran (≈ 10 min, CPU/MPS): `bins` (the design-count warning expected for a mini corpus), `embed --bin 0` (17 MiniLM cache files written under the scratch root), `topics --bin 0` (10,109 modelled posts; UMAP 0.3 min; one HDBSCAN fit at `min_cluster_size` 25 → 55 topics, 44.7 % raw outliers; `guard.enabled = false`, `guard.mode = "fixed"`; tiers T = 0.4785, C = 0.6108 → 905 reassigned, 2,035 dual, 15.6 % outliers left; four calibration samples; `topics_over_time.csv` with the four columns, empty keywords, monthly periods 2010-01 → 2026-07 and Σ n_docs = 6,493 = the counted posts; `report.html` headed "Reference model report" with the five kick-off markers; `documents.csv` with every post's true `period_bin` 1–9), `status` (the reference row, bin 0, guard `fixed`, after the nine missing bins), `reassign --bin 0` (same tiers), `recount --bin 0` (Σ n_total 6,493 → 6,493), `label --bin 0 --max-topics 3 --chunk-topics 1` (three topics named — Transfer Rumors, Goalkeeper Performances, Aston Villa vs Manchester Teams — one per chunk, `period_bin` 0 in `labels.jsonl`, `chunk_topics` recorded in `config.json`, `REVIEW_READY.txt` absent with 3 of 55 labelled), `sentiment`, `collect --allow-partial` (10,200 rows, bins 1–9 missing, no error: the reference folder is ignored) and `08-review-pack.py pack --bin 0` (resolves to `reference/`). The first attempt had stopped at the new corpus-size check, which compared the loaded documents with the design's 1,302,021; it now compares with `period_bins.csv` of the same root and only warns when that count differs from the design table.
- Re-rendering bin 1's report (`--stage report --bin 1`) and comparing with the committed file: the heading, the scope sentence
  and the chart titles are unchanged by the report edit; the file differs by its timestamp, by the K1 counts in the table (the
  committed report of 2026-10-25 shows the D9 counts, K7) and by plotly's HTML layout — restored from git afterwards.
- `git status`: no per-bin output, `post_variables.csv` or `labels_version` changed.

**Not run here:** the reference model itself (Colab, notebook section 6) — the stop point of cell 6.3 is where Alan decides
whether to label; the hand-back; script 16 on the real reference run.


**The pull before the commit** (`53122b6c`, t-pian, "Add files via upload") brought `results/analysis/manual review/BIN5_topics_labelled_TP_FINAL.csv`: TP's revised QUALTAG coding of bin 5 (448 topics in the 20-column layout of the pre-recode export — `QUALTAG` present, no `qt_*` columns; 27 `QUALTAG` cells differ from `BIN5_topics_labelled_TP.csv`, among them 3 of the 13 bin-5 topics on `QUALTAG_to_settle.csv`, e.g. the blank 5-234 → `non-political` and the conflict 5-222 → `player; Club; REVIEW`). The folder now holds **two bin-5 files**, and scripts 13, 14 and 16 stop on a duplicated bin by design. Which file stands, and the removal of the superseded one (as decisions Q5/Q6 did for bins 3 and 1), is Alan's call; nothing was changed.

## 6. Open after this session

1. **The Colab run** (notebook section 6): 6.1–6.3, the stop point (topic count, raw outlier share, labelling estimate), then
   6.4 and 6.5 with Alan's go-ahead; unzip the hand-back at the repository root; `Rscript code/16-reference-map.R`; commit the
   small tables and the mapping tables.
2. **The QUALTAG transfer rule** onto the reference topics (W2), after reading `reference_topics.csv` and figure G1; then the
   tags onto posts.
3. **TP's `BIN5_topics_labelled_TP_FINAL.csv`** (pulled 2026-10-30, above): decide which bin-5 file stands and remove the other
   (`git rm`), then `Rscript code/13-recode-qualtag.R` (the FINAL file has no `qt_*` columns yet) and `14`; 27 tags changed.
4. Still open from session 11: S6 in place and codebook v3.0 final (K5); BA's bin-2/4 format; TP's remaining to-settle topics; the
   inter-coder check at the editing stage (K2).
