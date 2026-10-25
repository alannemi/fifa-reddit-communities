# Session 07 (2026-10-25) — Code harmonization notes: one pipeline under `code/`, the path root, the renames

**Author of the code: Claude; decisions: Alan Nemirovski** (`00_approvals.md`). Read with the design
(`../session-04_2026-10-22/01_analytical-pipeline-design.md`, v1.3) and the implementation notes of sessions 05 and 06
(`../session-05_2026-10-23/01_implementation-notes.md`, `../session-06_2026-10-24/01_implementation-notes.md`), which still
describe how each stage works. This note says what was removed, moved or renamed, how the paths are now resolved, what
the notebook and the R scripts look like after the clean-up, what did **not** change, and how it was verified. The plan
approved before implementation is reproduced in §9.

## 1. `code/` after the session (one flow, in pipeline order)

| File | Role |
|---|---|
| `01-data-inspection.R` | load the raw export for interactive inspection (standalone) |
| `02-data-cleaning.R` | rebuild every cleaned file from the raw export (seven filter stages) |
| `03-data-validation.R` | 28 checks on the cleaned files; C4 accepts the v2.1 spec (42 columns) or the v3.0 spec (42 + 8) and reports which it found (H7) |
| `04-render-codebook.R` | codebook Markdown → PDF (standalone; Chrome path via `CHROME_PATH`) |
| `05-analysis-pipeline.py` | **the analytical pipeline** — stages bins, embed, topics, reassign, label, taxonomy, sentiment, collect, report (was `06-analysis-pipeline.py`) |
| `05b-analysis-pipeline-colab.ipynb` | its orchestration on a Colab GPU, rooted on `MyDrive/fifa-communities/` (was `06b-…`; regenerated) |
| `06-tag-corpus.R` | stage S6: append the eight analysis columns to the master and yearly files (was `07-tag-corpus.R`) |
| `07-validate-analysis.R` | stage S6 validation, 13 check families A0–A12 (was `08-validate-analysis.R`) |
| `R/pipeline-utils.R` | **new**: the helpers and column specifications shared by 02, 03, 06 and 07 (H4) |
| `topicmodel/` | the Python package behind 05: `config` (every setting; the path root), `data`, `bins`, `embed`, `model`, `prompts`, `agents`, `sentiment`, `tagging`, `report` |
| `tests/test_pipeline_units.py` | 20 unit checks without GPU or language model (19 + the path root and tone vocabulary) |
| `requirements-topicmodel.txt`, `requirements-colab.txt` | unchanged pins; headers name the pipeline |

Removed (decision H1; git history up to commit `1ad6e489`): `05-topic-model.py`, `05b-topic-model-colab.ipynb`,
`topicmodel/label.py`, and in the shared modules every function and setting only they used (§3).

## 2. The path root (default of the plan; design §10 note)

`config.py` resolves the five persisted paths once, from two environment variables, through the pure function
`resolve_paths(pipeline_root, yearly_dir)`; every bin's `config.json` records them.

| Setting | repository layout (`PIPELINE_ROOT` unset) | `PIPELINE_ROOT=<R>` (flat Drive layout, H2) | holds |
|---|---|---|---|
| `YEARLY_DIR` | `data/yearly datasets` | `<R>/yearly datasets` (or `PIPELINE_YEARLY_DIR`) | the 17 input CSVs |
| `ANALYSIS_DATA_DIR` | `data/analysis` | `<R>/analysis` | join tables by post id |
| `ANALYSIS_DIR` | `results/analysis` | `<R>/results` | `bin-N/` + aggregated tables |
| `EMBED_CACHE_DIR` | `results/embeddings` (was `results/topic-models/embeddings`) | `<R>/embeddings` | `<model-slug>/<year>.npz` |
| `LLM_CACHE_DIR` | `.hf-cache` | `<R>/hf-cache` | Hugging Face weights |

Precedence: `--out-root <O>` (dry runs; writes `<O>/results/analysis` and `<O>/data/analysis`, repository-shaped so the R
scripts read them) wins over `PIPELINE_ROOT` for those two folders only; the input and the caches are never redirected by
it (F33). The stage program sets `HF_HOME` to `LLM_CACHE_DIR` when unset (every loader already passes `cache_dir`), so the
`export HF_HOME=…` line disappears from the instructions. Every stage logs the effective paths (`Paths.describe()`) and
`post_variables.json` stores them. On Colab the notebook sets `PIPELINE_ROOT=/content/drive/MyDrive/fifa-communities` and
`PIPELINE_YEARLY_DIR=/content/data/yearly datasets` (the local copy of the CSVs, H3) and prints the resolved paths (F34).

## 3. Deleted, moved, renamed

**Settings (`config.py`).** Deleted, exploratory tool only: `WINDOW_MONTHS`, `DATA_END`, `ERAS`, `HDBSCAN_MIN_CLUSTER_FRACTION`,
`HDBSCAN_MIN_CLUSTER_FLOOR`, `MAX_TOPICS_BEFORE_REDUCE`, `REDUCE_OUTLIERS`, `TIME_BIN` (the year/era/window map),
`ERA_MERGE_MIN_SIMILARITY`, `LLM_MAX_NEW_TOKENS`, `LLM_TEMPERATURE`, `LLM_MAX_TURNS`, `LLM_AGENTS`, the functions
`_shift_months` and `world_cup_window`, `DATA_DIR`, `RESULTS_DIR`. Deleted, read by nothing: `LLM_MODEL_LARGE` (Qwen3-4B,
dropped 2026-10-24), `LLM_MODEL_ALT` (Phi-4-mini), `OUTLIER_STRATEGY`, `SENTIMENT_TEXT`. Renamed (F31; the keys of every
future `config.json` change accordingly; no committed run exists):

| before | after |
|---|---|
| `EMBEDDING_MODEL` (MiniLM) | `EMBEDDING_MODEL_DRY_RUN` — local dry runs only |
| `EMBEDDING_MODEL_HQ` = `ANALYSIS_EMBEDDING_MODEL` (mpnet) | `EMBEDDING_MODEL` — the pipeline's model |
| `BIN_MIN_CLUSTER_FRACTION`, `BIN_MIN_CLUSTER_FLOOR` | `MIN_CLUSTER_FRACTION`, `MIN_CLUSTER_FLOOR` |
| `BIN_TIME_BIN` | `TIME_BIN` (`{"tournament": "W", "interim": "M"}`) |
| `LLM_MAX_NEW_TOKENS_TAXONOMY` | `LLM_MAX_NEW_TOKENS_PARTITION` (partition protocol only) |
| `TONE_SET` (unused) + `label._TONE_STEMS` | `TONE_SET` + `TONE_STEMS`, checked at import to agree; the mapping is unchanged |
| — | `YEARLY_FILE`, `PIPELINE_ROOT`, `PIPELINE_YEARLY_DIR` (recorded in `config.json`) |

`effective_settings()` now also records the path constants (as strings).

**Functions.** `model.py`: deleted `min_cluster_size`, `build_topic_model` (exploratory), `fit_model` (exploratory),
`topic_table` (exploratory), `topics_over_time` (exploratory), `merge_year_models`; renamed `build_bin_topic_model` →
`build_topic_model`, `fit_bin_model` → `fit_model`, `bin_topic_table` → `topic_table`, `bin_topics_over_time` →
`topics_over_time` (the only builders left); `save_model`/`load_model` kept. `data.py`: deleted `load_years`,
`select_window`; added `yearly_file()`, `available_years()`, `n_tokens()` — the yearly file name pattern and the token rule
each have one definition (used by `bins`, `sentiment`, the stage program). `embed.py`: `embed_year(…, force=)` absorbs the
stale-cache deletion the stage script did itself; `cached()` added; the stage program no longer calls a private function.
`sentiment.py`: deleted `score_documents`, `by_topic`, `by_period`; `label_of()` is the single label rule (also used by the S5
checks in `tagging.py`); normal imports and the shared logger. `report.py`: deleted `write_tables`; `write_report(run_dir,
meta, topics, tot, sent_topic, markers)` reads the pipeline's own column names (`topic_id`, `n_cluster`, `n_total`, `name`,
`emotion_tone`, `description`, `needs_review`, `category_id`, `category_name`), shows the category of each topic, and drops
the unused `docs` parameter and the never-produced intertopic map (F36); the page is titled "Period bin report".
`agents.py`: `tone_category()` moved in from `label.py` (reads `config.TONE_STEMS`); the token budget of a debate is an
explicit per-stage table (label / define / assign / partition — no implicit fallback to the partition value);
`AGREEMENT_MODES` is asserted when a result is written; the partition-protocol docstrings say so. `bins.py`: the inline
min-cluster formula → `model.start_min_cluster_size`. The stage program: header rewritten for sessions 04–07, `Paths.describe()`,
`stage_embed` through `embed.embed_year`, `render_report` without the column-rename shim.

**Docstrings and comments** now state the current decision first (two-step taxonomy D5; Qwen3-1.7B only; mpnet for real
runs and MiniLM for local dry runs only; title-only `post_sentiment`) and keep a short dated provenance only where it
explains a value (8,000 tokens ← review R2; salvage ← N11; floor 25 ← design §6.1).

## 4. The notebook `05b-analysis-pipeline-colab.ipynb` (28 cells, regenerated)

Generated with a stdlib script kept in the session scratchpad (`build_05b_notebook.py`; nbformat 4.5, validated with
`nbformat.validate`, every code cell parsed). Cells: header with the Drive layout table → 0.1 mount, assert the 17 files,
create `analysis/ results/ embeddings/ hf-cache/ logs/ handback/ comparisons/`, open the session log → 0.2 clone/pull the
repository, copy the CSVs to `/content/data/yearly datasets`, set `PIPELINE_ROOT`, `PIPELINE_YEARLY_DIR`, print and assert the
effective paths → 0.3 `--no-deps` install + `pip check` + GPU/cuML/pins (unchanged) → 0.4 helpers (`run_logged` builds its
environment per call, `stage()` calls `05-analysis-pipeline.py`, `cfg()` and `taxonomy_summary()` read `results/bin-N/` on
Drive) → 1 S0 → 2.x Pass 1 (unchanged logic) → 3.1 full bin 1 = stop point of the test batch → 3.2 optional seeded comparison
into `comparisons/cmp-seeded` → 3.3 large-bin gate → 3.4 `T` → 3.5 bins 2–9 → 4 sentiment + collect → 5 hand-back zip that
rebuilds the repository paths (`results/analysis/bin-N/…`, `results/analysis/*.csv`, `data/analysis/*`, `logs/`) from the flat
folders. No symlinks; no `HF_HOME`; no `gdown` fallback (the input folder exists; if `drive.mount` fails in the VS Code
kernel the cell says to open the notebook at colab.research.google.com).

## 5. The R scripts

`code/R/pipeline-utils.R` holds `log_msg`, `check` (one width), `read_chr(p, …)`, `col_hash`, `find_root(sentinel)` (the
working directory, then the running script's own location; the OneDrive fallbacks are gone — the repository lives wherever
it was cloned), `ORIGINAL_COLS` (the 42 names of codebook v2.1), `N_ORIGINAL`, `ANALYSIS_COLS` (the 8 of §4.9), the bin
table (`BIN_START`, `BIN_END`, `BIN_TYPE`) and `bin_of()`. Each of 02, 03, 06 and 07 starts with a four-line bootstrap that
finds the helper file from `--file=` (Rscript) or from the working directory (RStudio), sources it and does `setwd(ROOT)`.
* `02`: logic untouched; its own `log_msg`/`find_root` removed.
* `03`: C4a/C4c accept the v2.1 or the v3.0 spec and print which one (H7); 28 checks as before.
* `06-tag-corpus.R`: `TAG_IN_ROOT` defaults to the repository root (was `.`), `TAG_OUT_ROOT` to `TAG_IN_ROOT`; the pre-flight
  compares the column names with `ORIGINAL_COLS` (was a count).
* `07-validate-analysis.R`: the same two variables with the same defaults (the direction of the default was reversed in the
  old `08`), `TAG_ANALYSIS_DIR` replaces the env name `ANALYSIS_DIR` (which collided with the Python setting); new check **A0**
  compares the pipeline's own `period_bins_summary.csv` (when present) with the mirrored bin table — the mirrored table is
  kept rather than read from the pipeline (plan refinement, review of 2026-10-25) so that R still checks S0 independently and
  dry-run roots without the summary file keep working; `expected_cols` = `ORIGINAL_COLS` + `ANALYSIS_COLS`.
* `01`: the hard-coded `setwd()` is gone (run from the repository root); `04`: the OneDrive fallback is gone, the Chrome path
  reads `CHROME_PATH`.

## 6. Repository plumbing

`git rm`: `code/05-topic-model.py`, `code/05b-topic-model-colab.ipynb`, `code/topicmodel/label.py`; `git rm --cached`
(files stay on disk): `data/jsonl_to_csv.py`, `data/data-inspection.R` (H6). `git mv`: the four renames of H5. The local
embedding cache moved from `results/topic-models/embeddings/` to `results/embeddings/` (`mv`; gitignored; `.gitignore`
follows, and its redundant `/data/analysis/sentiment/` line is gone).

## 7. What did not change

The stages and their order; every setting's value (only names changed); the prompts (v1.2) and the two-step protocol; the
D2 debate engine; the partition protocol as a switch (F26); the output files and their columns (`documents.csv`,
`topics.csv`, `topics_labelled.csv`, `taxonomy.json`, `categories.csv`, `topic_categories.csv`, `post_variables.csv`, the
aggregated tables); the eight codebook variables; the R write-back and its fingerprints; the Colab two-pass structure and
the stop point of the test batch.

## 8. Verification (2026-10-25, Apple M4, `.venv`)

| Check | Result |
|---|---|
| `code/tests/test_pipeline_units.py` | **20/20 PASS** (19 + `test_paths_and_tones`) |
| dangling `C.<NAME>` references across `code/` | none |
| `--stage bins` against the existing `data/analysis/period_bins.csv` | skipped as expected; the effective paths logged |
| `PIPELINE_ROOT=/tmp/h7-root PIPELINE_YEARLY_DIR=<repo>/data/yearly datasets` | flat layout resolved (`/tmp/h7-root/{analysis,results,embeddings,hf-cache}`; the yearly override honoured) |
| local MiniLM dry run of S1 + S2 on bin 1 (`--out-root results/dry-runs/h7-bin1`) | cache hit; 92 topics at `min_cluster_size` 25, 35.9 % raw outliers, 2.1 min; `report.html` rendered from the native columns |
| comparison with the session-05 dry run (`results/dry-runs/bin1-full`) | `umap_5d.npy` and `hdbscan_labels_raw.npy` **bit-identical**; `n_cluster` and `representative_ids` identical; 1 of 93 keyword lists differs — the same words in a different order (`rafael benítez` / `rafael benitez`; design §11: candidate-word embeddings on MPS are not bit-deterministic) |
| `06-tag-corpus.R` dry run on the session-05 mini corpus (`r-mini` → `r-out2`) | 13,638 rows tagged, 50 columns, fingerprints written |
| `07-validate-analysis.R` on `r-out2` (`TAG_ANALYSIS_DIR` = the dry run's `results/analysis`) | **ALL CHECKS PASSED** (A0 skipped: no summary file in the mini root; A2c skipped; A12 noted as partial) |
| `03-data-validation.R` on the real corpus (28 checks, ≈ 5 min) | **28/28 PASS**; C4a reports `v2.1 spec (42 columns)`; `find_root` from the shared helper file |
| `05b-analysis-pipeline-colab.ipynb` | `nbformat.validate` OK, 28 cells, unique ids, every code cell parses; no `fifa-reddit-communities` Drive path, symlink, `HF_HOME` or old script name |

Nothing ran on Colab in this session; the test batch is the first step of `02_runbook-colab-run.md`.

## 9. The approved plan (2026-10-25)

The plan file approved before implementation (`~/.claude/plans/dapper-finding-parrot.md` on the author's machine) is
reproduced verbatim below. Two defaults were refined during execution, both recorded in §5 and §8: the R validator keeps a
mirrored bin table with the cross-check A0 instead of reading `period_bins_summary.csv` (dry-run roots without that file
keep working), and its environment variable is `TAG_ANALYSIS_DIR` (the old name collided with the Python setting).

---

# Session 07 plan — harmonize `code/` into one pipeline and re-root Colab on `MyDrive/fifa-communities/`

**Prepared 2026-10-25 for Alan Nemirovski (decisions) · FIFA Communities chapter.** Basis: the design document
(`docs/session-04_2026-10-22/01_analytical-pipeline-design.md`, v1.2), the approvals of sessions 05 and 06
(`docs/session-05_2026-10-23/00_approvals.md`, `docs/session-06_2026-10-24/00_approvals.md`), the runbook for this
session (`docs/session-06_2026-10-24/03_runbook-session-07.md`), `memory/workflows.md`, and the code as committed in
`1ad6e489` (2026-10-24). Markers: **[Q]** = asked in this planning round, **[default]** = applied when the plan is
approved unless you object, **[flag]** = a consequence you should know about.

## Context

`code/` carries two generations of the topic-modelling code. Generation 1 (session 02, 2026-10-13) is the exploratory
tool `05-topic-model.py` (calendar-year, ±2-month window and era runs) with `05b-topic-model-colab.ipynb` and the
modules `topicmodel/{config,data,embed,model,label,sentiment,report}.py`. Generation 2 (sessions 05–06) is the
chapter's pipeline `06-analysis-pipeline.py` (stages bins → embed → topics → reassign → label → taxonomy → sentiment →
collect → report), which added `topicmodel/{bins,prompts,agents,tagging}.py` and appended to `config.py`, `model.py`
and `sentiment.py` while leaving the generation-1 code in place "as the exploratory tool" (session-05 decision). The
2010 test outputs of generation 1 were deleted on 2026-10-23; generation 1 has not been run since, and the design (§14,
F6) says the period bins supersede its run structure. Result: duplicated logic (two labelling engines, two sentiment
APIs, two model builders, two notebooks), settings in `config.py` that only the dead tool reads, and comments that
still describe superseded states (single-prompt S3b, the dropped Qwen3-4B comparison).

The Colab notebook `06b-analysis-pipeline-colab.ipynb` assumes a Drive layout `MyDrive/fifa-reddit-communities/{data/yearly
datasets, results, logs, handback, .hf-cache}`. The author's Drive folder is `MyDrive/fifa-communities/` and today holds
only `yearly datasets/`. Every Drive path must move under `fifa-communities/`, and the notebook's working directory must
be based there.

Goal of this session: one consistent code flow under `code/` that matches the decisions D1–D5, N1–N12, R1–R40 and the
documented conventions (`memory/workflows.md` §5), with nothing analytical changed (same stages, same outputs, same
eight codebook variables), and a Colab notebook rooted on `MyDrive/fifa-communities/`.

## Inventory (three read-only exploration passes, 2026-10-25)

**Generation 1 code that nothing in the pipeline uses** (`05-topic-model.py`, `05b-topic-model-colab.ipynb`, and in
`code/topicmodel/`): `label.py` except `tone_category` + `_TONE_STEMS` (label.py:88-103, imported by `agents.py:46`);
`model.py` 20-100 (`min_cluster_size`, `build_topic_model`, `fit_model`, `topic_table`, `topics_over_time`) and 113-116
(`merge_year_models`; its default argument reads `C.ERA_MERGE_MIN_SIMILARITY` at import time); `sentiment.py` 17-42
(`score_documents`, `by_topic`, `by_period`); `report.py` 18-26 (`write_tables`); `data.py` 44-45 (`load_years`), 62-65
(`select_window`); `config.py` 65-77 (`_shift_months`, `world_cup_window`) and 13 settings (`WINDOW_MONTHS`, `DATA_END`,
`ERAS`, `EMBEDDING_MODEL`, `HDBSCAN_MIN_CLUSTER_FRACTION`, `HDBSCAN_MIN_CLUSTER_FLOOR`, `MAX_TOPICS_BEFORE_REDUCE`,
`REDUCE_OUTLIERS`, `TIME_BIN`, `ERA_MERGE_MIN_SIMILARITY`, `LLM_MAX_NEW_TOKENS`, `LLM_TEMPERATURE`, `LLM_MAX_TURNS`,
`LLM_AGENTS`). Unused anywhere: `LLM_MODEL_LARGE`, `LLM_MODEL_ALT`, `TONE_SET`, `OUTLIER_STRATEGY`, `SENTIMENT_TEXT`,
`agents.AGREEMENT_MODES`.

**What the pipeline (`06-analysis-pipeline.py`) uses from the shared generation-1 modules:** `embed.{get_device,
load_embedding_model, embed_year, embeddings_for, _cache_path}`; `model.{save_model, load_model}`; `report.write_report`;
`data.{load_year, build_documents, clean_text, _TOKEN}` through `bins` and `sentiment`; `label.tone_category`;
`config.{WORLD_CUPS, SEED, YEARLY_DIR, LLM_MODEL, effective_settings}`. `WORLD_CUPS` must stay: `PERIOD_BINS` rows carry
`world_cup=2010 …` and the report stage reads the kick-off/final dates from it (06:430).

**Duplication inside the pipeline itself:** the yearly filename pattern is written five times (data:39, bins:68,71,
sentiment:107, 06:392); the min-cluster-size formula twice (model:201-202, bins:108); the sentiment label rule three
times (sentiment:21, sentiment:61-62, tagging:217); `stage_embed` (06:164-177) re-implements `embed_year`'s cache check;
`build_bin_topic_model` copies `build_topic_model`'s vectorizer/c-TF-IDF/representation chain; `render_report`
(06:402-433) renames six columns so that the 05-era `write_report` accepts them, passes a `docs` argument the writer
never reads, and merges `category_id` that the writer never shows; `sentiment.py` has mid-file underscore imports
(53-58) and its own `_log`.

**Stale comments and docstrings** (all describe superseded states as current): `__init__.py:14` (prompt v1.1),
`__init__.py:17-19` (06 = "sessions 04–05", 05 not marked superseded); `label.py:13` (Qwen3-4B default); `config.py:9-11`
(session 02 as the main source), `:82-83` (MiniLM/mpnet without the dry-run rule), `:121-123` (Qwen3-4B, Phi-4-mini
as live options), `:197` (`LLM_MAX_NEW_TOKENS_TAXONOMY` as "the S3b answer"); `agents.py:542` (partition format as THE
S3b answer), `:628` (partition token budget as the default for every non-label stage); `model.py:8-9`, `embed.py:7-8`,
`sentiment.py:8-10`, `report.py:2-7` (05 flow only); `requirements-topicmodel.txt:1` (names 05); `tests:3-9` (omits
the two-step tests); `06:10` (session-05 approvals only), `06:168`.

**Notebook `06b`** (28 cells): `DRIVE = /content/drive/MyDrive/fifa-reddit-communities`, data expected at
`{DRIVE}/data/yearly datasets`, repo cloned to `/content/fifa-reddit-communities`, CSVs copied to the clone, four
symlinks (`results/analysis`, `data/analysis`, `results/topic-models/embeddings`, `.hf-cache` → Drive), session log
`{DRIVE}/logs/`, hand-back zip `{DRIVE}/handback/`, the optional comparison writes to `/content/cmp-seeded` (never
reaches Drive). No `.py`/`.R` file contains a Drive path; everything derives from `config.ROOT`. `--out-root` redirects
only `results/analysis` and `data/analysis`, not the embedding cache, the HF cache or the yearly input.

**R scripts:** `read_chr`, `check`, `log_msg`, `col_hash` defined in two or three scripts each; `find_root` in three
variants with hard-coded OneDrive fallbacks (`01` has a hard-coded `setwd`); the nine bin dates repeated in `08:54-56`;
`07` defaults `TAG_OUT_ROOT` to `TAG_IN_ROOT` while `08` does the reverse; `03` C4a/C4c expect 42/43 columns and would
fail after S6 (50). Outside `code/`, git tracks two stale copies: `data/jsonl_to_csv.py` (byte-identical to the
untracked original in `data/data artefacts/`) and `data/data-inspection.R` (older loader with type guessing).

**References to the file names** (live documents that must follow a rename): `README.md` (9 × `06-analysis-pipeline`,
3 × `07-tag-corpus`, 2 × `08-validate-analysis`, 2 × `06b`), `memory/workflows.md`, `docs/README.md`,
`docs/session-06_2026-10-24/03_runbook-session-07.md`, `data/codebook_soccer_posts.md` (§1 file table, §8.4), the code
headers themselves (`07`/`08` name each other; `tagging.py` names `07`). The dated session documents (sessions 01–06)
name the files 50+ times and are the historical record.

## Decisions taken in this planning round (Alan Nemirovski, 2026-10-25)

| # | Question | Decision | Alternatives offered and declined |
|---|---|---|---|
| H1 | Fate of the exploratory tool (05, 05b, label.py's debate, 05-only functions and settings) | **Remove entirely**; keep only what the pipeline uses (tone mapper, `save_model`/`load_model`, `WORLD_CUPS`); git history (`1ad6e489`) keeps it; session-02/03 docs stay as the record with a dated note | archive a read-only copy under `code/archive/`; keep runnable |
| H2 | Drive layout under `MyDrive/fifa-communities/` | **Flat, purpose-named:** `yearly datasets/` (input, exists) · `analysis/` (join tables) · `results/` (bin-1 … bin-9 + aggregated tables) · `embeddings/` · `hf-cache/` · `logs/` · `handback/` | repository-shaped subfolders |
| H3 | Colab working directory | **Drive is the root for everything that persists; the code is cloned from GitHub into the VM; the CSVs are copied once per session to the VM's SSD; outputs are written straight into the Drive folder through one root setting (no symlinks)** | everything on Drive (cwd on Drive); hybrid |
| H4 | R scripts | **Shared helper file** `code/R/pipeline-utils.R` sourced by 02, 03 and the two S6 scripts; `find_root` from a cwd sentinel (OneDrive fallbacks dropped); env-var defaults aligned; `01` and `04` stay standalone | self-contained but aligned; leave R untouched |
| H5 | Numbering after the removal of 05/05b | **Renumber:** `06-analysis-pipeline.py` → `05-analysis-pipeline.py`, `06b-…-colab.ipynb` → `05b-…-colab.ipynb`, `07-tag-corpus.R` → `06-tag-corpus.R`, `08-validate-analysis.R` → `07-validate-analysis.R`; live documents updated, dated session docs keep the old names with a note | keep the gap |
| H6 | Stale tracked copies outside `code/` | **`git rm data/jsonl_to_csv.py data/data-inspection.R`** (originals stay in `data/data artefacts/` and `code/01-data-inspection.R`; nothing in the shared data folder is deleted by the assistant) | leave `data/` alone |
| H7 | `03-data-validation.R` and the 50-column spec | **Extend now:** C4a/C4c accept 42 (v2.1) or 50 (v3.0) and report which spec they found | wait for the S6 commit |

## Defaults applied with this plan unless you object

* **One root setting.** `config.py` reads the environment variable `PIPELINE_ROOT`. Unset → the repository layout
  (unchanged for local work). Set → the flat Drive layout of H2 under that folder. `PIPELINE_YEARLY_DIR` overrides the
  input folder alone (the SSD copy on Colab). `--out-root` keeps today's dry-run meaning (redirects the two output
  folders as `<out>/results/analysis` and `<out>/data/analysis`, repository-shaped so the R scripts work on them) and
  takes precedence over `PIPELINE_ROOT` for those two folders only.
* **Local cache folder** moves from `results/topic-models/embeddings/` to `results/embeddings/` (`results/topic-models/`
  existed only for the exploratory tool); the existing MiniLM cache is moved, not recomputed. `.gitignore` follows.
* **Setting names lose the `BIN_`/`ANALYSIS_` prefixes** that only distinguished them from the exploratory tool (table in
  §B). `config.json` keys change accordingly; no committed run exists yet, so nothing depends on the old keys (the
  gitignored dry-run folders keep theirs as history).
* **Tone vocabulary** becomes a reviewed setting: the stem table moves from `label.py` to `config.TONE_STEMS`, the
  controlled set `TONE_SET` is asserted to cover every stem's target; the mapping itself is byte-identical.
* **Report writer** consumes the pipeline's own column names, shows `category_id`/category name, and drops the
  intertopic map (never produced by the pipeline; the saved `model/` allows it later).
* **Notebook** is regenerated with `nbformat` from a generator script kept in the session scratchpad (the notebook is
  the committed artefact, as in sessions 05–06); the optional seeded-sampling comparison writes to
  `MyDrive/fifa-communities/comparisons/` instead of the VM's disk.
* **R bin table**: `07-validate-analysis.R` reads the bin boundaries from `data/analysis/period_bins_summary.csv`
  (written by stage S0) instead of a hard-coded copy of `PERIOD_BINS`.
* **Dated session documents** (sessions 01–06) are not rewritten; the renames and the Drive layout are explained once
  in a dated note at the top of `docs/README.md`, and the session-06 runbook gets a one-line pointer to its successor.
* **Qwen3-4B and Phi-4-mini** settings are deleted (decisions of 2026-10-24 and 2026-10-13); the `--llm` option still
  accepts any model id.

## Implementation

### A. Path root (`code/topicmodel/config.py`, `05-analysis-pipeline.py`)

```
                       unset (local)                       PIPELINE_ROOT=<R> (Colab, flat)      --out-root <O> (dry run)
YEARLY_DIR             data/yearly datasets                <R>/yearly datasets  (or PIPELINE_YEARLY_DIR)   unchanged
ANALYSIS_DATA_DIR      data/analysis                       <R>/analysis                        <O>/data/analysis
ANALYSIS_DIR           results/analysis                    <R>/results                         <O>/results/analysis
EMBED_CACHE_DIR        results/embeddings                  <R>/embeddings                      unchanged
LLM_CACHE_DIR          .hf-cache                           <R>/hf-cache                        unchanged
```
* `config.py`: a pure function `resolve_paths(root: str | None, yearly: str | None) -> dict` computes the five paths;
  module constants are set from it at import using `os.environ`; `effective_settings()` records them (as today).
* The stage script sets `HF_HOME` to `LLM_CACHE_DIR` when unset (every loader already passes `cache_dir`), so the
  "export HF_HOME" line disappears from the header and the docs; `TOKENIZERS_PARALLELISM` stays an environment hint.
* `Paths` (stage script) is built from `C.ANALYSIS_DIR` / `C.ANALYSIS_DATA_DIR` and the `--out-root` override; it is
  logged at the start of every stage and written into `post_variables.json` (as `out_root` is today).

### B. Modules in `code/topicmodel/` (same stages, same values, same outputs)

| File | Delete | Change |
|---|---|---|
| `config.py` | the 13 exploratory settings and `_shift_months`/`world_cup_window` (lines 52-77, 92-93, 98-99, 114-115, 121-128 except `LLM_SAMPLE_POSTS`); `OUTLIER_STRATEGY`, `SENTIMENT_TEXT`, `LLM_MODEL_LARGE`, `LLM_MODEL_ALT` | new header (pipeline-first; sessions 04–07); section order: paths (with `resolve_paths`) → documents → period bins (`PERIOD_BINS`, `WORLD_CUPS`, `PERIOD_BIN_EXPECTED_N`, `TIME_BIN`) → embeddings → BERTopic + guard → outliers → LLM and debate → taxonomy protocol → tone vocabulary → sentiment → `effective_settings`. Renames: `EMBEDDING_MODEL_HQ`+`ANALYSIS_EMBEDDING_MODEL` → `EMBEDDING_MODEL` (mpnet) and `EMBEDDING_MODEL` (MiniLM) → `EMBEDDING_MODEL_DRY_RUN`; `BIN_MIN_CLUSTER_FRACTION/FLOOR` → `MIN_CLUSTER_FRACTION/FLOOR`; `BIN_TIME_BIN` → `TIME_BIN`; `LLM_MAX_NEW_TOKENS_TAXONOMY` → `LLM_MAX_NEW_TOKENS_PARTITION`; `TONE_STEMS` added. Comments state the current decision first and keep a short dated provenance only where it explains a value (8,000 ← R2; salvage ← N11; floor 25 ← design §6.1) |
| `data.py` | `load_years`, `select_window` | add `yearly_file(year)`, `available_years()` (the glob) and `n_tokens(text)`; `load_year` uses `yearly_file`; docstring |
| `embed.py` | — | `embed_year(..., force=False)` absorbs the stale-cache deletion that `stage_embed` does today; `_cache_path` stays private; docstring (bins, not year/window/era runs) |
| `model.py` | lines 20-100 and 113-116 (five exploratory functions and `merge_year_models`) | `build_bin_topic_model` → `build_topic_model`, `bin_topic_table` → `topic_table`, `bin_topics_over_time` → `topics_over_time` (the only builders left); imports at the top (no `_json/_time/_Counter`); docstring describes the frozen-component pipeline |
| `label.py` | the file | `tone_category()` moves to `agents.py` (section "controlled tone vocabulary"), reading `C.TONE_STEMS` |
| `sentiment.py` | lines 17-42 (`score_documents`, `by_topic`, `by_period`) | normal imports and `log`; `validation_sample` uses `data.yearly_file`; `label_of` is the single label rule (also used by `tagging.py:217`); docstring |
| `report.py` | `write_tables`, the `docs` and `bt_model` parameters | `write_report(run_dir, meta, topics, tot, sent_topic, markers)` reads `topic_id`, `n_total`, `share`, `keywords`, `name`, `emotion_tone`, `description`, `needs_review`, `category_id`, `category_name`, `representative_titles`; page title "Period bin report"; marker keys `kick-off`/`final` |
| `agents.py` | the `from .label import tone_category` line | explicit per-stage token budgets (label → `_LABEL`, define → `_DEFINE`, assign → `_ASSIGN`, partition → `_PARTITION`; no implicit fallback); `AGREEMENT_MODES` asserted when a debate result is written; docstrings at 542 and 996 say "partition protocol only (v1.1)" / "two-step (D5)" |
| `prompts.py` | — | section comments mark `SYSTEM_TAXONOMY`/`USER_TAXONOMY`/`render_taxonomy_items` as partition-only; docstring |
| `bins.py` | the inline formula at 108 and `data._TOKEN` | uses `data.yearly_file`, `data.available_years`, `data.n_tokens`, `model.start_min_cluster_size` |
| `tagging.py` | the inline label rule | `sentiment.label_of`; header names `06-tag-corpus.R` |
| `__init__.py` | — | module list for one pipeline; one entry point |
| `tests/test_pipeline_units.py` | — | renamed symbols; docstring lists the two-step tests; one new check: `config.resolve_paths` for the three layouts (20 checks) |

### C. Stage script → `code/05-analysis-pipeline.py` (`git mv`)

Header rewritten (stages table unchanged in substance; HOW TO RUN for local, dry run and Colab via `PIPELINE_ROOT`;
S6 = `06-tag-corpus.R` + `07-validate-analysis.R`; approvals of sessions 05–07). `stage_embed` calls
`embed.embed_year(..., force=args.force)`; the `embeddings_for_modelled` wrapper goes; `stage_sentiment` and
`bins.build_period_bins` use `data.available_years()`; `render_report` passes the pipeline columns straight through
and joins `category_name` from `categories.csv`; `--embedding-model` defaults to `C.EMBEDDING_MODEL`; options otherwise
unchanged (`--taxonomy-protocol`, `--assign-batch`, `--max-docs`, `--out-root`, `--force` …).

### D. Notebook → `code/05b-analysis-pipeline-colab.ipynb` (regenerated)

| Cell | Content |
|---|---|
| 0 (md) | purpose; the Drive layout under `MyDrive/fifa-communities/`; two passes; stop point; hand-back |
| 0.1 | `drive.mount`; `DRIVE = '/content/drive/MyDrive/fifa-communities'`; assert `yearly datasets/` holds 17 files (no `gdown` fallback: the folder exists); create `analysis/ results/ embeddings/ hf-cache/ logs/ handback/ comparisons/`; session log `{DRIVE}/logs/colab_<date>.log`; `slog()` |
| 0.2 | clone or pull `alannemi/fifa-reddit-communities` into `/content/fifa-reddit-communities`; `os.chdir`; copy the 17 CSVs to `/content/data/yearly datasets` once per session; `os.environ['PIPELINE_ROOT'] = DRIVE`, `['PIPELINE_YEARLY_DIR'] = '/content/data/yearly datasets'`, `['TOKENIZERS_PARALLELISM'] = 'false'`; print and log the five effective paths (`python -c "from topicmodel import config as C; …"`) |
| 0.3 | `pip install -q --no-deps -r code/requirements-colab.txt`; `pip check`; GPU, cuML, pins → log (unchanged) |
| 0.4 | `run_logged`, `stage()`, `cfg(b)` (`{DRIVE}/results/bin-{b}/config.json`), `unit_checks()`, `taxonomy_summary(b)` (unchanged logic, new paths) |
| 1 | S0 → `{DRIVE}/analysis/period_bins.csv`; Σ assert |
| 2.1–2.4 | Pass 1 on bin 1 exactly as today (embed, topics with `accel=True`, label 12, two-step taxonomy, sentiment 2010, partial collect) |
| 3.1 | full bin-1 labelling, taxonomy `--force`, collect; **stop point of the Colab test batch** (unchanged) |
| 3.2 | optional seeded comparison → `--out-root {DRIVE}/comparisons/cmp-seeded` |
| 3.3–3.5 | large-bin gate, `T`/reassign, bins 2–9 (unchanged) |
| 4 | sentiment all years, complete collect (unchanged) |
| 5 | hand-back zip: `{DRIVE}/results/bin-N/…` → `results/analysis/bin-N/…` (no `model/`), aggregated CSVs → `results/analysis/`, `{DRIVE}/analysis/*` → `data/analysis/`, the log → `logs/`; written to `{DRIVE}/handback/bin-N_<date>.zip`; md cell: unzip at the repository root |

### E. R scripts

* New `code/R/pipeline-utils.R`: `log_msg`, `check` (width 64), `read_chr(p, ...)`, `col_hash`, `find_root(sentinel)`
  (cwd, then the script's own location from `--file=`, then stop with a clear message; no OneDrive fallback),
  `ORIGINAL_COLS` (the 42 names now in `03:97-103`), `ANALYSIS_COLS` (the 8), `read_period_bins_table(dir)` +
  `bin_of()` (from `period_bins_summary.csv`). Each script sources it with a four-line bootstrap that finds its own
  directory (`Rscript code/…` from the root, or `source()` in RStudio).
* `02-data-cleaning.R`: sources the utils for `log_msg`/`find_root` only; logic untouched.
* `03-data-validation.R`: sources the utils; C4a/C4c accept 42 or 50 columns and print the spec found (H7).
* `07-tag-corpus.R` → `06-tag-corpus.R`, `08-validate-analysis.R` → `07-validate-analysis.R` (`git mv`): shared
  helpers; both read `TAG_IN_ROOT` (default `.`) and `TAG_OUT_ROOT` (default `= TAG_IN_ROOT`); `ANALYSIS_DIR`
  default `<TAG_IN_ROOT>/results/analysis`; headers name each other and `05-analysis-pipeline.py`.
* `01-data-inspection.R`: the hard-coded `setwd` goes (comment: run from the repository root / the `.Rproj`).
  `04-render-codebook.R`: `find_root` from the utils is **not** used (standalone, H4); the OneDrive fallback goes and
  the Chrome path becomes `Sys.getenv("CHROME", "/Applications/…")`.

### F. Repository plumbing

`git rm code/05-topic-model.py code/05b-topic-model-colab.ipynb code/topicmodel/label.py data/jsonl_to_csv.py
data/data-inspection.R`; `git mv` the four renames; `mv results/topic-models/embeddings results/embeddings` (local,
gitignored); `.gitignore`: `/results/embeddings/` replaces `/results/topic-models/embeddings/`, the redundant
`/data/analysis/sentiment/` line goes; `requirements-topicmodel.txt` header names the pipeline (pins unchanged).

### G. Documentation (written for co-authors, as before)

* New `docs/session-07_2026-10-25/`: `00_approvals.md` (H1–H7 with alternatives, the defaults above, flags F31–F38),
  `01_code-harmonization-notes.md` (module map after; deleted / moved / renamed table with the setting-name mapping;
  the path root; notebook cells; R utils; what did not change), `02_runbook-colab-run.md` (the session-06 runbook
  carried forward with the new names and Drive layout; session-06's runbook gets a dated pointer to it).
* Live documents updated: `README.md` (§1 tree, §4, §5), `memory/workflows.md` (§6 folded into §7; commands;
  Drive layout; dry-run recipe), `docs/README.md` (session-07 row + the dated rename note), design document §10 and §16
  (dated notes: Drive layout, code layout after the renames), codebook §1 file table and §8.4 (script names; version
  stays 3.0 draft), `memory/2026-10-25_session-07.md`, the assistant's memory status file.

### H. Commits (unit checks pass after each)

1. `Session 07: approvals; remove the stale tracked copies under data/`
2. `Remove the session-02 exploratory tool; one pipeline in code/topicmodel/` (B, tests)
3. `PIPELINE_ROOT path root; stage script renamed to 05-analysis-pipeline.py` (A, C, F, tests)
4. `Colab notebook rooted on MyDrive/fifa-communities (05b)` (D)
5. `R: shared pipeline-utils.R, S6 scripts renamed, 03 accepts the v3.0 spec` (E)
6. `Session 07: docs, memory, README, workflows, codebook wording` (G); push.

## Verification

```bash
.venv/bin/python code/tests/test_pipeline_units.py                      # 20/20 PASS (after commits 2, 3)
.venv/bin/python code/05-analysis-pipeline.py --stage bins              # skips: data/analysis/period_bins.csv exists
# local MiniLM dry run of S1+S2 on bin 1 into a fresh out-root (≈ 2.5 min; cache moved, so embeddings are reused)
.venv/bin/python code/05-analysis-pipeline.py --stage embed  --bin 1 --embedding-model sentence-transformers/all-MiniLM-L6-v2 --out-root results/dry-runs/h7-bin1
.venv/bin/python code/05-analysis-pipeline.py --stage topics --bin 1 --embedding-model sentence-transformers/all-MiniLM-L6-v2 --out-root results/dry-runs/h7-bin1
#   expect 92 topics, 35.9 % raw outliers (session-05 dry-run log §3.1) and report.html rendered from the native columns
PIPELINE_ROOT=/tmp/h7-root .venv/bin/python -c "from topicmodel import config as C; print(C.YEARLY_DIR, C.ANALYSIS_DIR, C.EMBED_CACHE_DIR)"   # flat layout
Rscript code/03-data-validation.R                                       # 28/28, "v2.1 spec (42 columns)" (≈ 5 min)
TAG_IN_ROOT=results/dry-runs/bin1-full/r-mini TAG_OUT_ROOT=results/dry-runs/bin1-full/r-out ANALYSIS_DIR=results/dry-runs/bin1-full/results/analysis Rscript code/07-validate-analysis.R   # ALL CHECKS PASSED on the session-05 mini corpus
python3 -c "import nbformat; nbformat.validate(nbformat.read('code/05b-analysis-pipeline-colab.ipynb', 4))"
grep -rn "fifa-reddit-communities/data\|06-analysis-pipeline\|07-tag-corpus\|08-validate-analysis\|05-topic-model\|label\.py" README.md memory code data/codebook_soccer_posts.md docs/README.md docs/session-07_2026-10-25   # only the dated rename note may match
```
Nothing runs on Colab in this session; the test batch of the session-06 runbook is the next session's first step.

## Flags for your review (continuing the design's list)

| # | Issue | What the plan does |
|---|---|---|
| F31 | Renamed settings change the keys recorded in every future `config.json`; the gitignored dry runs of sessions 05–06 keep the old keys | mapping table in `01_code-harmonization-notes.md`; no committed run exists |
| F32 | The two-step taxonomy has still never run with the real model (local dry run stopped 2026-10-24); this session changes no agent logic, so that risk is unchanged | first run = the Colab test batch, next session |
| F33 | `--out-root` dry runs stay repository-shaped (`<out>/results/analysis`, `<out>/data/analysis`) while the Drive root is flat | the R scripts and the dry-run recipe keep working; the hand-back zip converts flat → repository paths |
| F34 | If `PIPELINE_YEARLY_DIR` is not set on Colab the CSVs are read from Drive (slow, but correct) | cell 0.2 sets it and prints the effective paths |
| F35 | `find_root` without the OneDrive fallback: R scripts must run from the repository root or via `Rscript code/…` | clear stop message; the `.Rproj` sets the root in RStudio |
| F36 | The report loses the intertopic map option | never produced by the pipeline; `model/` is saved for it |
| F37 | `requirements-topicmodel.txt` still uses `>=` for eight packages; left as is (environment rebuilds were tested on 2026-10-13 and 2026-10-23) | header fixed only |
| F38 | Deleting `data/jsonl_to_csv.py` and `data/data-inspection.R` from git leaves the local copies on disk (the shared data folder is yours) | noted in the approvals; you remove them when convenient |
