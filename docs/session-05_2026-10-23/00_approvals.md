# Session 05 (2026-10-23) — Approvals before implementation

**Decisions: Alan Nemirovski. Planning and review: Claude.** The author asked (2026-10-23) that every previously
flagged design change be reviewed and approved before any pipeline code is written. This document records what was
put to him, what he decided, and the implementation-level refinements accepted with the session plan. It complements
the design document `../session-04_2026-10-22/01_analytical-pipeline-design.md` (§13 flags, §14 decisions log).

## 1. Flags from session 04 that were still open — decided 2026-10-23

| Flag | Question put to the author | Decision | Alternatives offered and declined |
|---|---|---|---|
| F4 | Cross-bin comparison: topic and category ids are bin-local. Keep the manual `category_crosswalk.csv` (co-authors map each bin's categories to chapter themes) with the automated alignment S3c only as a hook? | **Approved as designed.** Manual crosswalk; S3c remains a documented hook; stage S5 writes an empty `results/analysis/category_crosswalk.csv` template for the co-authors to fill | build S3c now as an advisory stage; defer until after bin 1 |
| F6 | Period bins: the nine bins replace the ±2-month windows; tournament bins begin ≈ 4.3 months before kick-off and end ≈ 2.5 months after the final (bin 9 ends 12 days after the 2026 final, at the corpus end). Confirm the boundaries? | **Approved as designed.** Boundaries exactly as in the brief and design §4; `period_type` stays the only convenience column | add an `in_tournament` flag column (kick-off → final); revise the dates |
| F18 | Sentiment: `post_sentiment` = VADER on the title only; title + body kept as `post_sentiment_full`. Confirm? | **Approved as designed.** | swap (title + body primary); title only without the second column |
| — | Old test data: what to remove from the session-02/03 2010 test? | **Run folders and cache removed:** `results/topic-models/window-wc2010/` and `year-2010/` deleted from git (recoverable from history, commit `6eeb0fee` and earlier), the local MiniLM embedding cache deleted; `code/05-topic-model.py` and `05b-topic-model-colab.ipynb` kept as exploratory tools; the session-02/03 documents kept as the historical record with a dated note | also archive the old code; remove the run folders only |

All other flags (F1–F3, F5, F7–F17, F19–F21) were decided in session 04 (D1–D4 and the defaults accepted on
2026-10-22) and were not reopened. The three values that the design leaves open — the outlier threshold `T`, greedy
versus seeded sampling, `min_samples` for the large bins — are set at the quality gates (design §12) in session 06.

## 2. Implementation-level refinements accepted with the session plan (N1–N10)

These change *how* a stage is built, not any decision D1–D4 and not any of the eight codebook variables (§4.9).

| # | Refinement | Why |
|---|---|---|
| N1 | Outlier reassignment is its own cheap, replayable step (`--stage reassign`). Stage S2 always saves, for every raw outlier, its nearest topic and cosine similarity (`outlier_nearest.csv`), the calibration table (similarity quantiles) and three 100-post judgement samples (`outlier_calibration/`). Until the threshold is set, `topic_id = topic_id_raw` and `topic_assignment ∈ {cluster, outlier, not_modelled}`; `--stage reassign --bin N` applies `T` in seconds without refitting and rewrites `documents.csv`, `topics.csv`, `topics_over_time.csv` and `report.html` | D3 calibrates `T` only after bin 1 and one large bin exist; folding reassignment into S2 would force a refit per candidate |
| N2 | Frozen HDBSCAN labels: the topic-count guard runs HDBSCAN directly on the saved 5-d UMAP array; the chosen trial's labels are injected into BERTopic through a `FrozenCluster` (no second fit; `prediction_data` unnecessary because `calculate_probabilities=False`) | removes one source of GPU non-determinism within a run |
| N3 | S2 also saves `topic_embeddings.npy` next to `model/` | S3b's deterministic repair and the chunking fallback need them without loading BERTopic |
| N4 | The seeded-sampling comparison mode (F17 test) runs unbatched so that the per-call SHA-256 seed is exact; greedy mode is batched across topics | Hugging Face `generate` has no per-sequence seed |
| N5 | Requirements split: `requirements-topicmodel.txt` (local, pinned) and `requirements-colab.txt` (installed with `--no-deps` on top of Colab's RAPIDS stack, then `pip check`) | design §10: do not disturb the pre-installed RAPIDS stack |
| N6 | S5 may run partially: `post_variables.csv` always holds all 1,302,021 ids with empty topic/category columns for bins not yet modelled; the sidecar lists `bins_complete`; `07-tag-corpus.R` refuses to write back unless all nine bins are complete, except with `TAG_ALLOW_PARTIAL=1` **and** a `TAG_OUT_ROOT` outside the repository (dry runs only) | lets the bin-1 test exercise S5/S6 without touching the real files |
| N7 | Shared `config.py` defaults move to the design values: `LLM_MODEL` → Qwen3-1.7B (was 4B), `LLM_TEMPERATURE` → 0 (was 0.2), `EXTRA_STOPWORDS` + `vs`, `soccer`; a separate `BIN_MIN_CLUSTER_FLOOR = 25` keeps `05-topic-model.py`'s floor of 15 | one settings file (design §0); the old runs are deleted |
| N8 | S2 takes a bin's documents by joining `period_bins.csv` on `id`, not by recomputing dates | S0 is the single source of truth for `period_bin`; S6 checks consistency against it |
| N9 | `--stage sentiment` also writes `data/analysis/sentiment_validation_sample.csv` (300 posts stratified by bin × label × `is_self`, blank coder columns) | the validation sample of design §8 becomes a produced file |
| N10 | Dry-run artefacts go to a gitignored root (`--out-root results/dry-runs/<name>`, which redirects both `results/analysis` and `data/analysis`; only the embedding cache is shared) | the real bin-1 run must start clean |
| **N11 (added during the dry run, 2026-10-23; confirmed by the author on 2026-10-24, `../session-06_2026-10-24/00_approvals.md`)** | **Tolerant JSON recovery** (`config.LLM_JSON_SALVAGE = True`): when an answer fails the strict JSON parse, each category object is recovered field by field (flat arrays only) and then validated exactly as a parsed answer would be; the transcript records `parsed_via = "salvage"` and the parser error. Strict parsing is always tried first; truncated answers are never salvaged. `False` restores design §7.1 literally (second failure = no answer) | In the bin-1 dry run Qwen3-1.7B wrote every S3b category inside its own list (`}], [{` between categories, `]]]` endings) in all four attempts (both analysts, both repair turns); the content was a valid 4-category partition. Without recovery the taxonomy stage produced no categories and all topics were `unplaced`; with it the same answers validate. The alternative — changing prompt v1.1 — would alter an approved text; the switch keeps the decision reversible |
| **N12 (added after the 92-topic run, 2026-10-23; confirmed by the author on 2026-10-24, `../session-06_2026-10-24/00_approvals.md`)** | **Over-long inclusion-cue lists are kept, not rejected:** the S3a validator accepts any non-empty list, keeps the first ten cues and records `n_cues_given` in the answer when more were given (the prompt still asks for 3–6) | In the 92-topic run 10 of the 12 `no_answer` topics failed only because the 1.7B model copied 11–21 keywords as cues and the validator's own 1–10 cap (an implementation choice, not a design rule) rejected the whole label, repair turn included; the names, purposes and descriptions were sound |

## 3. Findings of the independent review of the plan (R1–R13), all accepted

| # | Finding | What the implementation does |
|---|---|---|
| R1 | BERTopic re-numbers cluster labels by size (`topic_mapper_`), so the HDBSCAN labels are not BERTopic's `topic_id` | labels are pre-renumbered by cluster size before freezing; `topic_mapper_.get_mappings()` is saved as `hdbscan_to_topic_map.json` next to `hdbscan_labels_raw.npy`; **`topic_id_raw` is defined as the BERTopic id before outlier reassignment** (codebook §4.9 wording adjusted) |
| R2 | 4,000 new tokens can truncate a 30–50-category taxonomy answer | **`LLM_MAX_NEW_TOKENS_TAXONOMY = 8000`** (design §7.1 said 4,000); truncation detected and recorded in the transcript |
| R3 | BERTopic stores 3 representative documents per topic; the design wants 5 | 5 are extracted after the fit (`_extract_representative_docs`, private API, version pinned), fallback = the 5 members nearest the topic embedding |
| R4 | Sizes, prevalence over time and the report depend on `T` | `topics_labelled.csv` carries no size columns; S5 joins sizes from `topics.csv`; `reassign` rewrites the size-dependent files; runbook order S2 → calibrate → reassign → S3a |
| R5 | `merge_into` semantics were unspecified | topic overrides re-map post `topic_id` (`topic_id_raw` kept), remove merged-away topics from the S3b items and sum sizes; `category_id_agent` = category before category overrides |
| R6 | The debate loop would only be exercised live | a `StubLLM` with scripted answers tests every `agreement_mode`, the repair and no-answer paths, resume and the chunking fallback |
| R7 | "No answer" after the repair turn was undefined | failed revision turn → the analyst keeps its previous answer (identical `revise`, stalemate rule applies); failed round-1 answer → the peer's answer is provisional, `agreed = False`, `needs_review = True`, **`agreement_mode = "no_answer"`** (new value) |
| R8 | A plain `pip install -r` on Colab can move numpy/scikit-learn/pandas against the RAPIDS pins | `--no-deps` installs + `pip check`; backend detected from the fitted objects' module and `cuml.accel.enabled()`; cuML fallback logging on; first item of the session-06 runbook |
| R9 | `--out-root` must also redirect `data/analysis` | done (N10) |
| R10 | S0 would load all `selftext` at once | per-year read with `usecols`; text dropped before concatenating |
| R11 | Qwen3's generation config defaults to sampling | greedy calls pass `do_sample=False, temperature=None, top_p=None, top_k=None`; left padding; explicit pad token |
| R12 | R memory on a 16 GB machine | master first, yearly files one at a time; per-column xxhash64 digests of bare character vectors before and after |
| R13 | Scope | no vLLM; `--stage report` limited to the existing report writer; seeded 5-topic run is a smoke test; empty crosswalk template at S5 |

## 4. How the approvals were obtained

Plan mode, 2026-10-23: one round of four targeted questions (F4, F6, F18, deletion scope) with the recommended
option first; the refinements N1–N10 were listed as defaults in the plan; the plan was then reviewed by an
independent reviewer (13 findings, all folded in) and approved by the author as a whole. Plan file:
`~/.claude/plans/mossy-growing-lark.md` (local to the author's machine; its content is reproduced in
`01_implementation-notes.md`).
