# Session 08 — 2026-10-25 — First Colab run: Pass 1 and the full bin 1; decision D6 (the assignment shortlist)

**For:** the author and co-authors (Alan Nemirovski decides; Claude ran the analysis of the outputs and the code changes).
**Reads with:** the runbook `../session-07_2026-10-25/02_runbook-colab-run.md`, the design document
`../session-04_2026-10-22/01_analytical-pipeline-design.md` (v1.4: §7.4 step 2, §12.4, §13 F41, §14 D6), the notebook
`code/05b-analysis-pipeline-colab.ipynb`. Every number below comes from the hand-back zip `bins-1_2026-10-25_1611.zip`,
committed as `results/analysis/bin-1/` (`844399e4`), and from the session log `logs/colab_2026-10-25_1524.log`.

## 1. Environment (Colab Pro+, 2026-10-25, 15:24–16:11 UTC)

NVIDIA A100-SXM4-80GB · torch 2.11.0+cu130 · CUDA 13.0 · cuML 26.06.00 · bertopic 0.17.4 · umap-learn 0.5.12 ·
hdbscan 0.8.44 · sentence-transformers 6.1.0 · plotly 5.24.1 · code at the session-07 commits (`3b55d984` → `d1a4c091`).
Drive layout `MyDrive/fifa-communities/` (H2/H3); the model weights on the Colab machine's disk (`PIPELINE_HF_CACHE_DIR`).

## 2. Notebook and code defects found by the first execution (all fixed in place and pushed the same day)

| # | Cell / stage | Symptom | Cause | Fix (commit) |
|---|---|---|---|---|
| 1 | 0.3 | `pip check`: "ipython 7.34.0 requires jedi" | a gap of Colab's own image | cell flags only complaints about the pipeline stack (`bd5cde80`) |
| 2 | 0.3 | `AttributeError: hdbscan has no __version__` | hdbscan 0.8.44 exposes none | versions read from the installed distributions (`79516ac4`) |
| 3 | 2.1 topics | `SafetensorError: header too large` loading mpnet | safetensors memory-maps the weights; Drive's FUSE mount cannot serve that | weights on the local disk, `PIPELINE_HF_CACHE_DIR` (`3b55d984`; flag F39) |
| 4 | 2.1–2.3 | no `report.html` on Drive after a "clean" Pass 1 | plotly 5.24.1: `add_vline` with a datetime position + annotation raises; the writer never fails a stage | markers as shape + annotation; `report_ok()` asserts the report after each rendering stage (`d1a4c091`; F40) |

The notebook is edited in place from now on; the session-07 generator script went stale after fix 1 and is not the source.

## 3. Pass 1 (cells 2.1–2.4) and the full bin 1 (cell 3.1): the gates

| Check | Result | Design gate |
|---|---|---|
| S0 | nine counts = the design table, Σ 1,302,021 | — |
| S1 + S2, bin 1 | cuML backend; 10,002 posts, 9,809 modelled; **65 topics** at `min_cluster_size` 25, guard converged at the first trial; **raw outliers 34.8 %**; 1.8 min | gate 1 ✓ (50–100 topics; dry run 35.9 % with MiniLM) |
| 2010 themes in the report | vuvuzelas (topics 26, 37), Jabulani (40), Paul the octopus (28), refereeing and the Suárez handball (2), the French camp (27), North Korea (34), Spain's win inside topic 0 | ✓ |
| S3a labels, 65 topics | **`needs_review` 0 of 65**; agreement: 61 × A accepts B, 3 × B accepts A, 1 identical; 64 of 65 agreed in round 2; 162 prompts, 0 truncated, **2.0 min** | gate 4 ✓ (≤ 15 %) |
| S3b definitions | agreed round 2 (A accepts B); 37 categories in the raw answer, 2 same-name duplicates merged → **35 defined** | — |
| S3b assignments | 65 items: 57 identical in round 1, 8 agreed in round 2 (A accepts B); 1 topic placed by the repair; 0 unplaced; 4 answers salvaged (N11); 3 truncated answers; **13.4 min** | — |
| Final taxonomy | **27 categories** (8 defined categories empty); 17 categories hold a single topic | — |
| S4 (2010 only) | 13,638 posts scored; validation sample 299 posts, 12 strata | — |
| S5 (partial) | 1,302,021 rows, bin 1 complete, 9,809 with a topic | — |

Timing on the A100 against the estimates of design §10: labelling 2 min for 65 topics (estimate 5–8), taxonomy 13 min
(estimate 3–8). Greedy decoding kept, the seeded comparison dropped (author, 2026-10-25; design §14). The D2 refinement
(name + tone as agreement) is moot: nothing was flagged.

## 4. Finding: the assignment step placed about 18 of 65 topics in the wrong category — and both analysts agreed

The gates measure agreement and completeness, not correctness. Reading `topic_categories.csv` against the category
definitions (`taxonomy_definitions.json`, the list both analysts saw) shows a systematic failure:

| Topic | Put in | A fitting defined category existed |
|---|---|---|
| 35 MLS Week 15 | cat_12 Live Football Coverage & Streaming | cat_33 MLS Week 15 & Match Previews |
| 24 MLS Support | cat_24 Goal Celebrations & Fan Reactions | cat_23 MLS Support & Fan Engagement |
| 30 Sevilla vs Barcelona | cat_33 MLS Week 15 & Match Previews | cat_29 Sevilla vs Barcelona Match Discussion |
| 19 Managerial Changes | cat_17 Squad Analysis & Team Formation | cat_20 Managerial Changes & Controversies |
| 21 Terry & Rooney Controversy | cat_33 MLS Week 15 & Match Previews | cat_20 Managerial Changes & Controversies |
| 43 Rafa Benitez Inter Deal, 57 Mourinho Real Madrid | cat_1 World Cup Predictions & Analysis | cat_20 Managerial Changes & Controversies |
| 39 Fabregas Transfer Saga | cat_1 World Cup Predictions & Analysis | cat_2 Transfer Rumors & Player Signings |
| 11 Live Football Coverage | cat_11 Jersey & Clothing Discussions | cat_12 Live Football Coverage & Streaming |
| 29 Coaching & Development | cat_17 Squad Analysis & Team Formation | cat_28 Coaching & Development in Football |
| 23, 32 World Cup Highlights | cat_1 / cat_16 | cat_21 World Cup Highlights & Player Performances |
| 38 Inter vs Roma, 48 Copa Libertadores, 49 Portsmouth Crisis, 41 Soccer News Aggregator | cat_1 World Cup Predictions & Analysis | none of these is World Cup prediction |
| 59 NSFW Fan Culture → cat_5 Iconic Goals; 60 Sex Scandal → cat_3 Refereeing & Rule Issues | | — |

Pattern: `cat_1` absorbed 13 topics, `cat_16` 9 and `cat_17` 6, while **the 8 categories that stayed empty (cat_20, 21,
23, 26, 28, 29, 31, 35) are exactly the ones the misplaced topics belonged to** — the categories late in a 35-line list.
The ids are consistent throughout (checkpoint = prompt = `categories.csv`), so this is not a bookkeeping error; in most
cases **both analysts wrote the same wrong id in round 1** (57 of 65 items "identical"). The design flagged the mechanism in
F17: two copies of one small model agree with each other, so agreement is no check on correctness. The debate caught
nothing because there was nothing to debate.

Consequence for the full run: the error would recur in every bin, worse in the interim bins (longer category lists); the
override file cannot repair bin 1 by hand because the right categories were dropped as empty.

## 5. Decisions (Alan Nemirovski, 2026-10-25)

| # | Decision | Alternatives declined |
|---|---|---|
| — | **Greedy decoding kept; the seeded-sampling comparison (F17, cell 3.2) dropped** | run the comparison first; switch to seeded |
| — | The D2 refinement (name + tone as agreement) is not needed: 0 % flagged | — |
| **D6** | **Step 2 of the taxonomy lists only each topic's 10 most similar categories, most similar first; the answer carries the category's id and its name (the name wins a conflict); a topic whose agreed category ranks worse than 3rd by similarity is flagged `assignment_needs_review`.** Re-run step 2 alone on bin 1 (`--force --keep-definitions`) and compare | (2) embedding-only assignment with the analysts confirming; (3) a review flag alone; (4) hand repair per bin |

## 6. Implementation of D6 (same day; unit checks 21/21)

* `config.py`: `TAXONOMY_SHORTLIST = 10` (0 = the v1.2 whole list), `TAXONOMY_REVIEW_RANK = 3`.
* `prompts.py` v1.3: the step-2 user text — two intros (shortlist / whole list), `category_name` in the schema, the rule to
  copy id and name from the same line. S3a and step-1 texts byte-identical to v1.2.
* `agents.py`: `_candidate_order()` ranks the categories per topic by the cosine similarity of the sentence embeddings of
  the category and topic lines (the texts shown, without ids); items list the shortlist (union, interleaved by rank, for
  items with several topics); `validate_assignments()` reads the name, resolves an id/name conflict in favour of the name
  (`resolved_from = name_over_id`, one repair turn); `topic_categories.csv` gains `assignment_similarity_rank` and
  `assignment_candidates`; `assignment_needs_review` also set when the rank exceeds the threshold; `taxonomy.json →
  shortlist` records k, the rank distribution and `n_flagged_by_rank`; the review template `topic_category_overrides.csv`
  shows `assignment_needs_review` and `similarity_rank`.
* `05-analysis-pipeline.py`: loads the sentence-transformer for the shortlist (`--embedding-model`, the same model as the
  documents); `--keep-definitions` makes `--force` keep the step-1 checkpoint so that only the assignments re-run.
* Notebook cell 3.1b (the re-run on bin 1); runbook §4; design v1.4 (§7.4, §7.6, §12.4, F41, D6, N13); this document.
* N13 (after the dry run below): `validate_assignments()` no longer spends a repair turn on extra pairs or an id/name conflict.
* Tests: `test_two_step_validators` (id + name cases), `test_taxonomy_shortlist` (new: shortlist of 4 of 12 categories,
  rank flag, name-over-id), `test_taxonomy_two_step` (template columns, whole-list record).

### 6.1 Local re-run of step 2 on the real bin-1 outputs (Mac, MPS, Qwen3-1.7B, mpnet for the shortlist)

Command: `--stage taxonomy --bin 1 --force --keep-definitions --out-root results/dry-runs/d6-bin1` on a copy of the Colab
outputs (the definitions resumed from the Colab checkpoint, 35 categories); Qwen3-1.7B bfloat16 on MPS, batch 4; mpnet for
the shortlist. 15.5 min, 157 prompts, 0 truncated, 4 answers salvaged. Outputs in `results/dry-runs/d6-bin1/` (tracked).

| | Colab run (v1.2, whole list) | D6 dry run (shortlist of 10) |
|---|---|---|
| Topics assigned by the agents / repaired | 64 / 1 | 65 / 0 |
| Assignment debates agreed in round 1 (identical) | 57 | 64 |
| Defined categories left empty | 8 (the ones the misplaced topics belonged to) | 1 |
| Final categories (single-topic ones) | 27 (17) | 34 (18) |
| Largest category | `cat_1`, 13 topics | 4 topics (`cat_21`, `cat_10`, `cat_20`, `cat_29`, `cat_16`) |
| Rank of the agreed category by similarity | — | 1: 44 · 2: 10 · 3: 2 · 4: 4 · 5: 2 · 6: 1 · 7: 1 · 8: 1 |
| Topics flagged `assignment_needs_review` | 0 | **9** (rank worse than 3) |
| Answers carrying `category_name` / id–name conflicts | — | 155 of 157 / 0 |

37 of 65 topics changed category. Of the misplacements listed in §4, 15 are now in the fitting category (35 → cat_33, 24 →
cat_23, 30 → cat_29, 19 → cat_20, 42/57/58 → cat_20, 43/49 → cat_10, 39 → cat_2, 11 → cat_12, 29 → cat_28, 23/32 → cat_21,
33 → cat_31, 27 → cat_26, 22 → cat_22); "World Cup Predictions & Analysis" shrank from 13 topics to 3. Remaining or new
errors, for the co-author review: topic 1 *Transfer Window Rumors* → cat_10 *Manchester City & Financial Issues* (both
analysts; rank 2 although cat_2 *Transfer Rumors* was the first candidate; not flagged); topic 0 *World Cup 2010
Predictions* → cat_21 *World Cup Highlights* (rank 5, flagged); topic 46 *EPL Team Support* → cat_22 *UEFA Champions League
Live* (rank 1, not flagged); topics 38, 45, 48 (match threads of other competitions) → cat_29 *Sevilla vs Barcelona Match
Discussion*, the model's generic match category; topic 64 → cat_26 (rank 7, flagged); topic 18 *World Cup Anthem & Chant* →
cat_18 *Advertising* (rank 8, flagged) — see N13 below. The flag catches most of the doubtful placements (9 flagged, of
which 3 look acceptable on reading) and misses a few (1, 46).

**Refinement N13 (applied the same day).** The transcripts show why topic 18 went wrong: both analysts answered `cat_19 World
Cup Anthems & Chants` correctly in round 1 but added pairs for topic ids that were not in the item; the validator treated the
extras as an error, spent the repair turn, and the repaired answer was `cat_18`. 16 of 65 items spent a repair turn on such
extras (foreign or repeated topic ids) and 5 answers changed category in it. Since N13 the extras are dropped and recorded
(`cleanup` in the parsed answer; `n_answers_cleaned` in `taxonomy.json`) without a repair turn, as is an id/name conflict;
the repair turn remains for a topic left without a category or an unresolvable reference. Replayed on the dry-run
transcripts, N13 changes two final categories (18 → cat_19, right; 47 → cat_1). Unit checks 21/21.

**What D6 does not fix** — the definitions. 35 categories for 65 topics, with near-duplicates (two vuvuzela categories),
categories named after one match or one player (*Sevilla vs Barcelona Match Discussion*, *Landon Donovan & Football Stars*,
*MLS Week 15 & Match Previews*). The designed remedy is the co-author's `category_overrides.csv` (merges); whether step 1
should be asked for fewer, broader categories is a later decision for the author (not taken today).

### 6.2 Colab re-run of step 2 with D6 + N13 (cell 3.1b, 17:22 UTC) — the taxonomy of record for bin 1

Hand-back `bins-1_2026-10-25_1722.zip`, committed as `results/analysis/bin-1/` (the pre-D6 files of the first run remain in
git at `844399e4`). A100, prompt v1.3, shortlist 10 of 35, mpnet for the shortlist: **2.6 min**, 133 prompts, 0 truncated,
4 salvaged, **26 answers cleaned by N13** (extras dropped without a repair turn).

| | First run (v1.2) | Local D6 (Mac) | **Colab D6 + N13** |
|---|---|---|---|
| Topics assigned by the agents / repaired | 64 / 1 | 65 / 0 | **65 / 0** |
| Assignment debates identical in round 1 | 57 | 64 | 62 (3 agreed in round 2) |
| Defined categories left empty | 8 | 1 | **1** (`cat_35`, the duplicate vuvuzela category) |
| Final categories (single-topic) | 27 (17) | 34 (18) | **34 (17)**; largest 5 topics |
| Rank distribution | — | 1: 44 … 8: 1 | **1: 47 · 2: 10 · 3: 2 · 4: 3 · 5: 1 · 6: 1 · 7: 1** |
| Flagged `assignment_needs_review` | 0 | 9 | **6** (topics 0, 43, 45, 56, 57, 64) |
| Same category as the local D6 run | — | — | 60 of 65 |

The five differences from the local run are the N13 effect and all read as improvements: 18 → `cat_19` *Anthems & Chants*
(right), 26 → `cat_25` (the two vuvuzela topics now share one category), 47 and 52 → `cat_1`, 40 *Jabulani* → `cat_18`
*Advertising & Branding* (adidas; debatable). For the co-author review: topic 1 *Transfer Window Rumors* → `cat_10`
*Manchester City & Financial Issues* (rank 2, not flagged; `cat_2` *Transfer Rumors* was the first candidate), topic 46 *EPL
Team Support* → `cat_22` *UEFA Champions League Live* (rank 1, not flagged), and among the flagged six, topic 0 → `cat_21`
and topic 64 → `cat_26` look wrong, 43, 45, 56, 57 defensible. `n_with_category` in the partial collect: 6,395 posts (all
65 topics carry a category; the rest of bin 1 are the 3,414 raw outliers).

**Verdict for the full run.** The pipeline has run end to end on bin 1 and every gate that can be read before the review is
passed; the assignment step is now sound enough to run on the other bins, with the review flag doing its job. Open before
bins 2–9, in the runbook's order: the review of bin 1 (§7), the large-bin gate, `T`. One granularity question is for the
author: step 1 defines many fine categories (35 for 65 topics, half of them single-topic). The brief's rule ("no minimum or
maximum number of categories") is kept as designed; the merges are the co-authors' category review, as in the template
paper's manual merge. Asking step 1 for broader categories would be a prompt change (v1.4) — not taken.

## 7. Next steps

1. ~~Colab cell 3.1b and the hand-back~~ — done 17:22 UTC (§6.2).
2. Review of bin 1 by the co-authors (runbook §4 and §6): `topic_overrides.csv` (merges: 26 + 37 vuvuzela, 42 + 57 Mourinho,
   24 + 35 MLS, 11 + 22 + 53 live-stream spam; renames of 0 and 3) → `--stage taxonomy --bin 1 --force` → `category_overrides.csv`
   (the streaming, MLS and match-thread categories; player-named singletons) and `topic_category_overrides.csv` (start with the
   six flagged topics, then 1 and 46) → `collect`. The outlier calibration samples of bin 1 (`belongs`) in the same session.
3. Large-bin gate on bin 4 (cell 3.3), `T` from the samples of bins 1 and 4 (cell 3.4), bins 2–9 (cell 3.5 with its two review
   pauses), sentiment + the 300-post hand-coded sample, collect, hand-back of all bins, S6 locally, codebook v3.0 final.

## 8. Large-bin gate (bin 4, interim Oct 2014 – Jan 2018, 331,216 posts) — first attempt failed on the outlier share

Cell 3.3, 2026-10-25 (after the D6 re-run). cuML backend; embeddings for 2014–2018 cached; UMAP + the guard 6.5 min.

| Guard trial | `min_cluster_size` | Topics | Raw outliers |
|---|---|---|---|
| 1 (0.25 % of the modelled posts) | 820 | 2 | 1.2 % |
| 2 (÷ 1.5) | 547 | **74** (in range → converged) | **56.2 %** |

The guard converged, so the topic count is fine; the gate failed on the second criterion (raw outliers ≤ 45 %, design §12.2),
and the cell stopped before labelling, as designed. Reading: with `min_samples` 10 and EOM selection, the condensed tree of
this three-year bin has two giant clusters near the root (trial 1) and, one step down, 74 clusters that cover only 44 % of the
posts. Bin 1 had 35 % raw outliers at the floor size of 25.

**What was added for the decision (same day, unit checks 21/21):** `--stage grid --bin N` (notebook cell 3.3b) refits HDBSCAN
on the cached UMAP array over `config.HDBSCAN_GRID` — `min_cluster_size` × {1, 0.67, 0.5} of the guard's 547, `min_samples`
{10, 5, 3}, selection {eom, leaf} — and writes `bin-4/hdbscan_grid.csv` (topics, outlier share, largest cluster, median size
per setting). The clustering settings (`HDBSCAN_MIN_SAMPLES`, `HDBSCAN_SELECTION_METHOD`, `HDBSCAN_EPSILON`,
`TOPIC_COUNT_RANGE`) may now be one value for every bin or a dict by period type; the values in force are recorded per bin
(`config.json → topics.hdbscan`). `--stage topics --force --keep-umap` (cell 3.3c) applies a decision without recomputing UMAP.
Verified locally on the bin-1 dry run: the default setting reproduces 92 topics / 35.9 % exactly; `min_samples` 5 gives 98 topics
/ 32.5 %.

**Decision pending (author):** the setting for the interim bins — and whether it applies to every bin (then bin 1 is
re-clustered and relabelled, minutes, before its review) or by period type (bin 1 stays). Options, to be read off the grid:
lower `min_samples` (fewer points declared noise at the same cluster count); `leaf` selection; a wider topic range for the
interim bins (e.g. 50–150, smaller clusters, fewer outliers — a change to design §6.2); `epsilon` > 0 (merges micro-clusters);
or accepting > 45 % raw outliers for the interim bins and letting the D3 reassignment carry more weight.

