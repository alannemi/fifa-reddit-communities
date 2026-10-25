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

## 7. Next steps

1. Colab: pull, run cell 3.1b (step 2 alone, minutes), then the hand-back cell; commit `results/analysis/bin-1/`.
2. Read the new `topic_categories.csv` against §4 above; the co-author review starts with the flagged topics.
3. Then the runbook from §4 on: review of bin 1 (topic merges — the two vuvuzela topics, the two Mourinho topics, the two
   MLS topics, the three live-stream spam topics), the large-bin gate (cell 3.3), `T`, bins 2–9, sentiment, S6.
