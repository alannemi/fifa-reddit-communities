# Session 11 (2026-10-30): the outlier calibration assessed; decisions K1–K7

*Participants: Alan Nemirovski (decisions), Claude (reading, analysis, plan, documents). Co-authors BA and AM, and Alan
himself (AN), coded the judgement samples. The assessment was made in plan mode; Alan answered two rounds of questions and
approved the plan on 2026-10-30 before any file in the repository was changed.*

## 1. The request

Alan asked for four things:

1. Pull the GitHub repository and merge it into the local copy.
2. Read all session memory and the design documents.
3. In plan mode, assess the outlier calibration documents: what the results mean for the outlier threshold, and how it
   might be adjusted.
4. Ask instead of assuming.

After the assessment he asked that the decision be recorded as option B (§3, K1), that any associated decisions be put to
him, and that the record be committed and pushed.

## 2. The pull

A fast-forward from `d8dcde22` to `101523b0` (six commits of co-author uploads, 18 files):

- AN's coded outlier bands of bins 2 and 4 (`sample_T*-coded-AN.csv`);
- AM's coded bands of bins 6 and 8 (`sample_T*-coded-AM.csv`);
- BA's review files `manual review/BIN2_topics_labelled_BA.csv` and `BIN4_topics_labelled_BA.csv` (the first QUALTAG
  files for interim bins; not processed in this session).

With BA's bands of bins 1, 3, 5 and 7 and AM's of bin 9 (sessions 08–10), every band of every bin is now coded:
9 bins × 4 bands × 100 posts = 3,600 judgements. The assessment is in `01_outlier-calibration-results.md`.

## 3. Decisions (Alan Nemirovski, 2026-10-30)

| # | Question | Decision | Why / note |
|---|---|---|---|
| **K1** | What should change in the outlier tiers, given that 43.6 % of dual-tier posts and 53.4 % of singly reassigned posts were judged to belong, and that precision barely rises with similarity | **Option B: keep the per-post tiers of D9 (`topic_assignment`, `topic_uid`, `topic_uid_2`, both similarities) exactly as they are; every topic-level aggregate (`n_total`, `share`, `mean_vader_sentiment`, `n_posts_scored`, topics over time, the QA shares) counts cluster + reassigned posts only.** Dual memberships are kept as separate counts | Aligns the topic table with the population the figures already use (E3, Q7). Reverses D9's counting rule ("a dual post counts once in each of its two topics"), under which 38.6 % of the memberships in `n_total` came from dual posts. Declined: keeping D9 unchanged with the precision reported; dropping the dual tier (outliers 15.7 % → 35.8 % of modelled posts); raising the floor to the 50th or 65th percentile (dual-band precision 40 % → 43 % or 48 % for 7–13 points of coverage) |
| **K2** | The coder effect is larger than the similarity effect (BA 34 %, AN 52 %, AM 59 % of posts judged to belong; coder and bin confounded). Cross-code a subsample now? | **No cross-coding now.** The codebook reports the precision as the **plain pooled average over all judgements** (each judgement counting once: dual 43.6 %, reassigned 53.4 %, all admitted 46.1 %), with the per-coder and per-bin tables beside it, states that the samples are single-coded and that the coders differ, and **flags inter-coder checks as a later step for the chapter's editing process** | Alan's answer: "use weighted average of coder precision; flag inter-coder checks as a later step that will be done during editing process". Asked which weights: **by number of judgements**. Declined: weights by the corpus posts each bin-and-band holds (dual 50.2 %, reassigned 60.9 %) and equal weight per coder (45.7 %, 55.2 %); both are reported alongside for transparency |
| **K3** | A cluster-member baseline (how often a HDBSCAN member would be judged to belong to its topic's name) | **No** | The tiers' precision is reported without a reference |
| **K4** | The large-bin gate's raw-outlier tolerance, raised during the run from the design's 45 % to 50 % for converged bins (assistant's proposal after bin 3; pending since 2026-10-25) | **0.50 for a bin whose guard converged is confirmed** as the decision of record | Bins 2 and 4 ran at 48.2 % and 48.4 %; the design's 45 % stays as the original reference |
| **K5** | S6 in place (the one-shot append of the 15 analysis columns to the master and the 17 yearly files), now unblocked | **Later, with its own go-ahead** | The data files stay v2.1 after this session; the collect output is what S6 reads |
| **K6** | `mean_vader_sentiment` under K1 | **One definition: the mean title score over the topic's cluster + reassigned posts**; `n_posts_scored` the same population | Declined: a second, dual-inclusive column |
| **K7** | The per-bin review-time files that show the D9 counts (`topics_over_time.csv`, `report.html`, `review_pack.md`) | **Leave them; note it in the docs.** The recount rewrites only `bin-N/topics.csv` | `topics_over_time.csv` needs the BERTopic model, which lives on Drive (Colab); the review packs are the record of what the co-authors reviewed |

## 4. Defaults stated with the plan (not objected to)

- The tiers' quantiles (`OUTLIER_REASSIGN_QUANTILE` 0.35, `OUTLIER_SINGLE_QUANTILE` 0.80) and the four calibration bands stay
  as they are, so the coded files keep their meaning.
- The 15 per-post columns of `post_variables.csv` do not change; `labels_version` does not change.
- The size columns are renamed rather than redefined: `n_first` and `n_as_second` go; `n_reassigned`, `n_dual_first` and
  `n_dual_second` come; `n_total = n_cluster + n_reassigned`; `share = n_total / n_modelled` (sums to at most 1 per bin).
- `topic_crosswalk.csv`, still a blank template, is regenerated with the new `n_total`.
- The calibration summary is an R script (`code/15-outlier-calibration.R`), the team's convention; the Python analysis
  made during the assessment is its independent check.
- `tagging.annotate_calibration_samples` is fixed to skip every coded sample file (today `-coded-AM` / `-coded-AN` names
  are not protected from a review-pack refresh, as noted in session 10).

## 5. What was run and changed

- **Assessment (read-only, plan mode):** the 36 coded files parsed and checked (100 rows each, ids identical to the blank
  samples, `belongs` ∈ {0, 1}, no notes); precision by bin, band, tier, coder and period type; Wilson intervals; logistic
  fits and AUC of similarity, within-bin rank, margin to the second topic and coder; attractor topics; the effect of every
  candidate quantile pair on the corpus-wide tiers (`01_outlier-calibration-results.md`).
- **Record of the decisions** (this file, the results document, design v1.9 §12.2 / §13 F46 / §14, `docs/README.md`,
  `memory/2026-10-30_session-11.md`), committed and pushed at Alan's request before the implementation (`40d7a0eb`).
- **The K1 counting in the pipeline.** `config.TOPIC_COUNT_ASSIGNMENTS = ("cluster", "reassigned")`; `model.topic_counts()`
  (model-free, shared by `topics.csv` and the collect), `model.topic_qa()`, `topic_table()` and `topics_over_time()` rebuilt
  on it; the collect's sizes and the topic-level VADER mean (K6) over the counted posts; `n_first` / `n_as_second` replaced by
  `n_reassigned`, `n_dual_first`, `n_dual_second`; the sidecar records `topic_count_assignments`; new `--stage recount`
  (rewrites the count columns of `bin-N/topics.csv` without the BERTopic model, `config.json -> recount`);
  `annotate_calibration_samples` skips every `*coded*` file. **Unit checks 14/14** (the topic-table and collect checks
  rewritten for K1; a `-coded-XX` file survives a refresh byte for byte).
- **Recount of the nine bins and the local collect** (seconds and 12 s). Verified: `post_variables.csv` byte-identical before
  and after (same md5), `labels_version` unchanged (`f2bc460bf46d`), `complete = true`, no errors; in
  `topic_labels_all_bins.csv` **Σ n_total = 827,744 = 712,218 cluster + 115,526 reassigned** (the figures' population),
  Σ n_dual_first = Σ n_dual_second = 260,054, `n_total = n_cluster + n_reassigned` for every topic, the shares of a bin sum to
  0.61–0.72; the per-bin `topics.csv` agree with the topic table cell for cell; no column changed besides the sizes and
  `mean_vader_sentiment` / `n_posts_scored` (6,179 of 6,245 topic means moved, mean absolute change 0.028, largest 0.282);
  `topic_crosswalk.csv` regenerated (still blank). Per bin, Σ n_total before → after: 1: 10,150 → 7,076 · 2: 191,665 →
  112,405 · 3: 85,375 → 51,723 · 4: 343,821 → 200,735 · 5: 91,664 → 57,468 · 6: 317,601 → 195,745 · 7: 42,872 → 29,606 ·
  8: 225,643 → 147,485 · 9: 39,061 → 25,501.
- **`code/15-outlier-calibration.R`** (new): every check PASS; the tables `tab_F1`–`tab_F4b` and the figures F1–F2 in
  `results/descriptives/`; its pooled figures equal the Python assessment to the decimal (dual 43.6 %, reassigned 53.4 %,
  all admitted 46.1 %; AM 58.7 %, AN 51.7 %, BA 33.8 %; corpus-weighted 50.2 / 60.9 / 53.5 %; equal-coder 45.7 / 55.2 / 48.1 %).
- **Regression of the figures:** `11`, `12` and `14` re-run; **every tracked PNG byte-identical** (the populations are the
  posts' tiers, which K1 leaves as they are); the PDFs, changed by their timestamp only, restored from git.
- **Scratch-root S6 chain:** not re-run — its input `post_variables.csv` is byte-identical to the one the session-09 chain test
  passed on, and `06`/`07` do not read the topic table.
- **Unplanned repair (`code/14-worldcup-figures.R`):** today's pull added BA's `BIN2_topics_labelled_BA.csv` and
  `BIN4_topics_labelled_BA.csv` to `manual review/`. They carry **no `QUALTAG` column**: BA filled the eleven `qt_` indicator
  columns directly (916 and 1,602 topics). Script 14 read every `BIN*` file and failed its check "the QUALTAG files cover the
  five World Cup bins" (3,903 topics); its file glob is now restricted to bins 1, 3, 5, 7 and 9, which is what the check and
  the figures intend, and it passes again (1,385 topics). **Nothing was decided about the interim bins' files:** how they are
  processed (script 13 expects a `QUALTAG` column and would stop on them; whether the interim bins get political variables at
  all) is an open question for Alan (§6).
- **Documentation:** design v1.9 (§6.3 counting paragraph and the samples' result, §6.4 `topics.csv` and `topics_over_time.csv`
  rows, §7.7 label-table columns, §12.2, §13 F46, §14 K1–K7); codebook v3.0 draft (§4.9 topic-table paragraph and the
  `topic_assignment` row, limitation 10 with the measured precision, §8.1, §9 row; PDF re-rendered); `memory/workflows.md` §7
  (recount) and §10; `results/descriptives/README.md` (F rows, reading note); `manual review/README.md` (the coded samples);
  session-08 run log §15 (dated pointer); `memory/2026-10-30_session-11.md`.

## 6. Open after this session

1. **S6 in place** (`06-tag-corpus.R`, `07-validate-analysis.R`, `03-data-validation.R`) and **codebook v3.0 final** — with
   Alan's go-ahead (K5). The collect output of today is what S6 reads.
2. **BA's interim-bin review files** (bins 2 and 4; `qt_` columns filled directly, no `QUALTAG`): whether the interim bins get
   the political variables, and how the files are checked (script 13 cannot process them as they are). Also still open: TP's
   41 topics in `QUALTAG_to_settle.csv`; the `qt_` variables onto the posts.
3. **Inter-coder check** on the outlier judgement samples at the chapter's editing stage (K2).
4. The per-bin `topics_over_time.csv`, `report.html` and `review_pack.md` keep the D9 counts until a Colab pass regenerates
   them (K7).
