# Session 13 (2026-10-31): dual posts weighted 0.5 in each of their two topics — decision M1

*Participants: Alan Nemirovski (reports the co-authors' joint decision; the record is his), Claude (documents). The decision was
taken jointly by the co-authors outside the session; Alan asked on 2026-10-31 that the documentation reflect it. No code, no
table and no data file changed in this session; the implementation follows with its own go-ahead (§4).*

## 1. The request

Alan's words: "update the documentation to reflect a new decision (made jointly by co-authors) that dual-tagged posts will be
included in aggregate measures, but the weight of the post is going to be split across the two topics (weighed 0.5 in each of
the two topics)."

Earlier in the same session (`../../memory/2026-10-31_session-13.md`): the engagement table of the tagged topics rebuilt on the
Desktop from the co-authors' review files (outside the repository), and the question whether `num_comments` is consistent over
time (it is not: codebook limitation 11).

## 2. Decision (the co-authors jointly; reported by Alan Nemirovski, 2026-10-31)

| # | Question | Decision | Why / note |
|---|---|---|---|
| **M1** | How dual posts (`topic_assignment = dual`: an outlier attached to its nearest topic with the second-nearest topic as a second label, decision D9) enter the topic-level aggregates. Under K1 (2026-10-30) they kept both labels but counted in neither topic | **Dual posts are included in every topic-level aggregate at half weight: 0.5 in the topic of `topic_uid` and 0.5 in the topic of `topic_uid_2`.** Cluster and reassigned posts keep the weight 1 in their topic; outliers and not-modelled posts count in none. The per-post tiers of D9 (`topic_assignment`, `topic_uid`, `topic_uid_2`, both similarities, quantiles 0.35 / 0.80) do not change | The co-authors' joint decision; no written rationale came with it (to add to this row). Replaces the counting half of K1 and decision K6; K1's first half (the tiers stand), K2–K5 and K7 are unchanged. Under K1 the 260,054 dual posts (20.2 % of the 1,289,838 modelled posts) were left out of every topic aggregate; under M1 every admitted post has a total weight of 1 |

## 3. What M1 means for each aggregate (as read by Claude; Alan to confirm — nothing implemented yet)

Weights: `w = 1` for a cluster or reassigned post in its topic; `w = 0.5` for a dual post in each of its two topics; `0` otherwise.

- **Sizes.** `n_total = n_cluster + n_reassigned + 0.5 × (n_dual_first + n_dual_second)` (a half-integer is possible);
  `share = n_total / n_modelled`. The shares of a bin sum to 1 minus the outlier share (about 0.83–0.88; under K1 they summed to
  0.61–0.72): every admitted post has a total weight of 1. `n_cluster`, `n_reassigned`, `n_dual_first` and `n_dual_second` stay
  plain counts.
- **`mean_vader_sentiment`** = the weighted mean of the title compound score over the topic's cluster, reassigned and dual posts
  (weights 1 / 1 / 0.5); **`n_posts_scored`** = the sum of the weights of the scored posts (equal to `n_total` when every counted
  post has a score). *To confirm: a plain count of the contributing posts instead.*
- **Topics over time** (`topics_over_time.csv`, the prevalence series): weighted counts per week / month.
- **The QA shares** of `topics.csv` (`share_match_thread`, `share_lang_en`, `mean_comments`): weighted.
- **The reference run (bin 0):** the same weights in its `topics.csv` and `topics_over_time.csv`.
- **The mapping (`code/16-reference-map.R`, W2):** a post enters a (period topic, reference topic) pair with the product of its two
  weights (1, 0.5 or 0.25); `n_both`, the shares and the QUALTAG composition follow; `unmapped` then means outlier or not modelled
  on the period side only. *To confirm.*
- **Not covered by the request, to decide:** the chapter's figures whose populations are the posts' tiers (E1–E7 and E4: cluster +
  reassigned posts, decisions E3 / Q7) — whether a dual post contributes 0.5 to each of its two topics' categories there too; the
  Desktop engagement table of the tagged topics (2026-10-27 / 2026-10-31), built over cluster + reassigned posts.

## 4. Implementation (not done; with Alan's go-ahead, as K5 was)

- `code/topicmodel/config.py`: `TOPIC_COUNT_ASSIGNMENTS = ("cluster", "reassigned")` becomes a weight per tier
  (`{"cluster": 1, "reassigned": 1, "dual": 0.5}`); the sidecar `post_variables.json → topic_count_assignments` records the weights.
- `code/topicmodel/model.py`: `topic_counts()`, `topic_qa()`, `topic_table()`, `topics_over_time_counts()` on the weights (a dual post
  contributes to two topics); `code/topicmodel/tagging.py` (the collect: the sizes and the K6 / M1 mean); `code/05-analysis-pipeline.py`
  (`--stage recount`); `code/tests/test_pipeline_units.py` (the topic-table and collect checks, rewritten for K1 on 2026-10-30).
- `code/16-reference-map.R` (the pair weights); `code/14-worldcup-figures.R` only if §3's last point says so. `code/15-outlier-calibration.R`
  reads the per-post judgements and is unaffected.
- Then `--stage recount --bins 1-9` and `--stage collect` (seconds; `post_variables.csv` and `labels_version` do not change, as on
  2026-10-30), bin 0 after the hand-back, `Rscript code/16-reference-map.R`. The per-bin `topics_over_time.csv`, `report.html` and
  `review_pack.md` keep the D9 counts until a Colab pass (K7).
- The codebook's §4.9 topic-table paragraph, limitation 10 and §8.1 then lose their "once the topic tables have been recounted"
  clauses; the PDF is re-rendered.

## 5. What was changed in this session (documentation only)

Design v2.1 (the version note, §6.3 counting, §6.4 `topics.csv` / `topics_over_time.csv`, §6.5, §7.7, §14 M1); codebook v3.0 draft
(§4.9 `topic_assignment` row and the topic-table paragraph, limitation 10, §8.1, §9; PDF re-rendered); `results/descriptives/README.md`
(reading note); `results/analysis/reference/README.md`; `memory/workflows.md` §7 and §10; dated pointers in
`../session-11_2026-10-30/00_approvals.md`, `../../memory/2026-10-30_session-11.md` and the session-08 run log §15; `docs/README.md`;
`../../memory/2026-10-31_session-13.md`. **Every table in the repository still carries the K1 counts** (`topic_labels_all_bins.csv`,
`bin-N/topics.csv`, `topic_crosswalk.csv`, `post_variables.json → topic_count_assignments`) until §4 runs.

## 6. Open after this session

1. The points of §3 marked *to confirm*, and the rationale of M1 for the record.
2. The implementation and the recount (§4), with Alan's go-ahead.
3. From sessions 11–12: the Colab reference run and the hand-back, `Rscript code/16-reference-map.R`, the QUALTAG transfer rule (W2);
   S6 in place and codebook v3.0 final (K5); the two bin-5 review files (TP's FINAL file of 2026-10-30 has no `qt_*` columns; scripts
   13, 14 and 16 stop on two files for one bin); BA's bin-2/4 format; TP's remaining to-settle topics; the inter-coder check (K2).
