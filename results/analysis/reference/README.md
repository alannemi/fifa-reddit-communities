# results/analysis/reference/

The whole-corpus **reference run** of the analytical pipeline (session 12, 2026-10-30; decisions W1–W4 in
`docs/session-12_2026-10-30/00_approvals.md`; design §6.5): the same BERTopic + two-analyst pipeline fitted once on all
1,302,021 posts — bin 0 inside the code (`python code/05-analysis-pipeline.py --stage <stage> --bin 0`; on Drive
`MyDrive/fifa-communities/results/reference/`) — as the corpus-wide reference list onto which the period topics are mapped
(`code/16-reference-map.R`) and, later, the co-authors' QUALTAG attributes. It is not a tenth bin: `collect` never reads it
and `post_variables.csv` is untouched (W3). The folder is empty until the Colab run (notebook section 6) and the hand-back.

| File | Content | In git? |
|---|---|---|
| `documents.csv` | one row per post of the corpus: `id, created_datetime, period_bin` (the post's **true** period bin, 1–9), `post_type, lang, num_comments, topic_id_raw, topic_id, topic_id_2, topic_assignment, similarity, similarity_2` — the reference assignment (D9 tiers from the run's own similarity quantiles) | **no** (≈ 85 MB; Drive and the hand-back zip; decision W4) |
| `topics.csv` | one row per reference topic (+ the −1 row): the K1 counts (`n_cluster, n_reassigned, n_dual_first, n_dual_second, n_total, share`), keywords, representative posts, QA shares | yes |
| `topics_over_time.csv` | monthly counts of the counted posts per topic (`topic_id, period, n_docs, keywords` with `keywords` empty): computed without the model, K1 counts — unlike the nine bins' files, which keep the D9 counts (K7) | yes |
| `topic_embeddings.npy` | row t = the mean mpnet embedding of topic t's posts (the centroid used by script 16; the same space as `bin-N/topic_embeddings.npy`) | yes |
| `umap_5d.npy`, `hdbscan_labels_raw.npy` | the intermediate arrays (replay, design §11) | no (Drive) |
| `outlier_nearest.csv` | every raw outlier's two nearest topics and similarities (computed in chunks) | no (Drive) |
| `outlier_calibration/` | the band table and four 100-post judgement samples, written as for every run; nobody is asked to code them | yes |
| `labels/` | the two analysts' full transcripts, one JSON per topic | no (≈ 150 MB; Drive) |
| `labels.jsonl`, `topics_labelled.csv`, `topic_overrides.csv` | the labelling record (one line per topic, `period_bin` 0; both analysts' final answers), the labelled table, the override template (no co-author review is planned for the reference list) | yes |
| `config.json`, `status.json`, `log.txt`, `report.html`, `report.md` | effective settings (guard `enabled = false`, `mode = "fixed"`, decision W1), status (`bin` 0), log, the report (its bar charts show the 200 largest topics) | yes |
| `map_pairs.csv`, `map_period_topics.csv`, `reference_topics.csv` | the mapping tables written by `code/16-reference-map.R` (below) | yes |

## The mapping tables (`Rscript code/16-reference-map.R`, locally, after the hand-back; ≈ 2–3 min)

A post counts on either side only when it is **counted** there (decision K1: `topic_assignment` `cluster` or `reassigned`);
dual, outlier and not-modelled posts appear as uncounted shares. Read every CSV as text.

- `map_pairs.csv` — one row per (period topic, reference topic) pair with posts counted on both sides: `period_bin, topic_uid,
  ref_topic, n_posts, share_of_period_topic` (÷ the period topic's `n_total`), `share_of_reference_topic` (÷ the reference
  topic's `n_total`), `centroid_cosine` (cosine of the two topic centroids).
- `map_period_topics.csv` — one row per period topic (`topic_labels_all_bins.csv`): `n_both` (its posts counted on both sides),
  `best_ref_topic` / `best_ref_name` / `best_n` / `best_share` (the reference topic holding most of its counted posts),
  `second_ref_topic` / `second_share`, `n_ref_topics`, `nearest_ref_topic` / `nearest_cosine` (by centroid), `agree` (best =
  nearest), and the period topic's QUALTAG coding as the coder wrote it (`qt_coder` TP / BA / empty; `qt_political`,
  `qt_host_country_domestic`, `qt_national_identity`, `qt_fifa_body`, `qt_ir`).
- `reference_topics.csv` — one row per reference topic: `name, tone_category, needs_review, keywords, n_cluster, n_reassigned,
  n_total, n_bin1 … n_bin9` (its counted posts per period bin), `n_both`, `coverage` (= `n_both / n_total`),
  `top_period_topics` (the three largest by overlap, "uid name (n) | …"), and the composition of its counted posts by the status
  of their period topic, as counts `n_<status>` and shares `share_<status>`: `tp_yes, tp_no, tp_review, tp_to_settle` (TP's
  QUALTAG scheme, bins 1, 3, 5, 7, 9), `ba_1, ba_0, ba_blank` (BA's direct `qt_political`, bins 2 and 4), `uncoded_bin` (bins
  6 and 8, no review file), `unmapped` (the post is dual, outlier or not modelled on the period side).
- `results/descriptives/fig_G1_reference_best_share_by_bin.png|pdf` + `tab_G1_reference_best_share_by_bin.csv`: the best-match
  share by period bin — how cleanly the period topics fall into reference topics.

No QUALTAG tag is written onto a reference topic by these tables (decision W2): the transfer rule is decided on reading them.
