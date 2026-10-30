# results/descriptives/

Descriptive figures of the r/soccer corpus (2010–2026) and the validation tables of the sentiment measure. Every figure comes as `fig_<id>.png` (300 dpi) and `.pdf`, with the numbers behind it in `tab_<id>.csv`.

- **Made by:** `code/11-descriptives.R`, which needs no other step first: `Rscript code/11-descriptives.R`, about 1 min. The E figures come from `code/14-worldcup-figures.R` (about 20 s; it reads the QUALTAG variables written by `code/13-recode-qualtag.R`). The D figures come from `code/12-topic-network.R` (about 20 s; it reads the topic embeddings through the `.venv` Python). The validation tables (`sentiment_validation_*.csv`) come from `code/10-sentiment-validation.R`, once the coders have returned their sheets. The F tables and figures come from `code/15-outlier-calibration.R` (about 30 s; it reads the 36 coded outlier judgement samples under `results/analysis/bin-N/outlier_calibration/`). The G figure comes from `code/16-reference-map.R` (about 2–3 min; it reads the whole-corpus reference run's post table, which is not in git, beside `post_variables.csv`; session 12).
- **Decisions and notes:** `docs/session-09_2026-10-27/` (`00_approvals.md` E1–E7, `02_session-notes.md` §3–§4). The F figures: `docs/session-11_2026-10-30/` (the assessment `01_outlier-calibration-results.md`, decisions K1–K7 in `00_approvals.md`).

| id | Figure | Population |
|---|---|---|
| A1_monthly_volume | Posts per month, World Cup bins shaded | all 1,302,021 posts |
| A2_worldcup_weekly_volume | Posts per week around each World Cup, aligned on kick-off | the five World Cup bins |
| B1_monthly_comments | Median `num_comments` per month, with IQR | all posts |
| B2_monthly_score | Median `score` per month, with IQR (the view is clipped above the second-highest month) | all posts |
| B3_engagement_by_bin | Both metrics per period bin, median and IQR | all posts |
| C1_monthly_vader_mean | Mean VADER title score per month, with 95 % CI | all posts |
| C2_monthly_vader_labels | Shares of positive / neutral / negative titles per month | all posts |
| C3_tone_by_bin | Topic tone composition per bin (heatmap) | `cluster` + `reassigned` posts (827,744) |
| C4_tone_ranks | Rank of the leading tones across the bins | as C3 |
| C_other_contents (table only) | The analysts' tones still mapped to `other`, per bin | as C3 |
| D1_topic_network_families | Topic network (semantic similarity, all bins), the eight largest families coloured and labelled F1–F8 | all 6,245 topics (6,053 in the main network) |
| D1b_topic_network_families_unlabelled | The same without labels on the map, with larger points | as D1 |
| D2_topic_network_periods | The same layout, one panel per period bin | as D1 |
| D1_topic_families / _nodes / _edges (tables) | Families (size, bins, largest topics, tones); topics with family and position; links with similarity | as D1 |
| E1_political_posts | Left: political posts as a share of all posts in each World Cup bin. Right: share of those posts in each political category (national identity, FIFA body, host country domestic, IR; a topic can carry several) | posts in cluster or singly reassigned to a topic coded political (`qt_political = yes`, `code/13-recode-qualtag.R`); denominator all posts of the bin |
| E1_political_categories (table) | The category counts and shares behind the right-hand panel | as E1 |
| E1b_political_categories_daily | The four categories day by day through each tournament: one panel per World Cup, days from the opening match, knockout stage shaded, shared y-axis; share of the day's posts over a centred 7-day window | as E1; denominator all posts of the day |
| E2_worldcup_weekly_vader | Mean VADER title score per week around each World Cup, aligned on kick-off, with 95 % CI | all posts of the five World Cup bins |
| E3_monthly_subscribers | Monthly median of `subreddit_subscribers`, March 2018 – July 2026 (crawl-time values; none before 2018) | posts with a subscriber value |
| E5_vader_by_political_category | Mean VADER title score of the political posts per category and World Cup, 95 % CI; non-political posts of the same bins as the reference (grey) | as E1 (a topic with several tags counts in each category); non-political = cluster + reassigned posts of `qt_political = no` topics |
| E6_tones_by_political_category | The analysts' topic tones in each political category, the five World Cups pooled (heatmap of shares); non-political column for reference | as E5 |
| E7_tone_ranks_by_political_category | E4's layout with one panel per political category and the non-political posts: the ten most frequent topic tones, ranked by share of posts, five World Cups pooled (the table ranks every tone, with its topic count) | as E5 |
| E4_worldcup_tone_ranks | The ten most frequent topic tones in each World Cup bin, ranked by share of posts (the table ranks every tone) | `cluster` + `reassigned` posts of the World Cup bins |
| F1_outlier_precision_by_band | Share of the admitted outlier posts judged to belong to their topic, per similarity band (35th, 50th, 65th, ≥ 80th percentile) and bin, with 95 % Wilson intervals and the coder named | the 36 coded judgement samples: 100 raw outliers per band and bin, 3,600 in all (BA bins 1, 3, 5, 7; AM 6, 8, 9; AN 2, 4) |
| F2_outlier_precision_by_similarity | The same share against the absolute cosine similarity of the post to its nearest topic, one line per coder and pooled | as F1 |
| F1 / F2 / F3 (tables) | Precision per bin × band; pooled per band, tier, coder and period type (plain pooling, the figure of record, beside the corpus-weighted and equal-coder variants); per absolute-similarity band | as F1 |
| F4 / F4b (tables) | What each candidate reassignment floor would leave as outliers, per bin; corpus-wide tiers under candidate (floor, single-topic) quantile pairs | every raw outlier's similarity to its nearest topic (`outlier_nearest.csv`, nine bins) |
| G1_reference_best_share_by_bin | Share of a period topic's counted posts that fall into its best-matching topic of the whole-corpus reference model, box plots per period bin (the table: median, quartiles, share above one half, agreement of the post-overlap and centroid methods) | the period topics with posts counted (cluster + reassigned) on both sides; the reference model of session 12 (`results/analysis/reference/`) |

**Reading notes:**
- `score` and `num_comments` are crawl-time values. The archive's crawl lag drops from weeks to hours in July 2023 (session notes §4), and more posts that moderators later removed remain in the clean corpus from then on. Volume and engagement before and after mid-2023 are therefore not directly comparable (codebook limitation 11, §8.5).
- The VADER variables are the sentimental valence of a post's wording, not the author's sentiment. Validation against one human coder: 56 % agreement, κ = 0.24 (codebook §8.2). The tables `sentiment_validation_*.csv` hold the validation figures.
- The outlier tiers (decision D9) admit about half right: 53.4 % of the singly reassigned posts and 43.6 % of the dual-tier posts were judged to belong to their topic (F1–F3; single-coded, and the coders differ). Since decision K1 (2026-10-30) the topic table counts cluster + reassigned posts only, the population the C, D and E figures already used.
