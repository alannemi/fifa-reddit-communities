# results/descriptives/

Descriptive figures of the r/soccer corpus (2010–2026) and the validation tables of the sentiment measure. Every figure comes as `fig_<id>.png` (300 dpi) and `.pdf`, with the numbers behind it in `tab_<id>.csv`.

- **Made by:** `code/11-descriptives.R`, which needs no other step first: `Rscript code/11-descriptives.R`, about 1 min. The D figures come from `code/12-topic-network.R` (about 20 s; it reads the topic embeddings through the `.venv` Python). The validation tables (`sentiment_validation_*.csv`) come from `code/10-sentiment-validation.R`, once the coders have returned their sheets.
- **Decisions and notes:** `docs/session-09_2026-10-27/` (`00_approvals.md` E1–E7, `02_session-notes.md` §3–§4).

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

**Reading notes:**
- `score` and `num_comments` are crawl-time values. The archive's crawl lag drops from weeks to hours in July 2023 (session notes §4), and more posts that moderators later removed remain in the clean corpus from then on. Volume and engagement before and after mid-2023 are therefore not directly comparable (codebook limitation 11, §8.5).
- The VADER variables are the sentimental valence of a post's wording, not the author's sentiment. Validation against one human coder: 56 % agreement, κ = 0.24 (codebook §8.2). The tables `sentiment_validation_*.csv` hold the validation figures.
