# results/descriptives/

Descriptive figures of the r/soccer corpus (2010–2026) and the validation tables of the sentiment measure. Every figure comes as `fig_<id>.png` (300 dpi) and `.pdf`, with the numbers behind it in `tab_<id>.csv`.

- **Made by:** `code/11-descriptives.R`, which needs no other step first: `Rscript code/11-descriptives.R`, about 1 min. The validation tables (`sentiment_validation_*.csv`) come from `code/10-sentiment-validation.R`, once the coders have returned their sheets.
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

**Reading notes:**
- `score` and `num_comments` are crawl-time values. The archive's crawl lag drops from weeks to hours in July 2023 (session notes §4), so engagement levels before and after mid-2023 are not directly comparable.
- VADER is not yet validated against human coding.
