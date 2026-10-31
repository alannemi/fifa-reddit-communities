# results/descriptives/

The chapter's descriptive figures and tables (1a–3d), written by `code/18-chapter-figures.R` from the cleaned corpus (the 17
yearly files), the join table `data/analysis/post_variables.csv` and the topic-tag table `results/analysis/topic_tags.csv`
(`code/17-topic-tags.R`). Every figure comes as `fig_<id>.png` (300 dpi) in the folder, its `.pdf` in `pdfs/` and the numbers behind it in
`tabs/tab_<id>.csv` (the same layout inside the alternatives subfolder; Alan Nemirovski, 2026-10-31); `alternative visualizations/` holds the counterpart of each political figure (raw counts where the main version
shows shares, shares where it shows counts; the counts behind 3b–3d). Specified by Alan Nemirovski on 2026-10-31 (decisions
V1–V14, `docs/session-14_2026-10-31/00_approvals.md`). The earlier A–E set (scripts 11 and 14) was retired the same day, and the
diagnostics that lived here (D, F, G1, the sentiment validation tables) moved to `results/diagnostics/`.

- **Re-run:** `Rscript code/17-topic-tags.R && Rscript code/18-chapter-figures.R` (about 1 min) after any change to the review
  files in `results/analysis/manual review/` or to the join table. Every CSV is read as text.
- **Titles only:** no subtitle or caption on any figure; the populations and reading notes below go with them into the chapter text.

## Populations and rules

- **Political topics** (`topic_tags.csv`, rule V3): a topic carrying any of the four analytical categories — domestic politics
  (tag Host Country Domestic), international relations (IR), FIFA governance (FIFA Body), identity & nationalism (National
  Identity) — is political; a topic with none is non-political unless it carries REVIEW, in which case it is **excluded**
  (REVIEW beside a category is political). In BA's even bins `qt_political = 1` without a category is political and
  *uncategorised* (in the political share, in no category); blank = non-political (V9). The qualifier tags (WC, Club, Intl
  non-WC, player, humour) never enter an aggregate (V14). A topic with several categories counts fully in each (Q8), so the
  category shares of 2c/2d can add up to more than 100 % of the political posts.
- **Post weights (decision M1, V4):** a cluster or reassigned post weighs 1 in its topic, a dual post 0.5 in each of its two
  topics, an outlier or not-modelled post 0. A post's political weight is 0, 0.5 or 1; its category weights likewise.
  Denominators are all posts of the month / day / bin, each with weight 1. **Non-political = everything not political** (V5):
  1 − the political weight, so outliers, excluded and uncoded posts are non-political, and 2b is the exact complement of the
  political share. A bin without a review file would be *uncoded* (its months blank in the political series; the 2a table
  reports the share); since 2026-10-31 every bin is coded. The weights of every post are in `data/analysis/post_political_weights.csv` (not in git).
- `score` and `num_comments` are crawl-time values; the archive's crawl lag drops from weeks to hours in July 2023 (codebook
  limitation 11, §8.5), and the figures do not mark the break (decision E5, 2026-10-27).
- The VADER variables are the sentimental valence of a title's wording, not the author's sentiment (κ = 0.24 against one human
  coder; codebook §8.2). 3b and 3c use the weighted mean with the weights above; se = sqrt(Σ w (x − m)² / Σ w) / sqrt(Σ w).

| id | Figure / table | Population |
|---|---|---|
| 1a_monthly_volume | Posts per month, World Cup bins shaded | all 1,302,021 posts |
| 1b_monthly_comments_per_post | Mean `num_comments` per post per month (the table carries the median and quartiles too) | all posts |
| 1c_monthly_comments | Comments per month (the sum of `num_comments`) | all posts |
| 2a_political_share (table) | Political posts as a share of all posts, per period bin and overall; the uncategorised, excluded and uncoded shares beside it | all posts; the political weight (M1) |
| 2b_political_share_monthly | Political and non-political posts as shares of the month's posts | all posts |
| 2c_political_categories (table) | Political posts carrying each category, as a share of the political posts and of all posts, per bin and overall | the category weights |
| 2d_political_categories_by_world_cup | The four categories as shares of the bin's political posts, one group of bars per World Cup bin (the former E1, right half) | the five tournament bins |
| 2e_political_categories_daily | The categories day by day through each tournament: days from the opening match, knockout stage shaded, share of the day's posts over a centred 7-day window (the former E1b) | the tournament days of the five World Cup bins |
| 2g_comments_by_political_category_daily | The comments under each category's posts through each tournament (post weight × `num_comments`), as a share of all comments of the day, both over the centred 7-day window; 2e's layout (the table also gives the share of the comments under political posts) | the tournament days of the five World Cup bins; denominator = all comments of the day (Alan Nemirovski, 2026-10-31) |
| 2f_comments_political_share (+ `_by_bin`) | Comments under political posts as a share of the month's comments (Σ political weight × comments / Σ comments); the second table per bin and overall | all posts |
| 3a_political_categories_monthly | Share of the month's posts carrying each category, four lines, whole corpus | all posts |
| 3b_vader_by_political_category | Weighted mean VADER title score per category and World Cup with its 95 % CI; the non-political posts of the same bins as the grey reference (the former E5) | the tournament bins |
| 3c_vader_by_category_monthly | The same per month, with 95 % CI ribbons, whole corpus | all posts of coded months |
| 3d_political_user_concentration (+ `_tournaments_pooled`, `_by_tournament`; tables `3d_…` and `3d_…_curve`) | Users ranked by their posts in the group: the cumulative share of users against the cumulative share of the group's posts, political and non-political on one panel, the equality line, the Gini of each curve in the corner; three versions — the whole corpus, the five tournament bins pooled, one panel per tournament bin. The table: per scope and group, users, top-1 / 5 / 10 % shares, Gini, the share of the group's weight whose author reads `[deleted]` | posts with a named author (whole corpus: 32,031 users with political posts, 156,540 with non-political ones). `[deleted]` authors hold 13 % of the political weight: about 30 % of the posts crawled late (2010 – June 2023) carry no author, none from 2024 (limitation 11) |

`alternative visualizations/`: `2a_political_counts`, `2b_political_counts_monthly`, `2c_political_category_counts`,
`2d_political_categories_by_world_cup_counts`, `2e_political_categories_daily_counts` (posts over the 7-day window),
`2g_comments_by_political_category_daily_counts` (comments over the 7-day window), `2f_comments_political_counts`, `3a_political_categories_monthly_counts` (also the counts behind 3c),
`3b_posts_by_political_category` (log scale, with the non-political reference), `3d_political_user_concentration_counts`
(cumulative political posts against user rank).

**Reading notes.** The monthly category lines (3a, 3c) rest on few posts in the early years (2010: about 400 posts a month) and
are noisy; the 3c ribbons widen accordingly. Engagement levels before and after July 2023 are not comparable (limitation 11).
The political shares are 0.5–4.5 points above the retired E1 figures of session 10 for the same bins — the effect of the
half-weighted dual posts (M1) and of rule V3 (REVIEW beside a category, conflicts and untagged topics now decided).
