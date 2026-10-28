##############################################################################
# FIFA COMMUNITIES CHAPTER
## 11. DESCRIPTIVE FIGURES  (post volume, engagement and sentiment over time)
##############################################################################
#
# PURPOSE
# -------
# The chapter's descriptive figures of r/soccer 2010-2026, each with the table
# behind it.  Plan and decisions: docs/session-09_2026-10-27/00_approvals.md
# (E2-E7, Alan Nemirovski, 2026-10-27).
#   A  volume       A1 posts per month 2010-2026; A2 posts per week in each World
#                   Cup window, aligned on the kick-off
#   B  engagement   B1 median num_comments per month; B2 median score per month;
#                   B3 both per period bin (whole corpus, not stratified: E2)
#   C  sentiment    C1 mean VADER title score per month; C2 shares of positive /
#                   neutral / negative titles per month; C3 topic tones per bin;
#                   C4 rank of the most frequent tones across the bins
#
# POPULATIONS
# -----------
# A, B, C1, C2: every post of the clean corpus (1,302,021).  C3, C4: posts whose
# topic is their HDBSCAN cluster or a single reassignment (topic_assignment =
# cluster or reassigned; E3) — the dual and outlier tiers are left out because
# the coded judgement samples found most of their posts outside the given topic.
#
# READING NOTES
# -------------
# * score and num_comments are crawl-time values (codebook §3): the archive
#   crawled posts weeks to years after posting until June 2023 and within hours
#   from July 2023, so the engagement series is not a constant-lag measurement.
#   Not marked in the figures (E5); documented in docs/session-09_2026-10-27/02_session-notes.md.
# * The VADER variables are the sentimental valence of a post's wording, not the author's sentiment (author,
#   2026-10-27; validation against one coder: 56 % agreement, kappa 0.24; codebook §8.2).
#
# HOW TO RUN (repository root; reads the 17 yearly files and data/analysis/post_variables.csv,
# so it does not depend on stage S6)
# ----------------------------
#   Rscript code/11-descriptives.R          # ~2 min
#
# OUTPUTS  (results/descriptives/)
# -------
#   fig_<id>_<name>.png (300 dpi) and .pdf, tab_<id>_<name>.csv for every figure;
#   tab_C_other_contents.csv (the analysts' tones still mapped to "other", per bin)
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2); library(scales) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                     # shared helpers and the bin table (code/R/pipeline-utils.R)
ROOT <- find_root()
setwd(ROOT)

YEARLY  <- "data/yearly datasets"
PV      <- "data/analysis/post_variables.csv"
LABELS  <- "results/analysis/topic_labels_all_bins.csv"
OUT_DIR <- "results/descriptives"
N_CORPUS <- 1302021L
# Kick-off and final of each World Cup (mirrors config.WORLD_CUPS)
WC <- tibble(period_bin = c(1L, 3L, 5L, 7L, 9L), cup = c("2010", "2014", "2018", "2022", "2026"),
             kickoff = as.Date(c("2010-06-11", "2014-06-12", "2018-06-14", "2022-11-20", "2026-06-11")),
             final   = as.Date(c("2010-07-11", "2014-07-13", "2018-07-15", "2022-12-18", "2026-07-19")))

# ---------------------------------------------------------------------------
# Visual system (palette, theme, save_fig / save_tab): code/R/plot-utils.R
# ---------------------------------------------------------------------------
source(file.path(dirname(.utils), "plot-utils.R"))

# The World Cup bins as shaded bands behind a monthly series
wc_bands <- tibble(xmin = BIN_START[c(1, 3, 5, 7, 9)], xmax = BIN_END[c(1, 3, 5, 7, 9)] + 1, cup = WC$cup)
bands <- function(label_y = Inf) list(
  geom_rect(data = wc_bands, aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf), inherit.aes = FALSE, fill = SHADE),
  geom_text(data = wc_bands, aes(x = if_else(cup == "2026", xmax, xmin + (xmax - xmin) / 2), y = label_y,
                                 hjust = if_else(cup == "2026", 1, 0.5), label = paste("WC", cup)), inherit.aes = FALSE,
            vjust = 1.4, size = 2.4, colour = MUTED))                  # the last band ends at the axis: right-aligned
month_axis <- scale_x_date(date_breaks = "2 years", date_labels = "%Y", expand = expansion(mult = 0.01))

# ---------------------------------------------------------------------------
# 1. Data: date and engagement from the yearly files, analysis variables from the join table
# ---------------------------------------------------------------------------
files <- list.files(YEARLY, pattern = "^soccer\\.data\\.cleaned-\\d{4}\\.csv$", full.names = TRUE)
posts <- bind_rows(lapply(files, read_chr, col_select = c("id", "created_datetime", "score", "num_comments", "is_self")))
pv <- read_chr(PV, col_select = c("id", "period_bin", "topic_uid", "topic_tone", "topic_assignment",
                                   "post_sentiment", "post_sentiment_label"))
d <- posts |> inner_join(pv, by = "id") |>
  mutate(date = as.Date(substr(created_datetime, 1, 10)), month = as.Date(paste0(substr(created_datetime, 1, 7), "-01")),
         period_bin = as.integer(period_bin), score = as.numeric(score), num_comments = as.numeric(num_comments),
         post_sentiment = as.numeric(post_sentiment))
rm(posts); invisible(gc())
check("every post of the corpus joined once", nrow(d) == N_CORPUS && !anyDuplicated(d$id), sprintf("%d", nrow(d)))
log_msg("posts: %d; score missing %d; num_comments missing %d", nrow(d), sum(is.na(d$score)), sum(is.na(d$num_comments)))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
n_lab <- function(n) format(n, big.mark = ",")

# ---------------------------------------------------------------------------
# A. Volume
# ---------------------------------------------------------------------------
a1 <- d |> count(month, name = "posts")
check("A1 monthly counts sum to the corpus", sum(a1$posts) == N_CORPUS)
save_tab(a1, "A1_monthly_volume")
p <- ggplot(a1, aes(month, posts)) + bands() +
  geom_line(colour = SERIES[1], linewidth = 0.6) +
  month_axis + scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.08))) +
  labs(title = "Posts per month in r/soccer, 2010–2026", x = NULL, y = "Posts per month") +   # title only (author, 2026-10-28);
  theme_fifa()   # shaded = the five World Cup period bins; all posts of the clean corpus v2.1 (Arctic Shift archive)
save_fig(p, "A1_monthly_volume")

a2 <- d |> filter(period_bin %in% WC$period_bin) |> inner_join(WC, by = "period_bin") |>
  mutate(week = floor(as.numeric(date - kickoff) / 7)) |>
  count(cup, period_bin, kickoff, week, name = "posts") |>
  filter(kickoff + 7 * week >= BIN_START[period_bin], kickoff + 7 * week + 6 <= BIN_END[period_bin]) |>   # whole weeks only
  select(cup, week, posts) |>
  left_join(WC |> transmute(cup, final_week = floor(as.numeric(final - kickoff) / 7)), by = "cup")
save_tab(a2, "A2_worldcup_weekly_volume")
p <- ggplot(a2, aes(week, posts)) +
  geom_rect(data = distinct(a2, cup, final_week), aes(xmin = 0, xmax = final_week + 1, ymin = -Inf, ymax = Inf),
            inherit.aes = FALSE, fill = SHADE) +
  geom_line(colour = SERIES[1], linewidth = 0.6) +
  geom_vline(xintercept = 0, colour = AXIS, linewidth = 0.3) +
  facet_wrap(~ paste("World Cup", cup), ncol = 1, scales = "free_y", strip.position = "top") +
  expand_limits(y = 0) +
  scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.1)), n.breaks = 3) +
  labs(title = "Posts per week around each World Cup",
       subtitle = "Weeks relative to the opening match (week 0); shaded: the tournament, kick-off to final. Y-scales differ by panel.",
       x = "Weeks from kick-off", y = "Posts per week",
       caption = "Each panel is the World Cup period bin (codebook §4.9), whole weeks only (partial weeks at the bin edges dropped).") +
  theme_fifa()
save_fig(p, "A2_worldcup_weekly_volume", h = 7)

# ---------------------------------------------------------------------------
# B. Engagement (whole corpus; medians with the interquartile range)
# ---------------------------------------------------------------------------
q_month <- function(var) d |> filter(!is.na(.data[[var]])) |> group_by(month) |>
  summarise(n = n(), median = median(.data[[var]]), q25 = quantile(.data[[var]], 0.25), q75 = quantile(.data[[var]], 0.75), .groups = "drop")
eng_plot <- function(t, title, ylab, what, clip_top = FALSE) {
  # clip_top: one month whose upper quartile is far above every other would flatten the series; the view stops just above
  # the second-highest month and the caption names the clipped month (its values stay in the table)
  ymax <- if (clip_top) sort(t$q75, decreasing = TRUE)[2] * 1.08 else NA
  top <- t[which.max(t$q75), ]
  note <- if (clip_top) sprintf(" Band clipped in %s (upper quartile %s).", format(top$month, "%B %Y"), n_lab(round(top$q75))) else ""
  ggplot(t, aes(month, median)) + bands() +
    geom_ribbon(aes(ymin = q25, ymax = q75), fill = SERIES[1], alpha = 0.15) +
    geom_line(colour = SERIES[1], linewidth = 0.6) +
    month_axis + scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, 0.05))) +
    coord_cartesian(ylim = c(0, ymax)) +
    labs(title = title, subtitle = "Monthly median (line) and interquartile range (band); shaded: World Cup period bins",
         x = NULL, y = ylab,
         caption = sprintf("All %s posts of the clean corpus. %s is the value when the archive crawled the post (codebook §3).%s", n_lab(N_CORPUS), what, note)) +
    theme_fifa()
}
b1 <- q_month("num_comments"); save_tab(b1, "B1_monthly_comments")
save_fig(eng_plot(b1, "Comments per post, 2010–2026", "Comments per post", "num_comments"), "B1_monthly_comments")
b2 <- q_month("score"); save_tab(b2, "B2_monthly_score")
save_fig(eng_plot(b2, "Score per post, 2010–2026", "Score (net votes)", "score", clip_top = TRUE), "B2_monthly_score")

b3 <- bind_rows(
  d |> group_by(period_bin) |> summarise(n = n(), median = median(num_comments, na.rm = TRUE), q25 = quantile(num_comments, 0.25, na.rm = TRUE),
                                         q75 = quantile(num_comments, 0.75, na.rm = TRUE), .groups = "drop") |> mutate(metric = "Comments per post"),
  d |> group_by(period_bin) |> summarise(n = n(), median = median(score, na.rm = TRUE), q25 = quantile(score, 0.25, na.rm = TRUE),
                                         q75 = quantile(score, 0.75, na.rm = TRUE), .groups = "drop") |> mutate(metric = "Score (net votes)")) |>
  mutate(period_type = BIN_TYPE[period_bin],
         bin_label = factor(sprintf("%d\n%s–%s", period_bin, format(BIN_START[period_bin], "%b %y"), format(BIN_END[period_bin], "%b %y")),
                            levels = sprintf("%d\n%s–%s", 1:9, format(BIN_START, "%b %y"), format(BIN_END, "%b %y"))))
save_tab(b3 |> select(-bin_label), "B3_engagement_by_bin")
p <- ggplot(b3, aes(bin_label, median, colour = period_type)) +
  geom_linerange(aes(ymin = q25, ymax = q75), linewidth = 0.8) + geom_point(size = 2.4) +
  facet_wrap(~ metric, ncol = 1, scales = "free_y") +
  scale_colour_manual(values = c(tournament = SERIES[1], interim = SERIES[2]), breaks = c("tournament", "interim"),
                      labels = c(tournament = "World Cup period", interim = "Interim period")) +
  scale_y_continuous(labels = label_comma()) +
  labs(title = "Engagement by period bin", subtitle = "Median (dot) and interquartile range (bar) per bin",
       x = "Period bin", y = NULL,
       caption = sprintf("All %s posts. Crawl-time values (codebook §3).", n_lab(N_CORPUS))) +
  theme_fifa() + theme(axis.text.x = element_text(size = 7))
save_fig(p, "B3_engagement_by_bin", h = 6)

# ---------------------------------------------------------------------------
# C. Sentiment
# ---------------------------------------------------------------------------
c1 <- d |> group_by(month) |>
  summarise(n = n(), mean = mean(post_sentiment), se = sd(post_sentiment) / sqrt(n()), .groups = "drop") |>
  mutate(lo95 = mean - 1.96 * se, hi95 = mean + 1.96 * se)
save_tab(c1, "C1_monthly_vader_mean")
p <- ggplot(c1, aes(month, mean)) + bands() +
  geom_hline(yintercept = 0, colour = AXIS, linewidth = 0.4) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), fill = SERIES[1], alpha = 0.18) +
  geom_line(colour = SERIES[1], linewidth = 0.6) +
  month_axis + labs(title = "Sentimental valence of post titles, 2010–2026",
                    subtitle = "Monthly mean VADER compound score of the title (−1 to 1), with its 95 % confidence interval; shaded: World Cup period bins",
                    x = NULL, y = "Mean VADER score",
                    caption = sprintf("All %s posts (post_sentiment): the valence of the title's wording, not the author's sentiment. Validation against one human coder: 56 %% agreement, κ = 0.24 (codebook §8.2).", n_lab(N_CORPUS))) +
  theme_fifa()
save_fig(p, "C1_monthly_vader_mean")

c2 <- d |> count(month, post_sentiment_label) |> group_by(month) |> mutate(share = n / sum(n)) |> ungroup() |>
  mutate(post_sentiment_label = factor(post_sentiment_label, c("positive", "neutral", "negative")))
save_tab(c2, "C2_monthly_vader_labels")
last <- c2 |> filter(month == max(month))
p <- ggplot(c2, aes(month, share, colour = post_sentiment_label)) + bands() +
  geom_line(linewidth = 0.6) +
  geom_text(data = last, aes(label = post_sentiment_label), hjust = -0.1, size = 2.8, colour = INK2, show.legend = FALSE) +
  scale_colour_manual(values = c(positive = POS, neutral = NEU, negative = NEG)) +
  scale_x_date(breaks = as.Date(sprintf("%d-01-01", seq(2011, 2025, 2))), date_labels = "%Y", expand = expansion(mult = c(0.01, 0.07))) +
  scale_y_continuous(labels = label_percent(), limits = c(0, NA), expand = expansion(mult = c(0, 0.05))) +
  labs(title = "Positive, neutral and negative valence of post titles, 2010–2026",
       subtitle = "Monthly share of posts by VADER label (positive > 0.05, negative < −0.05); shaded: World Cup period bins",
       x = NULL, y = "Share of posts",
       caption = sprintf("All %s posts (post_sentiment_label): the valence of the title's wording, not the author's sentiment.\nVADER gives many titles that a reader sees as neutral a positive or negative label (validation against one human coder: 56 %% agreement, κ = 0.24; codebook §8.2).", n_lab(N_CORPUS))) +
  theme_fifa()
save_fig(p, "C2_monthly_vader_labels")

# C3/C4: the topic tone, on cluster and reassigned posts (E3)
tp <- d |> filter(topic_assignment %in% c("cluster", "reassigned"))
check("C3 population = cluster + reassigned posts, each with a tone", nrow(tp) == 827744 && all(tp$topic_tone != ""), n_lab(nrow(tp)))
c3 <- tp |> count(period_bin, topic_tone) |> group_by(period_bin) |> mutate(share = n / sum(n)) |> ungroup()
check("C3 tone shares sum to 1 in every bin", all(abs(tapply(c3$share, c3$period_bin, sum) - 1) < 1e-9))
order_tones <- c3 |> group_by(topic_tone) |> summarise(n = sum(n), .groups = "drop") |> arrange(desc(n)) |> pull(topic_tone)
order_tones <- c(setdiff(order_tones, "other"), "other")            # the residual category last
c3 <- c3 |> complete(period_bin = 1:9, topic_tone = order_tones, fill = list(n = 0, share = 0)) |>
  mutate(topic_tone = factor(topic_tone, rev(order_tones)), bin_lab = factor(period_bin, 1:9, sprintf("%d%s", 1:9, ifelse(BIN_TYPE == "tournament", "\nWC", ""))))
save_tab(c3 |> select(period_bin, topic_tone, n, share), "C3_tone_by_bin")
p <- ggplot(c3, aes(bin_lab, topic_tone, fill = share)) +
  geom_tile(colour = SURFACE, linewidth = 0.8) +
  geom_text(aes(label = ifelse(share >= 0.005, sprintf("%.0f", 100 * share), "·"), colour = share > 0.12), size = 2.5, show.legend = FALSE) +
  scale_fill_gradientn(colours = SEQ, labels = label_percent(), name = "Share of posts") +
  scale_colour_manual(values = c(`FALSE` = INK, `TRUE` = "#ffffff")) +
  labs(title = "Emotional tone of the topics, by period bin",
       subtitle = "Share of posts (%) whose topic the two analysts gave each tone; columns sum to 100. WC = World Cup period bin",
       x = "Period bin", y = NULL,
       caption = sprintf("Posts in their HDBSCAN cluster or singly reassigned to a topic (%s of %s; dual-topic and outlier posts excluded). Tone = config.TONE_SET incl. the four\ncategories added on 2026-10-27 (E6); 'other' = analyst tones outside the vocabulary (tab_C_other_contents.csv). '·' = under 0.5 %%.", n_lab(nrow(tp)), n_lab(N_CORPUS))) +
  theme_fifa() + theme(panel.grid.major.y = element_blank(), axis.line.x = element_blank(), legend.position = "right",
                       legend.title = element_text(colour = INK2, size = 8), axis.text.y = element_text(colour = INK2, size = 8))
save_fig(p, "C3_tone_by_bin", h = 6.5)

c4 <- c3 |> mutate(topic_tone = as.character(topic_tone)) |> group_by(period_bin) |>
  mutate(rank = rank(-share, ties.method = "first")) |> ungroup()
top <- c4 |> filter(rank <= 5) |> distinct(topic_tone) |> pull(topic_tone)
top <- intersect(order_tones, top)[seq_len(min(8, length(top)))]       # at most eight series, in overall order
c4 <- c4 |> filter(topic_tone %in% top)
save_tab(c4 |> select(period_bin, topic_tone, n, share, rank) |> arrange(period_bin, rank), "C4_tone_ranks")
cols <- setNames(SERIES[seq_along(setdiff(top, "other"))], setdiff(top, "other")); if ("other" %in% top) cols["other"] <- MUTED
p <- ggplot(c4, aes(period_bin, rank, colour = topic_tone)) +
  geom_line(linewidth = 0.7) + geom_point(size = 2.6, colour = SURFACE) + geom_point(size = 2) +
  geom_text(data = filter(c4, period_bin == 9), aes(label = topic_tone), hjust = -0.25, size = 2.8, colour = INK2) +
  scale_y_reverse(breaks = 1:max(c4$rank)) +
  scale_x_continuous(breaks = 1:9, labels = sprintf("%d%s", 1:9, ifelse(BIN_TYPE == "tournament", "\nWC", "")), expand = expansion(add = c(0.3, 1.4))) +
  scale_colour_manual(values = cols, breaks = top) +
  labs(title = "Which tones lead, bin by bin",
       subtitle = "Rank of each tone by share of posts in the bin (1 = most frequent); the tones that reach the top five in any bin",
       x = "Period bin", y = "Rank",
       caption = sprintf("Same population as the tone heatmap: %s posts in their cluster or singly reassigned. 'other' = analyst tones outside the vocabulary.", n_lab(nrow(tp)))) +
  theme_fifa() + theme(panel.grid.major.y = element_line(colour = GRID, linewidth = 0.2))
save_fig(p, "C4_tone_ranks")

# What is still "other": the analysts' own words, weighted by posts
lab <- read_chr(LABELS, col_select = c("topic_uid", "emotion_tone"))
oth <- tp |> filter(topic_tone == "other") |> count(period_bin, topic_uid) |> left_join(lab, by = "topic_uid") |>
  mutate(emotion_tone = tolower(trimws(emotion_tone))) |>
  group_by(period_bin, emotion_tone) |> summarise(n_topics = n(), n_posts = sum(n), .groups = "drop") |>
  group_by(period_bin) |> mutate(share_of_other = round(n_posts / sum(n_posts), 4)) |> arrange(period_bin, desc(n_posts)) |> ungroup()
save_tab(oth, "C_other_contents")

if (!all(unlist(results))) stop("a check failed")
log_msg("all figures and tables written to %s", OUT_DIR)
