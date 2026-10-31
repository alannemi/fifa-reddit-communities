##############################################################################
# FIFA COMMUNITIES CHAPTER
## 18. CHAPTER FIGURES  (volume, comments, political posts, categories, sentiment and users, 2010-2026)
##############################################################################
#
# PURPOSE
# -------
# The descriptive figures and tables of the chapter, as Alan Nemirovski specified them on 2026-10-31 (decisions
# V1-V14, docs/session-14_2026-10-31/00_approvals.md), built from the cleaned corpus (the 17 yearly files), the join
# table (data/analysis/post_variables.csv) and the topic-tag table of 17-topic-tags.R.  They replace the A-C and E
# figures of the retired scripts 11-descriptives.R and 14-worldcup-figures.R.
#   1a posts per month; 1b mean comments per post per month; 1c comments per month
#   2a share of political posts, table (per bin and overall); 2b political and non-political posts per month (shares)
#   2c political posts per category, table; 2d categories by World Cup (share of the bin's political posts)
#   2e categories day by day through each World Cup; 2f share of comments under political posts per month; 2g the comments
#   under each category's posts through each World Cup, as a share of all comments of the day
#   3a categories per month, whole corpus (shares); 3b comments under each category's posts per month, share of the month's
#   comments; 3c mean VADER score by category and World Cup (non-political as the reference); 3d mean VADER score by category
#   per month; 3e concentration of political posts among users (renumbered 2026-10-31: 3b-3d became 3c-3e when 3b was added)
# The alternative versions (raw counts where the main figure shows shares, shares where it shows counts; the counts
# behind 3c-3e) go to results/descriptives/alternative visualizations/ with the same ids and a "_counts" suffix.
#
# POPULATIONS AND RULES
# ---------------------
# * Topic status and categories: results/analysis/topic_tags.csv (17-topic-tags.R; rule V3: political / non-political /
#   excluded; cat_domestic, cat_ir, cat_fifa, cat_identity).  A bin without a review file is "uncoded".
# * Post weights (decision M1, 2026-10-31; V4): a cluster or reassigned post weighs 1 in its topic, a dual post 0.5 in
#   each of its two topics, an outlier or not-modelled post 0.  w_political = the post's weight in political topics
#   (0, 0.5 or 1); w_<category> the same per category (a topic with several categories counts fully in each, Q8);
#   w_excluded and w_uncoded likewise.  Non-political = everything not political (V5): w_nonpol = 1 - w_political,
#   so outliers, excluded and uncoded posts are non-political.  Denominators are all posts of the month / day / bin,
#   each with weight 1.  Months inside an uncoded bin show no political share (a gap).
# * The qualifier tags (WC, Club, Intl non-WC, player, humour) never enter an aggregate (V14).
# * Comments are crawl-time values; the July-2023 crawl break (codebook limitation 11) is not marked (decision E5).
# * 3c / 3d weighted mean and se: m = sum(w x) / sum(w); se = sqrt(sum(w (x - m)^2) / sum(w)) / sqrt(sum(w)); effective n = sum(w).
# * 3e: authors other than "[deleted]", ranked by their political weight; x = the cumulative share of users, y = the
#   cumulative share of political weight; the Gini index from the same ranking; the same curve for the non-political posts
#   (1 - the political weight) as the reference on the same panel, the Ginis in the corner; for the whole corpus and one
#   panel per tournament bin.  "[deleted]" authors are about 30 % of
#   the posts the archive crawled late (2010 - June 2023) and none from 2024: the table reports their share.
# * Titles only: no subtitle, no caption (author, 2026-10-28 / 2026-10-31).
#
# HOW TO RUN (repository root; after 17-topic-tags.R)
# ----------------------------
#   Rscript code/18-chapter-figures.R          # ~3 min
#   FIG_TAGS, FIG_OUT_DIR, FIG_WEIGHTS redirect the inputs / outputs for a test.
#
# OUTPUTS
# -------
#   results/descriptives/fig_<id>.png for 1a-3e, the tables tab_<id>.csv in tabs/, the PDFs in pdfs/ (2a, 2c: tables only; 2f and 3d: a second table);
#   results/descriptives/alternative visualizations/ (the alternatives); data/analysis/post_political_weights.csv (every
#   post's weights; not in git, like post_variables.csv)
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2); library(scales) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                                     # shared helpers and the bin table (code/R/pipeline-utils.R)
source(file.path(dirname(.utils), "plot-utils.R")) # palette, theme, save_fig / save_tab
ROOT <- find_root()
setwd(ROOT)

YEARLY   <- "data/yearly datasets"
PV       <- "data/analysis/post_variables.csv"
TAGS     <- Sys.getenv("FIG_TAGS", "results/analysis/topic_tags.csv")
OUT_DIR  <- Sys.getenv("FIG_OUT_DIR", "results/descriptives")
ALT_DIR  <- file.path(OUT_DIR, "alternative visualizations")
WEIGHTS  <- Sys.getenv("FIG_WEIGHTS", "data/analysis/post_political_weights.csv")
N_CORPUS <- 1302021L
# Kick-off and final of each World Cup (mirrors config.WORLD_CUPS); first knockout match (2026: the round of 32)
WC <- tibble(period_bin = c(1L, 3L, 5L, 7L, 9L), cup = c("2010", "2014", "2018", "2022", "2026"),
             kickoff = as.Date(c("2010-06-11", "2014-06-12", "2018-06-14", "2022-11-20", "2026-06-11")),
             final   = as.Date(c("2010-07-11", "2014-07-13", "2018-07-15", "2022-12-18", "2026-07-19")),
             knockout = as.Date(c("2010-06-26", "2014-06-28", "2018-06-30", "2022-12-03", "2026-06-28")))
ROLL_HALF <- 3L                    # 2e: centred rolling window of 2 x 3 + 1 = 7 days
# The four analytical categories, named as in the request of 2026-10-31; each keeps the colour it had in the E figures
CATS      <- c(cat_domestic = "Domestic politics", cat_ir = "International relations", cat_fifa = "FIFA governance",
               cat_identity = "Identity & nationalism")
CAT_COLS  <- setNames(SERIES[c(3, 4, 2, 1)], CATS)
CAT_LINES <- setNames(c("solid", "dashed", "dotted", "dotdash"), CATS)
POL_LAB <- "Political"; NONPOL_LAB <- "Non-political"
POL_COLS <- setNames(c(INK2, MUTED), c(POL_LAB, NONPOL_LAB))
POL_LINES <- setNames(c("solid", "longdash"), c(POL_LAB, NONPOL_LAB))   # 2b, 3d: the non-political line dashed (Alan, 2026-10-31)
W_CATS  <- paste0("w_", names(CATS))
n_lab   <- function(n) format(n, big.mark = ",")
cup_lab <- function(b) factor(paste("WC", WC$cup[match(b, WC$period_bin)]), paste("WC", WC$cup))
pct     <- label_percent(accuracy = 1)
# The World Cup bins as shaded bands behind a monthly series (as 11-descriptives.R did)
wc_bands <- tibble(xmin = BIN_START[WC$period_bin], xmax = BIN_END[WC$period_bin] + 1, cup = WC$cup)
bands <- function(label_y = Inf) list(
  geom_rect(data = wc_bands, aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf), inherit.aes = FALSE, fill = SHADE),
  geom_text(data = wc_bands, aes(x = if_else(cup == "2026", xmax, xmin + (xmax - xmin) / 2), y = label_y,
                                 hjust = if_else(cup == "2026", 1, 0.5), label = paste("WC", cup)), inherit.aes = FALSE,
            vjust = 1.4, size = 2.4, colour = MUTED))                  # the last band ends at the axis: right-aligned
month_axis <- scale_x_date(date_breaks = "2 years", date_labels = "%Y", expand = expansion(mult = 0.01))
y_count <- function(top = 0.08) scale_y_continuous(labels = label_comma(), expand = expansion(mult = c(0, top)))
y_share <- function(top = 0.08) scale_y_continuous(labels = pct, limits = c(0, NA), expand = expansion(mult = c(0, top)))
line_series <- function(data, y, title, ylab, cols, lines = NULL, legend_rows = 1) {
  p <- ggplot(data, aes(month, {{ y }}, colour = group)) + bands() + geom_line(linewidth = 0.6, na.rm = TRUE) +
    scale_colour_manual(values = cols) + month_axis + labs(title = title, x = NULL, y = ylab) + theme_fifa() +
    guides(colour = guide_legend(nrow = legend_rows, byrow = TRUE))
  if (!is.null(lines)) p <- p + aes(linetype = group) + scale_linetype_manual(values = lines) +
    guides(colour = guide_legend(nrow = legend_rows, byrow = TRUE), linetype = guide_legend(nrow = legend_rows, byrow = TRUE)) +
    theme(legend.key.width = unit(2.2, "lines"))
  p
}
roll_sum <- function(x, k) vapply(seq_along(x), function(i) sum(x[max(1, i - k):min(length(x), i + k)]), numeric(1))
wstats <- function(x, w) {                         # weighted mean and se (header); effective n = sum(w)
  ok <- !is.na(x) & w > 0; x <- x[ok]; w <- w[ok]; sw <- sum(w)
  m  <- if (sw > 0) sum(w * x) / sw else NA_real_
  se <- if (sw > 1) sqrt(sum(w * (x - m)^2) / sw) / sqrt(sw) else NA_real_
  tibble(w_posts = sw, mean = m, se = se)
}
dir.create(ALT_DIR, recursive = TRUE, showWarnings = FALSE)
fig <- function(p, id, w = 8, h = 4.5, dir = OUT_DIR) save_fig(p, id, w, h, dir = dir, pdf_dir = file.path(dir, "pdfs"))   # PNG in the folder,
tab <- function(t, id, dir = OUT_DIR) save_tab(t, id, dir = file.path(dir, "tabs"))                                   # PDF in pdfs/, table in tabs/ (Alan, 2026-10-31)

# ---------------------------------------------------------------------------
# 1. Data: dates, comments and authors from the yearly files; topics, tiers and sentiment from the join table;
#    the topic status from 17-topic-tags.R -> every post's weights (M1)
# ---------------------------------------------------------------------------
files <- list.files(YEARLY, pattern = "^soccer\\.data\\.cleaned-\\d{4}\\.csv$", full.names = TRUE)
posts <- bind_rows(lapply(files, read_chr, col_select = c("id", "created_datetime", "num_comments", "author")))
pv <- read_chr(PV, col_select = c("id", "period_bin", "topic_uid", "topic_uid_2", "topic_assignment", "post_sentiment"))
d <- posts |> inner_join(pv, by = "id") |>
  mutate(date = as.Date(substr(created_datetime, 1, 10)), month = as.Date(paste0(substr(created_datetime, 1, 7), "-01")),
         period_bin = as.integer(period_bin), num_comments = suppressWarnings(as.numeric(num_comments)),
         post_sentiment = suppressWarnings(as.numeric(post_sentiment)))
rm(posts, pv); invisible(gc())
check("every post of the corpus joined once", nrow(d) == N_CORPUS && !anyDuplicated(d$id), n_lab(nrow(d)))
log_msg("num_comments missing %d; post_sentiment missing %d", sum(is.na(d$num_comments)), sum(is.na(d$post_sentiment)))

tags <- read_chr(TAGS) |>
  transmute(topic_uid, period_bin = as.integer(period_bin), political = as.numeric(status == "political"),
            excluded = as.numeric(status == "excluded"), pol_uncat = as.numeric(uncategorised == "1"),
            across(all_of(names(CATS)), ~ as.numeric(.x == "1")))
coded_bins <- sort(unique(tags$period_bin)); uncoded_bins <- setdiff(1:9, coded_bins)
if (length(uncoded_bins)) log_msg("WARNING: no topic tags for bin(s) %s: their posts count as uncoded (no political share)",
                                  paste(uncoded_bins, collapse = ", "))
TAG_VARS <- c("political", "excluded", "pol_uncat", names(CATS))
t1 <- tags |> select(topic_uid, all_of(TAG_VARS)) |> rename_with(~ paste0(.x, "_1"), all_of(TAG_VARS))
t2 <- tags |> select(topic_uid, all_of(TAG_VARS)) |> rename_with(~ paste0(.x, "_2"), all_of(TAG_VARS))
d <- d |> left_join(t1, by = "topic_uid") |> left_join(t2, by = c("topic_uid_2" = "topic_uid")) |>
  mutate(w1 = case_when(topic_assignment %in% c("cluster", "reassigned") ~ 1, topic_assignment == "dual" ~ 0.5, TRUE ~ 0),
         w2 = if_else(topic_assignment == "dual", 0.5, 0), coded = period_bin %in% coded_bins)
check("every topic that carries weight in a coded bin has a row in topic_tags.csv",
      !anyNA(d$political_1[d$w1 > 0 & d$coded]) && !anyNA(d$political_2[d$w2 > 0 & d$coded]))
check("a dual post carries both topics, every other post at most one", all((d$w2 > 0) == (d$topic_assignment == "dual")))
wsum <- function(a, b) d$w1 * coalesce(a, 0) + d$w2 * coalesce(b, 0)
d$w_political <- wsum(d$political_1, d$political_2); d$w_excluded <- wsum(d$excluded_1, d$excluded_2)
d$w_pol_uncat <- wsum(d$pol_uncat_1, d$pol_uncat_2)
for (c in names(CATS)) d[[paste0("w_", c)]] <- wsum(d[[paste0(c, "_1")]], d[[paste0(c, "_2")]])
d <- d |> mutate(w_uncoded = if_else(coded, 0, w1 + w2), w_nonpol = 1 - w_political) |>
  select(-ends_with("_1"), -ends_with("_2"))
CC_CATS <- paste0("comments_", names(CATS))                   # the comments under a category's posts: weight x num_comments
for (c in names(CATS)) d[[paste0("comments_", c)]] <- d[[paste0("w_", c)]] * coalesce(d$num_comments, 0)
check("weights lie in [0, 1] and political + excluded + uncoded <= 1",
      all(d$w1 + d$w2 <= 1) && all(d$w_political >= 0 & d$w_political <= 1) &&
        all(d$w_political + d$w_excluded + d$w_uncoded <= 1 + 1e-9) && all(d$w_political[!d$coded] == 0))
check("a category weight never exceeds the political weight", all(d[W_CATS] <= d$w_political + 1e-9))
dir.create(dirname(WEIGHTS), recursive = TRUE, showWarnings = FALSE)
write_csv(d |> select(id, period_bin, topic_assignment, w_political, w_nonpol, w_excluded, w_uncoded, w_pol_uncat, all_of(W_CATS)),
          WEIGHTS, na = "")
log_msg("weights written: %s (political weight %s of %s posts, %.1f %%)", WEIGHTS, n_lab(round(sum(d$w_political))), n_lab(nrow(d)),
        100 * sum(d$w_political) / nrow(d))

# ---------------------------------------------------------------------------
# 2. The monthly table behind 1a-1c, 2b, 2f, 3a (a month lies inside one bin: the bins start on the first of a month)
# ---------------------------------------------------------------------------
monthly <- d |> group_by(month) |>
  summarise(posts = n(), posts_with_comments = sum(!is.na(num_comments)), comments = sum(num_comments, na.rm = TRUE),
            mean_comments = mean(num_comments, na.rm = TRUE), median_comments = median(num_comments, na.rm = TRUE),
            q25_comments = quantile(num_comments, 0.25, na.rm = TRUE), q75_comments = quantile(num_comments, 0.75, na.rm = TRUE),
            political = sum(w_political), non_political = sum(w_nonpol), excluded = sum(w_excluded), uncoded = sum(w_uncoded),
            political_uncategorised = sum(w_pol_uncat),
            comments_political = sum(w_political * num_comments, na.rm = TRUE), comments_non_political = sum(w_nonpol * num_comments, na.rm = TRUE),
            across(all_of(W_CATS), sum), across(all_of(CC_CATS), sum), .groups = "drop") |>
  rename_with(~ sub("^w_", "", .x), all_of(W_CATS)) |>
  mutate(period_bin = findInterval(month, BIN_START), coded = period_bin %in% coded_bins,
         share_political = if_else(coded, political / posts, NA_real_), share_non_political = if_else(coded, non_political / posts, NA_real_),
         share_comments_political = if_else(coded, comments_political / comments, NA_real_))
check("the monthly table covers every month from 2010-01 to 2026-07 once", nrow(monthly) == 199 && !anyDuplicated(monthly$month))
check("monthly posts sum to the corpus", sum(monthly$posts) == N_CORPUS)

# 1a-1c. Volume and comments
tab(monthly |> select(month, posts), "1a_monthly_volume")
p <- ggplot(monthly, aes(month, posts)) + bands() + geom_line(colour = SERIES[1], linewidth = 0.6) + month_axis + y_count() +
  labs(title = "Posts per month in r/soccer, 2010–2026", x = NULL, y = "Posts per month") + theme_fifa()
fig(p, "1a_monthly_volume")
tab(monthly |> select(month, posts, posts_with_comments, mean_comments, median_comments, q25_comments, q75_comments), "1b_monthly_comments_per_post")
p <- ggplot(monthly, aes(month, mean_comments)) + bands() + geom_line(colour = SERIES[1], linewidth = 0.6) + month_axis + y_count() +
  labs(title = "Mean comments per post by month, 2010–2026", x = NULL, y = "Mean comments per post") + theme_fifa()
fig(p, "1b_monthly_comments_per_post")
tab(monthly |> select(month, posts, posts_with_comments, comments), "1c_monthly_comments")
p <- ggplot(monthly, aes(month, comments)) + bands() + geom_line(colour = SERIES[1], linewidth = 0.6) + month_axis + y_count() +
  labs(title = "Comments per month in r/soccer, 2010–2026", x = NULL, y = "Comments per month") + theme_fifa()
fig(p, "1c_monthly_comments")

# ---------------------------------------------------------------------------
# 2a / 2c. Political posts and their categories per bin and overall (tables)
# ---------------------------------------------------------------------------
by_bin <- d |> group_by(period_bin) |>
  summarise(posts = n(), political = sum(w_political), political_uncategorised = sum(w_pol_uncat), non_political = sum(w_nonpol),
            excluded = sum(w_excluded), uncoded = sum(w_uncoded), across(all_of(W_CATS), sum), .groups = "drop")
by_bin <- bind_rows(by_bin, by_bin |> summarise(period_bin = NA_integer_, across(-period_bin, sum))) |>
  mutate(period_type = if_else(is.na(period_bin), "all bins", BIN_TYPE[period_bin]), coded = is.na(period_bin) | period_bin %in% coded_bins) |>
  relocate(period_type, .after = period_bin)
shares <- function(x, n, coded) if_else(coded, x / n, NA_real_)
a2 <- by_bin |> mutate(share_political = shares(political, posts, coded), share_political_uncategorised = shares(political_uncategorised, posts, coded),
                       share_non_political = shares(non_political, posts, coded), share_excluded = shares(excluded, posts, coded),
                       share_uncoded = uncoded / posts)
check("2a political + non-political = posts in every row", all(abs(a2$political + a2$non_political - a2$posts) < 1e-6))
tab(a2 |> select(period_bin, period_type, posts, starts_with("share_")), "2a_political_share")
tab(a2 |> select(period_bin, period_type, posts, political, political_uncategorised, non_political, excluded, uncoded), "2a_political_counts", dir = ALT_DIR)

c2 <- by_bin |> select(period_bin, period_type, posts, political, coded, all_of(W_CATS)) |>
  pivot_longer(all_of(W_CATS), names_to = "column", values_to = "category_posts") |>
  mutate(column = sub("^w_", "", column), category = factor(CATS[column], CATS),
         share_of_political = if_else(coded & political > 0, category_posts / political, NA_real_), share_of_all = shares(category_posts, posts, coded)) |>
  arrange(period_bin, category)
tab(c2 |> select(period_bin, period_type, category, share_of_political, share_of_all, political, posts), "2c_political_categories")
tab(c2 |> select(period_bin, period_type, category, category_posts, political, posts), "2c_political_category_counts", dir = ALT_DIR)

# 2d. The categories by World Cup: share of the bin's political posts (E1's right half), and the counts
d2 <- c2 |> filter(period_bin %in% WC$period_bin) |> mutate(bin_lab = cup_lab(period_bin))
check("2d odd-bin category counts are 2c's", identical(d2$category_posts, c2$category_posts[c2$period_bin %in% WC$period_bin]))
tab(d2 |> select(period_bin, bin_lab, category, category_posts, political, posts, share_of_political, share_of_all), "2d_political_categories_by_world_cup")
cat_bars <- function(data, y, title, ylab, labels) {
  ggplot(data, aes(bin_lab, {{ y }}, fill = category)) +
    geom_col(position = position_dodge2(preserve = "single", padding = 0.12), width = 0.82) +
    geom_text(aes(label = labels), position = position_dodge2(width = 0.82, preserve = "single", padding = 0.12), vjust = -0.5, size = 2.3, colour = INK2) +
    scale_fill_manual(values = CAT_COLS) + labs(title = title, x = NULL, y = ylab) + theme_fifa()
}
p <- cat_bars(d2, share_of_political, "Political categories by World Cup, share of the bin's political posts", "Share of political posts",
              sprintf("%.0f %%", 100 * d2$share_of_political)) + y_share(0.12)
fig(p, "2d_political_categories_by_world_cup", w = 10, h = 4.8)
p <- cat_bars(d2, category_posts, "Political posts by category and World Cup", "Political posts", n_lab(round(d2$category_posts))) + y_count(0.12)
fig(p, "2d_political_categories_by_world_cup_counts", w = 10, h = 4.8, dir = ALT_DIR)

# ---------------------------------------------------------------------------
# 2b / 2f / 3a. Monthly series: political vs non-political posts, comments under political posts, the categories
# ---------------------------------------------------------------------------
b2 <- monthly |> select(month, posts, political, non_political, share_political, share_non_political) |>
  pivot_longer(c(political, non_political), names_to = "group", values_to = "n") |>
  mutate(share = if_else(group == "political", share_political, share_non_political),
         group = factor(if_else(group == "political", POL_LAB, NONPOL_LAB), c(POL_LAB, NONPOL_LAB))) |>
  select(month, posts, group, n, share)
check("2b the two shares sum to 1 in every coded month", all(abs(monthly$share_political[monthly$coded] + monthly$share_non_political[monthly$coded] - 1) < 1e-9))
tab(b2, "2b_political_share_monthly")
fig(line_series(b2, share, "Share of political and non-political posts by month, 2010–2026", "Share of the month's posts", POL_COLS, POL_LINES) + y_share(),
         "2b_political_share_monthly")
fig(line_series(b2, n, "Political and non-political posts per month, 2010–2026", "Posts per month", POL_COLS, POL_LINES) + y_count(),
         "2b_political_counts_monthly", dir = ALT_DIR)

f2 <- monthly |> select(month, comments, comments_political, comments_non_political, share_comments_political)
tab(f2, "2f_comments_political_share")
f2_bin <- bind_rows(d |> group_by(period_bin) |> summarise(comments = sum(num_comments, na.rm = TRUE), comments_political = sum(w_political * num_comments, na.rm = TRUE), .groups = "drop"),
                    d |> summarise(period_bin = NA_integer_, comments = sum(num_comments, na.rm = TRUE), comments_political = sum(w_political * num_comments, na.rm = TRUE))) |>
  mutate(period_type = if_else(is.na(period_bin), "all bins", BIN_TYPE[period_bin]), comments_non_political = comments - comments_political,
         share_comments_political = if_else(is.na(period_bin) | period_bin %in% coded_bins, comments_political / comments, NA_real_)) |>
  relocate(period_type, .after = period_bin)
tab(f2_bin, "2f_comments_political_share_by_bin")
p <- ggplot(f2, aes(month, share_comments_political)) + bands() + geom_line(colour = INK2, linewidth = 0.6, na.rm = TRUE) + month_axis + y_share() +
  labs(title = "Share of comments under political posts by month, 2010–2026", x = NULL, y = "Share of the month's comments") + theme_fifa()
fig(p, "2f_comments_political_share")
f2l <- f2 |> pivot_longer(c(comments_political, comments_non_political), names_to = "group", values_to = "n") |>
  mutate(group = factor(if_else(group == "comments_political", POL_LAB, NONPOL_LAB), c(POL_LAB, NONPOL_LAB)))
fig(line_series(f2l, n, "Comments under political and non-political posts by month, 2010–2026", "Comments per month", POL_COLS) + y_count(),
         "2f_comments_political_counts", dir = ALT_DIR)

a3 <- monthly |> select(month, posts, coded, all_of(names(CATS))) |>
  pivot_longer(all_of(names(CATS)), names_to = "column", values_to = "category_posts") |>
  mutate(group = factor(CATS[column], CATS), share = if_else(coded, category_posts / posts, NA_real_)) |>
  select(month, posts, category = group, category_posts, share)
tab(a3, "3a_political_categories_monthly")
fig(line_series(a3 |> rename(group = category), share, "Political categories by month, share of posts, 2010–2026", "Share of the month's posts", CAT_COLS, CAT_LINES, 2) + y_share(),
         "3a_political_categories_monthly")
fig(line_series(a3 |> rename(group = category), category_posts, "Political posts by category and month, 2010–2026", "Posts per month", CAT_COLS, CAT_LINES, 2) + y_count(),
         "3a_political_categories_monthly_counts", dir = ALT_DIR)   # also the counts behind 3d

# 3b. Comments under each category's posts per month, as a share of the month's comments (Alan Nemirovski, 2026-10-31: 2g's
# measure across the whole corpus); the comment counts in the alternatives
b3 <- monthly |> select(month, comments, coded, all_of(paste0("comments_", names(CATS)))) |>
  pivot_longer(all_of(paste0("comments_", names(CATS))), names_to = "column", values_to = "category_comments") |>
  mutate(column = sub("^comments_", "", column), group = factor(CATS[column], CATS),
         share = if_else(coded & comments > 0, category_comments / comments, NA_real_)) |>
  select(month, comments, category = group, category_comments, share)
check("3b a category's comments never exceed the month's comments", all(b3$category_comments <= b3$comments + 1e-6))
tab(b3, "3b_comments_by_political_category_monthly")
fig(line_series(b3 |> rename(group = category), share, "Share of comments by political category and month, 2010–2026", "Share of the month's comments", CAT_COLS, CAT_LINES, 2) + y_share(),
    "3b_comments_by_political_category_monthly")
fig(line_series(b3 |> rename(group = category), category_comments, "Comments under political posts by category and month, 2010–2026", "Comments per month", CAT_COLS, CAT_LINES, 2) + y_count(),
    "3b_comments_by_political_category_monthly_counts", dir = ALT_DIR)

# 2e. The categories day by day through each World Cup (E1b): share of the day's posts over a centred 7-day window
days <- d |> filter(period_bin %in% WC$period_bin) |> inner_join(WC, by = "period_bin") |>
  mutate(day = as.integer(date - kickoff)) |> filter(day >= 0, date <= final)
day_all <- days |> count(cup, day, name = "posts")
day_cat <- days |> group_by(cup, day) |> summarise(across(all_of(W_CATS), sum), .groups = "drop") |>
  pivot_longer(all_of(W_CATS), names_to = "column", values_to = "category_posts") |> mutate(column = sub("^w_", "", column))
e2 <- WC |> transmute(cup, day = Map(seq, 0L, as.integer(final - kickoff))) |> unnest(day) |> crossing(column = names(CATS)) |>
  left_join(day_all, by = c("cup", "day")) |> left_join(day_cat, by = c("cup", "day", "column")) |>
  mutate(posts = coalesce(posts, 0L), category_posts = coalesce(category_posts, 0)) |>
  arrange(cup, column, day) |> group_by(cup, column) |>
  mutate(share_day = category_posts / posts, share_7d = roll_sum(category_posts, ROLL_HALF) / roll_sum(posts, ROLL_HALF),
         category_posts_7d = roll_sum(category_posts, ROLL_HALF)) |> ungroup() |>
  mutate(category = factor(CATS[column], CATS))
check("2e every tournament day has posts", all(e2$posts > 0))
check("2e daily category weights add up to the tournament days' totals", abs(sum(e2$category_posts) - sum(days[W_CATS])) < 1e-6)
tab(e2 |> select(cup, day, category, posts, category_posts, share_day, share_7d, category_posts_7d), "2e_political_categories_daily")
ko <- WC |> transmute(cup, xmin = as.integer(knockout - kickoff), xmax = as.integer(final - kickoff))
daily_plot <- function(data, y, title, ylab) {
  ggplot(data, aes(day, {{ y }}, colour = category, linetype = category)) +
    geom_rect(data = ko, aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf), inherit.aes = FALSE, fill = SHADE) +
    geom_text(data = ko, aes(x = (xmin + xmax) / 2, y = Inf, label = "Knockout"), inherit.aes = FALSE, vjust = 1.5, size = 2.3, colour = MUTED) +
    geom_line(linewidth = 0.7) + facet_wrap(~ cup, nrow = 1) +
    scale_colour_manual(values = CAT_COLS) + scale_linetype_manual(values = CAT_LINES) +
    scale_x_continuous(breaks = c(0, 15, 30), limits = c(0, max(data$day)), expand = expansion(add = 1)) +
    labs(title = title, x = "Days from the opening match", y = ylab) +
    theme_fifa() + theme(legend.key.width = unit(2.2, "lines"), panel.spacing.x = unit(1.1, "lines"),
                         strip.text = element_text(colour = INK2, size = 10, face = "bold", hjust = 0.5))
}
fig(daily_plot(e2, share_7d, "Political categories through each World Cup", "Share of the day's posts") + y_share(0.12), "2e_political_categories_daily", w = 11, h = 4)
fig(daily_plot(e2, category_posts_7d, "Political posts by category through each World Cup", "Posts, 7-day window") + y_count(0.12),
         "2e_political_categories_daily_counts", w = 11, h = 4, dir = ALT_DIR)

# 2g. Comments under each category's posts through each World Cup (Alan Nemirovski, 2026-10-31, reframed the same day): the
# category's comments (post weight x num_comments) as a share of all comments of the day, both summed over the centred 7-day
# window; the share of the comments under political posts stays in the table; the counts version in the alternatives.  A
# post's comments count fully in each of its categories (Q8).
day_pol_c <- days |> group_by(cup, day) |>
  summarise(comments_all = sum(num_comments, na.rm = TRUE), comments_political = sum(w_political * num_comments, na.rm = TRUE), .groups = "drop")
day_cat_c <- days |> group_by(cup, day) |> summarise(across(all_of(CC_CATS), sum), .groups = "drop") |>
  pivot_longer(all_of(CC_CATS), names_to = "column", values_to = "category_comments") |> mutate(column = sub("^comments_", "", column))
g2 <- WC |> transmute(cup, day = Map(seq, 0L, as.integer(final - kickoff))) |> unnest(day) |> crossing(column = names(CATS)) |>
  left_join(day_pol_c, by = c("cup", "day")) |> left_join(day_cat_c, by = c("cup", "day", "column")) |>
  mutate(comments_all = coalesce(comments_all, 0), comments_political = coalesce(comments_political, 0), category_comments = coalesce(category_comments, 0)) |>
  arrange(cup, column, day) |> group_by(cup, column) |>
  mutate(share_day = if_else(comments_all > 0, category_comments / comments_all, NA_real_),
         comments_all_7d = roll_sum(comments_all, ROLL_HALF), comments_political_7d = roll_sum(comments_political, ROLL_HALF),
         category_comments_7d = roll_sum(category_comments, ROLL_HALF),
         share_7d = if_else(comments_all_7d > 0, category_comments_7d / comments_all_7d, NA_real_),
         share_of_political_7d = if_else(comments_political_7d > 0, category_comments_7d / comments_political_7d, NA_real_)) |> ungroup() |>
  mutate(category = factor(CATS[column], CATS))
check("2g every tournament day has comments over the 7-day window", all(g2$comments_all_7d > 0 & g2$comments_political_7d > 0))
check("2g a category's comments never exceed the day's comments under political posts", all(g2$category_comments <= g2$comments_political + 1e-6))
check("2g the categories' comments add up to 2f's political comments of the tournament days or more (Q8)",
      sum(g2$category_comments) >= sum(day_pol_c$comments_political) - 1e-6)
tab(g2 |> select(cup, day, category, comments_all, comments_political, category_comments, share_day, comments_all_7d, comments_political_7d,
                      category_comments_7d, share_7d, share_of_political_7d),
         "2g_comments_by_political_category_daily")
fig(daily_plot(g2, share_7d, "Share of comments by political category through each World Cup", "Share of the day's comments") + y_share(0.12),
         "2g_comments_by_political_category_daily", w = 11, h = 4)
fig(daily_plot(g2, category_comments_7d, "Comments under political posts by category through each World Cup", "Comments, 7-day window") + y_count(0.12),
         "2g_comments_by_political_category_daily_counts", w = 11, h = 4, dir = ALT_DIR)

# ---------------------------------------------------------------------------
# 3c / 3d. Sentimental valence (VADER title score) by category: per World Cup, and per month
# ---------------------------------------------------------------------------
group_stats <- function(dd, by) {                  # weighted mean VADER per group (the four categories + non-political) and `by`
  parts <- lapply(c(names(CATS), "nonpol"), function(g) {
    dd |> mutate(.w = if (g == "nonpol") w_nonpol else .data[[paste0("w_", g)]]) |> filter(.w > 0) |>
      group_by(across(all_of(by))) |> summarise(wstats(post_sentiment, .w), .groups = "drop") |>
      mutate(group = if (g == "nonpol") NONPOL_LAB else CATS[[g]])
  })
  bind_rows(parts) |> mutate(group = factor(group, c(NONPOL_LAB, CATS)), lo95 = mean - 1.96 * se, hi95 = mean + 1.96 * se)
}
GRP_COLS <- c(setNames(MUTED, NONPOL_LAB), CAT_COLS)
b3 <- group_stats(d |> filter(period_bin %in% WC$period_bin) |> mutate(cup = cup_lab(period_bin)), "cup") |> arrange(cup, group)
check("3c category weights are 2d's", all(abs(b3 |> filter(group != NONPOL_LAB) |> arrange(cup, group) |> pull(w_posts) -
                                             d2 |> arrange(bin_lab, category) |> pull(category_posts)) < 1e-6))
tab(b3 |> select(cup, group, w_posts, mean, se, lo95, hi95), "3c_vader_by_political_category")
dodge <- position_dodge(width = 0.72)
p <- ggplot(b3, aes(cup, mean, colour = group, group = group)) +
  geom_hline(yintercept = 0, colour = AXIS, linewidth = 0.4) +
  geom_errorbar(aes(ymin = lo95, ymax = hi95), width = 0, linewidth = 0.6, position = dodge) +
  geom_point(size = 2.6, colour = SURFACE, position = dodge) + geom_point(size = 2, position = dodge) +
  scale_colour_manual(values = GRP_COLS) + guides(colour = guide_legend(nrow = 1)) +
  labs(title = "Sentimental valence of political posts by category and World Cup", x = NULL, y = "Mean VADER score") + theme_fifa()
fig(p, "3c_vader_by_political_category", w = 10, h = 4.8)
p <- ggplot(b3, aes(cup, w_posts, fill = group)) + geom_col(position = position_dodge2(preserve = "single", padding = 0.12), width = 0.82) +
  scale_fill_manual(values = GRP_COLS) + guides(fill = guide_legend(nrow = 1)) +
  scale_y_log10(labels = label_comma(), expand = expansion(mult = c(0, 0.08))) +
  labs(title = "Posts behind the valence estimates by group and World Cup", x = NULL, y = "Posts (log scale)") + theme_fifa()
fig(p, "3c_posts_by_political_category", w = 10, h = 4.8, dir = ALT_DIR)

c3 <- group_stats(d |> filter(coded), "month") |> arrange(month, group)
tab(c3 |> select(month, group, w_posts, mean, se, lo95, hi95), "3d_vader_by_category_monthly")
p <- ggplot(c3, aes(month, mean, colour = group, fill = group, linetype = group)) + bands() +
  geom_hline(yintercept = 0, colour = AXIS, linewidth = 0.4) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.12, colour = NA, na.rm = TRUE) + geom_line(linewidth = 0.6, na.rm = TRUE) +
  scale_colour_manual(values = GRP_COLS) + scale_fill_manual(values = GRP_COLS) +
  scale_linetype_manual(values = c(setNames("solid", NONPOL_LAB), CAT_LINES)) + month_axis +
  guides(colour = guide_legend(nrow = 1), fill = guide_legend(nrow = 1), linetype = guide_legend(nrow = 1)) +
  labs(title = "Sentimental valence by political category and month, 2010–2026", x = NULL, y = "Mean VADER score") +
  theme_fifa() + theme(legend.key.width = unit(2.2, "lines"))
fig(p, "3d_vader_by_category_monthly")

# ---------------------------------------------------------------------------
# 3e. Concentration of political posts among users (authors other than [deleted]), the non-political posts as the
#     reference curve on the same panel, the Gini of each curve in the corner (Alan Nemirovski, 2026-10-31): the whole
#     corpus and one panel per tournament bin
# ---------------------------------------------------------------------------
concentration <- function(dd, w, label, scope) {  # w: the posts' weight in the group; users ranked by their weight, descending
  pol <- dd[w > 0, ] |> mutate(.w = w[w > 0])
  total_w <- sum(pol$.w); deleted_w <- sum(pol$.w[pol$author == "[deleted]"])
  users <- pol |> filter(author != "[deleted]") |> group_by(author) |> summarise(w = sum(.w), .groups = "drop") |>
    arrange(desc(w), author) |> mutate(rank = row_number(), cum_w = cumsum(w), share_users = rank / n(), share_posts = cum_w / sum(w))
  xs <- sort(users$w); n_u <- length(xs)
  gini <- (2 * sum(seq_len(n_u) * xs)) / (n_u * sum(xs)) - (n_u + 1) / n_u
  top_share <- function(q) users$share_posts[max(1, ceiling(q * n_u))]
  attributed_w <- sum(users$w); max_w <- max(users$w); one_post <- sum(users$w <= 1)   # before the tibble: `users` becomes a column there
  stats <- tibble(scope = scope, group = label, users = n_u, weight_attributed = attributed_w, weight_total = total_w,
                  deleted_share = deleted_w / total_w, top1_share = top_share(0.01), top5_share = top_share(0.05),
                  top10_share = top_share(0.10), gini = gini, max_posts_one_user = max_w, users_with_one_post = one_post)
  idx <- unique(c(1L, round(seq(1, n_u, length.out = 2000L)), n_u))
  list(stats = stats, curve = users[idx, ] |> transmute(scope = scope, group = label, rank, share_users, cum_w, share_posts))
}
both_groups <- function(dd, scope) {
  a <- concentration(dd, dd$w_political, POL_LAB, scope); b <- concentration(dd, dd$w_nonpol, NONPOL_LAB, scope)
  list(stats = bind_rows(a$stats, b$stats), curve = bind_rows(a$curve, b$curve))
}
SCOPE_CORPUS <- "whole corpus"               # the five tournament bins pooled was a third version until 2026-10-31 (removed at Alan's request)
scopes <- c(list(both_groups(d, SCOPE_CORPUS)),
            lapply(WC$period_bin, function(b) both_groups(d |> filter(period_bin == b), as.character(cup_lab(b)))))
SCOPE_LEVELS <- c(SCOPE_CORPUS, levels(cup_lab(1L)))
d3 <- bind_rows(lapply(scopes, `[[`, "stats")) |> mutate(scope = factor(scope, SCOPE_LEVELS), group = factor(group, c(POL_LAB, NONPOL_LAB)))
curves <- bind_rows(lapply(scopes, `[[`, "curve")) |> mutate(scope = factor(scope, SCOPE_LEVELS), group = factor(group, c(POL_LAB, NONPOL_LAB)))
check("3e every curve ends at 100 % of users and posts, Gini in [0, 1]",
      all(abs(curves |> group_by(scope, group) |> summarise(u = last(share_users), p = last(share_posts), .groups = "drop") |> select(u, p) - 1) < 1e-9) &&
        all(d3$gini >= 0 & d3$gini <= 1))
tab(d3, "3e_political_user_concentration")
tab(curves, "3e_political_user_concentration_curve")
gini_labels <- function(st) st |> group_by(scope) |>
  summarise(label = sprintf("Gini\nPolitical %.2f\nNon-political %.2f", gini[group == POL_LAB], gini[group == NONPOL_LAB]), .groups = "drop")
conc_plot <- function(sc, title) {               # both curves, the equality line clipped to the unit square, the Ginis in the corner
  cv <- curves |> filter(scope %in% sc); st <- d3 |> filter(scope %in% sc)
  ggplot(cv, aes(share_users, share_posts, colour = group, linetype = group)) +
    annotate("segment", x = 0, y = 0, xend = 1, yend = 1, colour = AXIS, linetype = "dashed", linewidth = 0.4) +
    geom_line(linewidth = 0.7) +
    geom_text(data = gini_labels(st), aes(x = 0.97, y = 0.03, label = label), inherit.aes = FALSE, hjust = 1, vjust = 0,
              size = 2.5, colour = INK2, lineheight = 0.95) +
    scale_colour_manual(values = POL_COLS) + scale_linetype_manual(values = POL_LINES) +
    scale_x_continuous(labels = pct, limits = c(0, 1), expand = expansion(mult = 0.01)) +
    scale_y_continuous(labels = pct, limits = c(0, 1), expand = expansion(mult = c(0, 0.04))) +
    labs(title = title, x = "Share of users, ranked by posts in the group", y = "Share of the group's posts") +
    theme_fifa() + theme(legend.key.width = unit(2.2, "lines"))
}
fig(conc_plot(SCOPE_CORPUS, "Concentration of political and non-political posts among users"), "3e_political_user_concentration", w = 6.5, h = 5.4)
p <- conc_plot(levels(cup_lab(1L)), "Concentration of political and non-political posts among users by World Cup") +
  facet_wrap(~ scope, nrow = 1) +
  theme(panel.spacing.x = unit(1.1, "lines"), strip.text = element_text(colour = INK2, size = 10, face = "bold", hjust = 0.5))
fig(p, "3e_political_user_concentration_by_tournament", w = 12, h = 3.9)
p <- ggplot(curves |> filter(scope == SCOPE_CORPUS, group == POL_LAB), aes(rank, cum_w)) + geom_line(colour = INK2, linewidth = 0.7) +
  scale_x_continuous(labels = label_comma(), expand = expansion(mult = 0.01)) + y_count(0.04) +
  labs(title = "Cumulative political posts by user rank", x = "Users, ranked by political posts", y = "Cumulative political posts") + theme_fifa()
fig(p, "3e_political_user_concentration_counts", w = 6, h = 5.2, dir = ALT_DIR)

# ---------------------------------------------------------------------------
# 4. Gate
# ---------------------------------------------------------------------------
MAIN <- c("fig_3b_comments_by_political_category_monthly", "fig_3e_political_user_concentration_by_tournament",
          "fig_1a_monthly_volume", "fig_1b_monthly_comments_per_post", "fig_1c_monthly_comments", "tab_2a_political_share",
          "fig_2b_political_share_monthly", "tab_2c_political_categories", "fig_2d_political_categories_by_world_cup",
          "fig_2e_political_categories_daily", "fig_2g_comments_by_political_category_daily", "fig_2f_comments_political_share", "fig_3a_political_categories_monthly",
          "fig_3c_vader_by_political_category", "fig_3d_vader_by_category_monthly", "fig_3e_political_user_concentration")
ALT <- c("fig_3b_comments_by_political_category_monthly_counts", "tab_2a_political_counts", "fig_2b_political_counts_monthly", "tab_2c_political_category_counts",
         "fig_2d_political_categories_by_world_cup_counts", "fig_2e_political_categories_daily_counts", "fig_2g_comments_by_political_category_daily_counts", "fig_2f_comments_political_counts",
         "fig_3a_political_categories_monthly_counts", "fig_3c_posts_by_political_category", "fig_3e_political_user_concentration_counts")
exists_out <- function(dir, id) file.exists(if (startsWith(id, "fig_")) file.path(dir, paste0(id, ".png")) else file.path(dir, "tabs", paste0(id, ".csv")))
check("every main output exists", all(vapply(MAIN, exists_out, logical(1), dir = OUT_DIR)))
check("every alternative output exists", all(vapply(ALT, exists_out, logical(1), dir = ALT_DIR)))
if (!all(unlist(results))) stop("a check failed: ", paste(names(results)[!unlist(results)], collapse = ", "))
log_msg("figures 1a-3e and their tables written to %s (alternatives in %s)", OUT_DIR, ALT_DIR)
cat("ALL CHECKS PASSED\n")
