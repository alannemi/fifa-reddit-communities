##############################################################################
# FIFA COMMUNITIES CHAPTER
## 14. WORLD CUP FIGURES  (political posts, sentiment, subscribers and tones in the five World Cup bins)
##############################################################################
#
# PURPOSE
# -------
# Four figures requested by Alan Nemirovski on 2026-10-28 (decisions Q7-Q10 in
# docs/session-10_2026-10-28/00_approvals.md), each with the table behind it:
#   E1  political posts in each World Cup bin: their share of all the bin's
#       posts, and the share of the political posts in each of the four
#       political categories (national identity, FIFA body, host country
#       domestic, IR)
#   E2  mean VADER title score per week around each World Cup, aligned on the
#       kick-off
#   E3  r/soccer subscribers per month, 2018-2026
#   E4  rank of the topic tones by share of posts within each World Cup bin
#
# POPULATIONS AND RULES
# ---------------------
# * "Political" = a topic whose QUALTAG coding gives qt_political = "yes"
#   (code/13-recode-qualtag.R; the co-authors' files in results/analysis/manual
#   review/).  REVIEW topics and the topics TP has not yet settled
#   (QUALTAG_to_settle.csv) are not political here; the table counts them.
# * E1: political posts = posts in their HDBSCAN cluster or singly reassigned
#   (topic_assignment cluster / reassigned) whose topic is political; the share
#   is over ALL posts of the bin (Q7, option b).  The table also gives the share
#   over the cluster + reassigned posts (option a) and the share counting a dual
#   post as political when either of its two topics is (option c).  A topic with
#   several political tags counts in each of its categories (Q8), so the
#   category shares can add up to more than 100 %.
# * E2: every post of the World Cup bins (as C1); whole weeks only.
# * E3: the monthly median of subreddit_subscribers, which the archive recorded
#   when it fetched the post (crawl time), present from March 2018 only (Q10).
# * E4: the topic tones (topic_tone) of cluster + reassigned posts (as C3/C4,
#   E3 of session 09; Q9).
#
# HOW TO RUN (repository root; reads the 17 yearly files, data/analysis/post_variables.csv
# and the bin files of results/analysis/manual review/ after 13-recode-qualtag.R)
# ----------------------------
#   Rscript code/14-worldcup-figures.R          # ~1 min
#
# OUTPUTS  (results/descriptives/)
# -------
#   fig_E1_political_posts.png|pdf, fig_E2_worldcup_weekly_vader, fig_E3_monthly_subscribers,
#   fig_E4_worldcup_tone_ranks; tab_E1_political_posts.csv, tab_E1_political_categories.csv,
#   tab_E2_worldcup_weekly_vader.csv, tab_E3_monthly_subscribers.csv, tab_E4_worldcup_tone_ranks.csv
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2); library(scales) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                                     # shared helpers and the bin table (code/R/pipeline-utils.R)
source(file.path(dirname(.utils), "plot-utils.R")) # palette, theme, save_fig / save_tab
ROOT <- find_root()
setwd(ROOT)

YEARLY     <- "data/yearly datasets"
PV         <- "data/analysis/post_variables.csv"
REVIEW_DIR <- "results/analysis/manual review"
OUT_DIR    <- "results/descriptives"
N_CORPUS   <- 1302021L
TOP_TONES  <- 10L                  # tones shown per World Cup panel in E4 (the table has all)
# Kick-off and final of each World Cup (mirrors config.WORLD_CUPS and 11-descriptives.R)
WC <- tibble(period_bin = c(1L, 3L, 5L, 7L, 9L), cup = c("2010", "2014", "2018", "2022", "2026"),
             kickoff = as.Date(c("2010-06-11", "2014-06-12", "2018-06-14", "2022-11-20", "2026-06-11")),
             final   = as.Date(c("2010-07-11", "2014-07-13", "2018-07-15", "2022-12-18", "2026-07-19")))
CUP_COLS <- setNames(SERIES[1:5], WC$cup)          # one fixed colour per World Cup, in order
POL_CATS <- c(qt_national_identity = "National identity", qt_fifa_body = "FIFA body",
              qt_host_country_domestic = "Host country domestic", qt_ir = "IR")
n_lab <- function(n) format(n, big.mark = ",")
cup_lab <- function(b) paste0("WC ", WC$cup[match(b, WC$period_bin)], "\n(bin ", b, ")")

# ---------------------------------------------------------------------------
# 1. Data: dates and subscribers from the yearly files, analysis variables from the join table,
#    the QUALTAG variables from the co-authors' bin files
# ---------------------------------------------------------------------------
files <- list.files(YEARLY, pattern = "^soccer\\.data\\.cleaned-\\d{4}\\.csv$", full.names = TRUE)
posts <- bind_rows(lapply(files, read_chr, col_select = c("id", "created_datetime", "subreddit_subscribers")))
pv <- read_chr(PV, col_select = c("id", "period_bin", "topic_uid", "topic_uid_2", "topic_assignment", "topic_tone", "post_sentiment"))
d <- posts |> inner_join(pv, by = "id") |>
  mutate(date = as.Date(substr(created_datetime, 1, 10)), month = as.Date(paste0(substr(created_datetime, 1, 7), "-01")),
         period_bin = as.integer(period_bin), post_sentiment = as.numeric(post_sentiment),
         subscribers = suppressWarnings(as.numeric(subreddit_subscribers)))
rm(posts, pv); invisible(gc())
check("every post of the corpus joined once", nrow(d) == N_CORPUS && !anyDuplicated(d$id), n_lab(nrow(d)))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

qt_files <- list.files(REVIEW_DIR, pattern = "^BIN\\d_topics_labelled_.*\\.csv$", full.names = TRUE)
qt <- bind_rows(lapply(qt_files, function(p)
  read_chr(p, col_select = c("topic_id", "qt_political", all_of(names(POL_CATS)))) |>
    mutate(period_bin = as.integer(substr(basename(p), 4, 4))))) |>
  mutate(topic_uid = paste(period_bin, topic_id, sep = "-"))
check("the QUALTAG files cover the five World Cup bins, one row per topic",
      setequal(unique(qt$period_bin), WC$period_bin) && !anyDuplicated(qt$topic_uid), sprintf("%d topics", nrow(qt)))
wc <- d |> filter(period_bin %in% WC$period_bin)
check("every topic of the World Cup bins has a QUALTAG row",
      all(setdiff(unique(c(wc$topic_uid, wc$topic_uid_2)), "") %in% qt$topic_uid))

# ---------------------------------------------------------------------------
# E1. Political posts: share of the bin, and the four political categories
# ---------------------------------------------------------------------------
pol_uid <- qt$topic_uid[qt$qt_political == "yes"]
status  <- setNames(qt$qt_political, qt$topic_uid)
tp_wc   <- wc |> filter(topic_assignment %in% c("cluster", "reassigned"))
e1 <- wc |> group_by(period_bin) |>
  summarise(posts_bin = n(),
            posts_cluster_reassigned = sum(topic_assignment %in% c("cluster", "reassigned")),
            political_posts = sum(topic_assignment %in% c("cluster", "reassigned") & topic_uid %in% pol_uid),
            review_posts    = sum(topic_assignment %in% c("cluster", "reassigned") & status[topic_uid] %in% "review"),
            unsettled_posts = sum(topic_assignment %in% c("cluster", "reassigned") & status[topic_uid] %in% ""),
            political_posts_incl_dual = sum(topic_uid %in% pol_uid | (topic_assignment == "dual" & topic_uid_2 %in% pol_uid)),
            .groups = "drop") |>
  mutate(share_of_bin = political_posts / posts_bin,                                # Q7 (b): the figure
         share_of_cluster_reassigned = political_posts / posts_cluster_reassigned, # (a)
         share_incl_dual = political_posts_incl_dual / posts_bin,                   # (c)
         review_share_of_bin = review_posts / posts_bin, unsettled_share_of_bin = unsettled_posts / posts_bin) |>
  left_join(WC |> select(period_bin, cup), by = "period_bin") |> relocate(cup, .after = period_bin)
check("E1 political posts lie inside the cluster + reassigned posts", all(e1$political_posts <= e1$posts_cluster_reassigned))
save_tab(e1, "E1_political_posts")

e1c <- tp_wc |> filter(topic_uid %in% pol_uid) |> left_join(qt |> select(topic_uid, all_of(names(POL_CATS))), by = "topic_uid") |>
  pivot_longer(all_of(names(POL_CATS)), names_to = "column", values_to = "tag") |>
  group_by(period_bin, column) |> summarise(posts = sum(tag == "1"), .groups = "drop") |>
  left_join(e1 |> select(period_bin, cup, political_posts, posts_bin), by = "period_bin") |>
  mutate(category = factor(POL_CATS[column], POL_CATS), share_of_political = posts / political_posts, share_of_bin = posts / posts_bin) |>
  arrange(period_bin, category)
check("E1 every political post carries at least one political category",
      all(tp_wc |> filter(topic_uid %in% pol_uid) |> left_join(qt, by = "topic_uid") |>
            transmute(k = rowSums(across(all_of(names(POL_CATS)), ~ .x == "1"))) |> pull(k) >= 1))
save_tab(e1c |> select(period_bin, cup, category, column, posts, political_posts, share_of_political, share_of_bin, posts_bin), "E1_political_categories")

panel_lab <- c(total = "Political posts, share of all posts in the bin", cats = "Political categories, share of the bin's political posts")
e1p <- bind_rows(e1 |> transmute(period_bin, panel = "total", group = "All political posts", share = share_of_bin, n = political_posts),
                 e1c |> transmute(period_bin, panel = "cats", group = as.character(category), share = share_of_political, n = posts)) |>
  mutate(panel = factor(panel_lab[panel], panel_lab), group = factor(group, c("All political posts", POL_CATS)),
         bin_lab = factor(cup_lab(period_bin), cup_lab(WC$period_bin)))
fills <- c("All political posts" = INK2, setNames(SERIES[1:4], POL_CATS))
p <- ggplot(e1p, aes(bin_lab, share, fill = group)) +
  geom_col(position = position_dodge2(preserve = "single", padding = 0.12), width = 0.82) +
  geom_text(data = filter(e1p, group == "All political posts"), aes(label = sprintf("%.1f %%", 100 * share)),
            vjust = -0.5, size = 2.7, colour = INK2) +
  facet_wrap(~ panel, nrow = 1, scales = "free_y") +
  scale_fill_manual(values = fills, breaks = unname(POL_CATS)) +     # unnamed: a named `breaks` would label the legend with the names
  scale_y_continuous(labels = label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.1))) +
  labs(title = "Political posts in the five World Cup period bins",
       subtitle = "Left: posts in topics coded political, as a share of all posts in the bin. Right: share of those posts whose topic carries each political category",
       x = NULL, y = NULL,
       caption = paste0("Political = topics coded political in the co-authors' QUALTAG review (qt_political = yes; REVIEW and not-yet-settled topics excluded). Counted: posts in their\n",
                        "HDBSCAN cluster or singly reassigned to a political topic (", n_lab(sum(e1$political_posts)), " posts); dual-topic and outlier posts are not counted as political. A topic can carry\n",
                        "more than one category, so the right-hand shares of a bin can add up to more than 100 %. Other denominators and the REVIEW posts: tab_E1_political_posts.csv.")) +
  theme_fifa()
save_fig(p, "E1_political_posts", w = 10, h = 4.8)

# ---------------------------------------------------------------------------
# E2. Mean VADER title score per week around each World Cup (all posts of the bins, whole weeks)
# ---------------------------------------------------------------------------
e2 <- wc |> inner_join(WC, by = "period_bin") |>
  mutate(week = floor(as.numeric(date - kickoff) / 7)) |>
  filter(kickoff + 7 * week >= BIN_START[period_bin], kickoff + 7 * week + 6 <= BIN_END[period_bin]) |>   # whole weeks only
  group_by(cup, week) |>
  summarise(posts = n(), mean = mean(post_sentiment), se = sd(post_sentiment) / sqrt(n()), .groups = "drop") |>
  mutate(lo95 = mean - 1.96 * se, hi95 = mean + 1.96 * se)
save_tab(e2, "E2_worldcup_weekly_vader")
final_weeks <- floor(as.numeric(WC$final - WC$kickoff) / 7)
p <- ggplot(e2, aes(week, mean, colour = cup, fill = cup)) +
  annotate("rect", xmin = 0, xmax = max(final_weeks) + 1, ymin = -Inf, ymax = Inf, fill = SHADE) +
  geom_hline(yintercept = 0, colour = AXIS, linewidth = 0.4) +
  geom_vline(xintercept = 0, colour = AXIS, linewidth = 0.3) +
  geom_ribbon(aes(ymin = lo95, ymax = hi95), alpha = 0.10, colour = NA) +
  geom_line(linewidth = 0.6) +
  scale_colour_manual(values = CUP_COLS, labels = paste("World Cup", WC$cup), aesthetics = c("colour", "fill")) +
  scale_x_continuous(breaks = seq(-24, 16, 4)) +
  labs(title = "Sentimental valence of post titles around each World Cup",
       subtitle = "Weekly mean VADER compound score of the title (−1 to 1), with its 95 % confidence interval; weeks relative to the opening match (week 0)",
       x = "Weeks from kick-off", y = "Mean VADER score",
       caption = sprintf(paste0("All %s posts of the five World Cup period bins (post_sentiment), whole weeks only. Shaded: the tournaments, kick-off to final (weeks 0–%d).\n",
                                "The valence of the title's wording, not the author's sentiment (validation against one human coder: 56 %% agreement, κ = 0.24; codebook §8.2)."),
                         n_lab(sum(e2$posts)), max(final_weeks))) +
  theme_fifa()
save_fig(p, "E2_worldcup_weekly_vader")

# ---------------------------------------------------------------------------
# E3. Subscribers per month (crawl-time values, from March 2018)
# ---------------------------------------------------------------------------
e3 <- d |> filter(!is.na(subscribers)) |> group_by(month) |>
  summarise(posts_with_value = n(), median = median(subscribers), min = min(subscribers), max = max(subscribers), .groups = "drop") |>
  left_join(d |> count(month, name = "posts"), by = "month") |> relocate(posts, .after = month)
check("E3 subscribers recorded from 2018 on only", min(e3$month) >= as.Date("2018-01-01"), format(min(e3$month)))
save_tab(e3, "E3_monthly_subscribers")
wc_in <- tibble(xmin = BIN_START[c(5, 7, 9)], xmax = BIN_END[c(5, 7, 9)] + 1, cup = WC$cup[3:5]) |>
  mutate(xmin = pmax(xmin, min(e3$month)))
p <- ggplot(e3, aes(month, median / 1e6)) +
  geom_rect(data = wc_in, aes(xmin = xmin, xmax = xmax, ymin = -Inf, ymax = Inf), inherit.aes = FALSE, fill = SHADE) +
  geom_text(data = wc_in, aes(x = if_else(cup == "2026", xmax, xmin + (xmax - xmin) / 2), y = Inf,
                              hjust = if_else(cup == "2026", 1, 0.5), label = paste("WC", cup)), inherit.aes = FALSE,
            vjust = 1.4, size = 2.4, colour = MUTED) +
  geom_line(colour = SERIES[1], linewidth = 0.6) +
  scale_x_date(date_breaks = "1 year", date_labels = "%Y", expand = expansion(mult = 0.01)) +
  scale_y_continuous(breaks = seq(0, 10, 2), labels = label_number(accuracy = 1, suffix = " M"), limits = c(0, NA),
                     expand = expansion(mult = c(0, 0.08))) +                    # whole-million breaks: the labels round
  labs(title = "Subscribers to r/soccer, 2018–2026",
       subtitle = "Monthly median of the subscriber count recorded with each post; shaded: World Cup period bins",
       x = NULL, y = "Subscribers",
       caption = sprintf(paste0("subreddit_subscribers, recorded by the Arctic Shift archive when it fetched the post (crawl time), not when the post was made; present from %s only (no values\n",
                                "for 2010–2017). Until mid-2023 the archive fetched posts weeks to months late, so the series moves in steps and lags the real count (codebook limitation 11)."),
                         format(min(e3$month), "%B %Y"))) +
  theme_fifa()
save_fig(p, "E3_monthly_subscribers")

# ---------------------------------------------------------------------------
# E4. Tone ranks within each World Cup bin (cluster + reassigned posts)
# ---------------------------------------------------------------------------
e4 <- tp_wc |> count(period_bin, topic_tone, name = "posts") |> group_by(period_bin) |>
  mutate(share = posts / sum(posts), rank = rank(-share, ties.method = "first")) |> ungroup() |>
  left_join(WC |> select(period_bin, cup), by = "period_bin") |> arrange(period_bin, rank)
check("E4 every cluster + reassigned post has a tone; shares sum to 1 per bin",
      all(tp_wc$topic_tone != "") && all(abs(tapply(e4$share, e4$period_bin, sum) - 1) < 1e-9))
save_tab(e4 |> select(period_bin, cup, rank, topic_tone, posts, share), "E4_worldcup_tone_ranks")
e4p <- e4 |> filter(rank <= TOP_TONES) |>
  mutate(bin_lab = factor(paste0("World Cup ", cup, " (bin ", period_bin, ")"), paste0("World Cup ", WC$cup, " (bin ", WC$period_bin, ")")),
         key = factor(paste(period_bin, rank, sep = "_"), rev(paste(rep(WC$period_bin, each = TOP_TONES), 1:TOP_TONES, sep = "_"))),
         lab = paste0(rank, ". ", topic_tone))
p <- ggplot(e4p, aes(share, key)) +
  geom_col(fill = SERIES[1], width = 0.72) +
  geom_text(aes(label = sprintf("%.1f %%", 100 * share)), hjust = -0.15, size = 2.5, colour = INK2) +
  facet_wrap(~ bin_lab, nrow = 1, scales = "free_y") +
  scale_y_discrete(labels = setNames(e4p$lab, as.character(e4p$key))) +
  scale_x_continuous(labels = label_percent(accuracy = 1), expand = expansion(mult = c(0, 0.32)), n.breaks = 3) +
  labs(title = "Which tones lead in each World Cup period",
       subtitle = sprintf("The %d most frequent topic tones in each World Cup bin, ranked by share of posts", TOP_TONES),
       x = "Share of posts", y = NULL,
       caption = sprintf(paste0("Posts in their HDBSCAN cluster or singly reassigned to a topic (%s; dual-topic and outlier posts excluded), with the tone the two analysts gave their topic\n",
                                "(controlled vocabulary incl. the four categories of 2026-10-27; 'other' = tones outside it). Every tone and rank: tab_E4_worldcup_tone_ranks.csv; all nine bins: figure C4."),
                         n_lab(nrow(tp_wc)))) +
  theme_fifa() + theme(panel.grid.major.y = element_blank(), panel.grid.major.x = element_line(colour = GRID, linewidth = 0.3),
                       axis.line.x = element_blank(), axis.text.y = element_text(colour = INK2, size = 7.5),
                       strip.text = element_text(size = 8.5))
save_fig(p, "E4_worldcup_tone_ranks", w = 12, h = 4.2)

if (!all(unlist(results))) stop("a check failed: ", paste(names(results)[!unlist(results)], collapse = ", "))
log_msg("figures E1-E4 and their tables written to %s", OUT_DIR)
