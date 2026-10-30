##############################################################################
# FIFA COMMUNITIES CHAPTER
## 16. REFERENCE MAP  (the period topics mapped onto the whole-corpus reference topics)
##############################################################################
#
# PURPOSE
# -------
# The whole-corpus reference model (session 12, 2026-10-30; results/analysis/reference/,
# bin 0 of the pipeline) gives every post a corpus-wide topic beside the period
# topic it carries in data/analysis/post_variables.csv.  This script maps the
# 6,245 period topics onto the reference topics by post overlap, checks the
# mapping against the similarity of the topic centroids, and writes for every
# reference topic the composition of its posts by the co-authors' QUALTAG coding
# of the period topics (TP: bins 1, 3, 5, 7, 9 through the QUALTAG scheme; BA:
# bins 2 and 4, qt_ columns typed in directly).  These tables are the evidence
# from which the transfer of the QUALTAG attributes onto the reference topics
# will be decided; no tag is written here (decision W2, Alan Nemirovski,
# 2026-10-30, docs/session-12_2026-10-30/00_approvals.md).
#
# METHOD
# ------
# * A post counts on either side only when it is counted there (decision K1:
#   topic_assignment `cluster` or `reassigned`); dual, outlier and not-modelled
#   posts are reported as uncounted shares, never silently dropped.
# * map_pairs: every (period topic, reference topic) pair with posts counted on
#   both sides; the shares are relative to each topic's n_total (its counted posts).
# * map_period_topics: per period topic the reference topic holding most of its
#   counted posts (best), the second, the reference topic nearest by centroid
#   cosine (the mean mpnet embedding of the topic's posts: bin-N/topic_embeddings.npy
#   and reference/topic_embeddings.npy, one embedding space, as in 12-topic-network.R)
#   and whether the two methods agree.
# * reference_topics: per reference topic its size, its posts per period bin, the
#   three largest period topics by overlap, and the QUALTAG composition of its
#   counted posts (counts and shares) by the status of their period topic.
# * Figure G1: the best-match share by period bin — how cleanly the period topics
#   fall into reference topics.  The reference run's `period_bin` column is each
#   post's true bin (load_bin(0) keeps it), so no join is needed for the bins.
#
# INPUTS (every CSV read as text)
# ------
#   results/analysis/reference/documents.csv            id, period_bin, topic_id, topic_assignment — from the hand-back zip; NOT in git (W4)
#   data/analysis/post_variables.csv                    id, period_bin, topic_uid, topic_assignment
#   results/analysis/topic_labels_all_bins.csv          the period topics (the spine of the mapping)
#   results/analysis/reference/topics.csv, topics_labelled.csv   the reference topics; the labels may be incomplete while the labelling runs
#   results/analysis/bin-N/topic_embeddings.npy, results/analysis/reference/topic_embeddings.npy   through reticulate and the .venv
#   results/analysis/manual review/BIN<N>_topics_labelled_*.csv   TP's files carry QUALTAG + qt_*, BA's qt_* only (read-only here)
# The locations can be redirected for a test (as 06/07 do through TAG_*): REF_ANALYSIS_DIR (results/analysis), REF_DATA_DIR
# (data/analysis), REF_REFERENCE_DIR (the reference run's folder), REF_TABLES_DIR (where the three tables go; default = the
# reference folder), REF_FIG_DIR (results/descriptives), REF_REVIEW_DIR (manual review), REF_PYTHON (.venv/bin/python).
#
# HOW TO RUN (repository root; needs the .venv for the .npy arrays)
# ----------------------------
#   Rscript code/16-reference-map.R          # ~2-3 min
#
# OUTPUTS
# -------
#   results/analysis/reference/map_pairs.csv            one row per (period topic, reference topic) pair with posts counted on both sides
#   results/analysis/reference/map_period_topics.csv    one row per period topic: best / second reference topic by posts, nearest by centroid, QUALTAG
#   results/analysis/reference/reference_topics.csv     one row per reference topic: sizes, posts per bin, largest period topics, QUALTAG composition
#   results/descriptives/fig_G1_reference_best_share_by_bin.png|pdf, tab_G1_reference_best_share_by_bin.csv
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2); library(scales); library(stringr) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                     # shared helpers (code/R/pipeline-utils.R): log_msg, check, read_chr, find_root, BIN_TYPE
ROOT <- find_root()
setwd(ROOT)
source(file.path(dirname(.utils), "plot-utils.R"))   # palette, theme_fifa, save_fig / save_tab

ANALYSIS_DIR  <- Sys.getenv("REF_ANALYSIS_DIR", "results/analysis")
DATA_DIR      <- Sys.getenv("REF_DATA_DIR", "data/analysis")
REFERENCE_DIR <- Sys.getenv("REF_REFERENCE_DIR", file.path(ANALYSIS_DIR, "reference"))
TABLES_DIR    <- Sys.getenv("REF_TABLES_DIR", REFERENCE_DIR)
OUT_DIR       <- Sys.getenv("REF_FIG_DIR", "results/descriptives")              # save_fig / save_tab write here
REVIEW_DIR    <- Sys.getenv("REF_REVIEW_DIR", file.path(ANALYSIS_DIR, "manual review"))
PYTHON        <- Sys.getenv("REF_PYTHON", ".venv/bin/python")
COUNT    <- c("cluster", "reassigned")     # config.TOPIC_COUNT_ASSIGNMENTS (decision K1, 2026-10-30)
N_CORPUS <- 1302021L                       # the cleaned corpus (codebook §1)
QT       <- c("qt_political", "qt_host_country_domestic", "qt_national_identity", "qt_fifa_body", "qt_ir")
STATUS   <- c("tp_yes", "tp_no", "tp_review", "tp_to_settle", "ba_1", "ba_0", "ba_blank", "uncoded_bin", "unmapped")
FILE_RE  <- "^BIN(\\d)_topics_labelled_.*\\.csv$"
dir.create(TABLES_DIR, showWarnings = FALSE, recursive = TRUE); dir.create(OUT_DIR, showWarnings = FALSE, recursive = TRUE)

# ---------------------------------------------------------------------------
# 1. The two post tables: the reference run's and the period bins'
# ---------------------------------------------------------------------------
p_ref <- file.path(REFERENCE_DIR, "documents.csv")
if (!file.exists(p_ref)) stop(p_ref, " not found: it comes with the hand-back zip of the reference run (results/analysis/reference/) and is not in git")
ref <- read_chr(p_ref, col_select = c("id", "period_bin", "topic_id", "topic_assignment")) |>
  transmute(id, period_bin = as.integer(period_bin), ref_topic = topic_id, ref_counted = topic_assignment %in% COUNT)
pv <- read_chr(file.path(DATA_DIR, "post_variables.csv"), col_select = c("id", "period_bin", "topic_uid", "topic_assignment")) |>
  transmute(id, pv_bin = as.integer(period_bin), topic_uid, per_counted = topic_assignment %in% COUNT)
log_msg("reference posts %s, period posts %s", format(nrow(ref), big.mark = ","), format(nrow(pv), big.mark = ","))
check("both post tables hold the corpus (1,302,021 posts each)", nrow(ref) == N_CORPUS && nrow(pv) == N_CORPUS, sprintf("%d / %d", nrow(ref), nrow(pv)))
check("no duplicated post id", !anyDuplicated(ref$id) && !anyDuplicated(pv$id))
d <- inner_join(ref, pv, by = "id")
check("the two tables hold the same posts", nrow(d) == nrow(ref) && nrow(d) == nrow(pv), sprintf("%d joined", nrow(d)))
check("period_bin agrees post by post", all(d$period_bin == d$pv_bin))
check("every reference-counted post has a non-negative topic id", all(!d$ref_counted | (d$ref_topic != "" & d$ref_topic != "-1")))
log_msg("counted on the reference side %s (%.1f %%), on the period side %s (%.1f %%), on both %s",
        format(sum(d$ref_counted), big.mark = ","), 100 * mean(d$ref_counted), format(sum(d$per_counted), big.mark = ","), 100 * mean(d$per_counted),
        format(sum(d$ref_counted & d$per_counted), big.mark = ","))

# ---------------------------------------------------------------------------
# 2. The topics on both sides
# ---------------------------------------------------------------------------
lab <- read_chr(file.path(ANALYSIS_DIR, "topic_labels_all_bins.csv"),
                col_select = c("period_bin", "topic_id", "topic_uid", "name", "tone_category", "n_total")) |>
  mutate(across(c(period_bin, topic_id, n_total), as.integer)) |> arrange(period_bin, topic_id)
check("one row per period topic, unique uids", !anyDuplicated(lab$topic_uid) && all(lab$topic_uid == paste(lab$period_bin, lab$topic_id, sep = "-")),
      sprintf("%d topics", nrow(lab)))
per_n <- d |> filter(per_counted) |> count(topic_uid, name = "n_counted")
check("every counted period topic is a labelled topic", all(per_n$topic_uid %in% lab$topic_uid))
chk <- lab |> left_join(per_n, by = "topic_uid") |> mutate(n_counted = coalesce(n_counted, 0L))
check("per period topic: counted posts = n_total (K1)", all(chk$n_counted == chk$n_total), sprintf("%d differ", sum(chk$n_counted != chk$n_total)))

rt <- read_chr(file.path(REFERENCE_DIR, "topics.csv")) |> filter(topic_id != "-1") |>
  transmute(ref_topic = topic_id, n_cluster = as.integer(n_cluster), n_reassigned = as.integer(n_reassigned), n_total = as.integer(n_total), keywords) |>
  arrange(as.integer(ref_topic))
p_rl <- file.path(REFERENCE_DIR, "topics_labelled.csv")
rl <- if (file.exists(p_rl)) read_chr(p_rl, col_select = c("topic_id", "name", "tone_category", "needs_review")) |> rename(ref_topic = topic_id) else
  tibble(ref_topic = character(), name = character(), tone_category = character(), needs_review = character())
rt <- rt |> left_join(rl, by = "ref_topic")
log_msg("reference topics %d, named %d%s", nrow(rt), sum(!is.na(rt$name)), if (sum(is.na(rt$name))) " (labelling incomplete: names empty where missing)" else "")
check("reference topic ids are 0..n-1 in order", identical(as.integer(rt$ref_topic), seq_len(nrow(rt)) - 1L))
ref_n <- d |> filter(ref_counted) |> count(ref_topic, name = "n_counted")
check("every counted reference topic is in reference/topics.csv", all(ref_n$ref_topic %in% rt$ref_topic))
chk <- rt |> left_join(ref_n, by = "ref_topic") |> mutate(n_counted = coalesce(n_counted, 0L))
check("per reference topic: counted posts = n_total (K1)", all(chk$n_counted == chk$n_total), sprintf("%d differ", sum(chk$n_counted != chk$n_total)))

# ---------------------------------------------------------------------------
# 3. The pairs: posts counted on both sides
# ---------------------------------------------------------------------------
both  <- d |> filter(ref_counted, per_counted)
pairs <- both |> count(period_bin, topic_uid, ref_topic, name = "n_posts") |>
  left_join(lab |> select(topic_uid, per_total = n_total), by = "topic_uid") |>
  left_join(rt |> select(ref_topic, ref_total = n_total), by = "ref_topic") |>
  mutate(share_of_period_topic = n_posts / per_total, share_of_reference_topic = n_posts / ref_total)
check("Σ n_posts over the pairs = posts counted on both sides", sum(pairs$n_posts) == nrow(both), format(nrow(both), big.mark = ","))
s_per <- pairs |> group_by(topic_uid) |> summarise(s = sum(share_of_period_topic), .groups = "drop")
s_ref <- pairs |> group_by(ref_topic) |> summarise(s = sum(share_of_reference_topic), .groups = "drop")
check("the shares of a period topic sum to at most 1", all(s_per$s <= 1 + 1e-9), sprintf("max %.4f", max(s_per$s)))
check("the shares of a reference topic sum to at most 1", all(s_ref$s <= 1 + 1e-9), sprintf("max %.4f", max(s_ref$s)))

# ---------------------------------------------------------------------------
# 4. Centroid similarity (the mean mpnet embedding of each topic's posts; one space on both sides)
# ---------------------------------------------------------------------------
reticulate::use_python(PYTHON, required = TRUE)
np <- reticulate::import("numpy")
emb_per <- do.call(rbind, lapply(1:9, function(b) {
  e <- np$load(file.path(ANALYSIS_DIR, sprintf("bin-%d/topic_embeddings.npy", b)))
  ids <- lab$topic_id[lab$period_bin == b]
  check(sprintf("bin %d: one embedding per labelled topic", b), nrow(e) == length(ids) && identical(ids, seq_len(nrow(e)) - 1L), sprintf("%d", nrow(e)))
  e
}))
e_ref <- np$load(file.path(REFERENCE_DIR, "topic_embeddings.npy"))
check("reference: one embedding per topic", nrow(e_ref) == nrow(rt), sprintf("%d rows, %d topics", nrow(e_ref), nrow(rt)))
check("embeddings of the same dimension on both sides", ncol(e_ref) == ncol(emb_per), sprintf("%d / %d", ncol(e_ref), ncol(emb_per)))
unit <- function(m) m / sqrt(rowSums(m^2))
cos_mat <- tcrossprod(unit(emb_per), unit(e_ref))             # period topics × reference topics
pairs$centroid_cosine <- cos_mat[cbind(match(pairs$topic_uid, lab$topic_uid), match(pairs$ref_topic, rt$ref_topic))]
nearest <- tibble(topic_uid = lab$topic_uid, nearest_ref_topic = rt$ref_topic[max.col(cos_mat, ties.method = "first")],
                  nearest_cosine = apply(cos_mat, 1, max))
check("cosines within [-1, 1]", all(abs(cos_mat) <= 1 + 1e-9))

# ---------------------------------------------------------------------------
# 5. The co-authors' QUALTAG coding per period topic
# ---------------------------------------------------------------------------
files  <- list.files(REVIEW_DIR, pattern = FILE_RE)
bins_f <- as.integer(sub(FILE_RE, "\\1", files))
check("one review file per coded bin", !anyDuplicated(bins_f), paste(sort(bins_f), collapse = " "))
qt <- bind_rows(lapply(files, function(f) {
  b <- as.integer(sub(FILE_RE, "\\1", f)); x <- read_chr(file.path(REVIEW_DIR, f))
  coder <- if ("QUALTAG" %in% names(x)) "TP" else "BA"      # the QUALTAG scheme files (TP; bin 1 with BA) vs BA's direct qt_ files (bins 2, 4)
  miss <- setdiff(QT, names(x)); if (length(miss)) stop(f, ": column(s) missing: ", paste(miss, collapse = ", "))
  x |> transmute(period_bin = b, topic_uid = paste(b, topic_id, sep = "-"), qt_coder = coder, across(all_of(QT), ~ str_trim(.x)))
}))
check("no duplicated topic in the review files", !anyDuplicated(qt$topic_uid))
check("every period topic of a coded bin has a review row", all(lab$topic_uid[lab$period_bin %in% bins_f] %in% qt$topic_uid))
check("every review row names a labelled topic", all(qt$topic_uid %in% lab$topic_uid))
ok_vals <- qt |> mutate(ok = if_else(qt_coder == "TP", qt_political %in% c("yes", "no", "review", ""), qt_political %in% c("1", "0", ""))) |> pull(ok)
check("qt_political within each coder's value set (TP yes/no/review/blank; BA 1/0/blank)", all(ok_vals), sprintf("%d outside", sum(!ok_vals)))
qt <- qt |> mutate(qt_status = case_when(qt_coder == "TP" & qt_political == "yes" ~ "tp_yes", qt_coder == "TP" & qt_political == "no" ~ "tp_no",
                                         qt_coder == "TP" & qt_political == "review" ~ "tp_review", qt_coder == "TP" ~ "tp_to_settle",
                                         qt_coder == "BA" & qt_political == "1" ~ "ba_1", qt_coder == "BA" & qt_political == "0" ~ "ba_0", TRUE ~ "ba_blank"))
log_msg("QUALTAG rows: %s", paste(sprintf("%s %d", names(table(qt$qt_status)), table(qt$qt_status)), collapse = ", "))

# ---------------------------------------------------------------------------
# 6. map_period_topics: one row per period topic
# ---------------------------------------------------------------------------
best <- pairs |> arrange(topic_uid, desc(n_posts), as.integer(ref_topic)) |> group_by(topic_uid) |>
  summarise(n_both = sum(n_posts), n_ref_topics = n(), best_ref_topic = first(ref_topic), best_n = first(n_posts), best_share = first(share_of_period_topic),
            second_ref_topic = nth(ref_topic, 2), second_share = nth(share_of_period_topic, 2), .groups = "drop")
mpt <- lab |> left_join(best, by = "topic_uid") |> left_join(nearest, by = "topic_uid") |>
  left_join(rt |> select(best_ref_topic = ref_topic, best_ref_name = name), by = "best_ref_topic") |>
  left_join(qt |> select(topic_uid, qt_coder, all_of(QT)), by = "topic_uid") |>
  mutate(n_both = coalesce(n_both, 0L), n_ref_topics = coalesce(n_ref_topics, 0L),
         agree = if_else(!is.na(best_ref_topic), best_ref_topic == nearest_ref_topic, NA),
         qt_coder = coalesce(qt_coder, ""), across(all_of(QT), ~ coalesce(.x, ""))) |>
  select(period_bin, topic_id, topic_uid, name, tone_category, n_total, n_both, best_ref_topic, best_ref_name, best_n, best_share,
         second_ref_topic, second_share, n_ref_topics, nearest_ref_topic, nearest_cosine, agree, qt_coder, all_of(QT))
check("map_period_topics: one row per period topic", nrow(mpt) == nrow(lab) && !anyDuplicated(mpt$topic_uid))
check("n_both never exceeds n_total", all(mpt$n_both <= mpt$n_total))
log_msg("period topics with counted overlap %d of %d; best-match share median %.3f; best = nearest by centroid for %.1f %% of them",
        sum(mpt$n_both > 0), nrow(mpt), median(mpt$best_share, na.rm = TRUE), 100 * mean(mpt$agree, na.rm = TRUE))

# ---------------------------------------------------------------------------
# 7. reference_topics: one row per reference topic
# ---------------------------------------------------------------------------
per_bin <- d |> filter(ref_counted) |> count(ref_topic, period_bin) |> mutate(period_bin = paste0("n_bin", period_bin)) |>
  pivot_wider(names_from = period_bin, values_from = n, values_fill = 0L)
for (col in paste0("n_bin", 1:9)) if (!col %in% names(per_bin)) per_bin[[col]] <- 0L
comp <- d |> filter(ref_counted) |> left_join(qt |> select(topic_uid, qt_status), by = "topic_uid") |>
  mutate(status = case_when(!per_counted ~ "unmapped", !is.na(qt_status) ~ qt_status, TRUE ~ "uncoded_bin")) |>
  count(ref_topic, status) |> pivot_wider(names_from = status, values_from = n, values_fill = 0L)
for (col in STATUS) if (!col %in% names(comp)) comp[[col]] <- 0L
top3 <- pairs |> arrange(ref_topic, desc(n_posts), topic_uid) |> group_by(ref_topic) |> slice_head(n = 3) |>
  left_join(lab |> select(topic_uid, per_name = name), by = "topic_uid") |>
  summarise(top_period_topics = paste0(topic_uid, " ", per_name, " (", n_posts, ")", collapse = " | "), .groups = "drop")
n_both_ref <- pairs |> group_by(ref_topic) |> summarise(n_both = sum(n_posts), .groups = "drop")
rtop <- rt |> left_join(per_bin, by = "ref_topic") |> left_join(n_both_ref, by = "ref_topic") |> left_join(top3, by = "ref_topic") |>
  left_join(comp, by = "ref_topic") |>
  mutate(across(c(starts_with("n_bin"), n_both, all_of(STATUS)), ~ coalesce(.x, 0L)), coverage = if_else(n_total > 0, n_both / n_total, NA_real_),
         top_period_topics = coalesce(top_period_topics, ""))
check("reference_topics: posts per bin sum to n_total", all(rowSums(rtop[paste0("n_bin", 1:9)]) == rtop$n_total))
check("reference_topics: the QUALTAG composition sums to n_total", all(rowSums(rtop[STATUS]) == rtop$n_total))
check("reference_topics: n_both = n_total - unmapped", all(rtop$n_both == rtop$n_total - rtop$unmapped))
shares <- rtop |> transmute(ref_topic, across(all_of(STATUS), ~ if_else(n_total > 0, .x / n_total, NA_real_), .names = "share_{.col}"))
rtop <- rtop |> left_join(shares, by = "ref_topic") |>
  select(ref_topic, name, tone_category, needs_review, keywords, n_cluster, n_reassigned, n_total, all_of(paste0("n_bin", 1:9)), n_both, coverage,
         top_period_topics, all_of(STATUS), starts_with("share_")) |>
  rename_with(~ paste0("n_", .x), all_of(STATUS))
check("reference_topics: one row per reference topic", nrow(rtop) == nrow(rt) && !anyDuplicated(rtop$ref_topic))
log_msg("reference topics with a TP-coded majority share: yes %d, no %d (share > 0.5 of counted posts)",
        sum(rtop$share_tp_yes > 0.5, na.rm = TRUE), sum(rtop$share_tp_no > 0.5, na.rm = TRUE))

# ---------------------------------------------------------------------------
# 8. Write the tables (readr, na = "")
# ---------------------------------------------------------------------------
r4 <- function(x) round(x, 4)
write_csv(pairs |> mutate(across(c(share_of_period_topic, share_of_reference_topic, centroid_cosine), r4)) |>
            select(period_bin, topic_uid, ref_topic, n_posts, share_of_period_topic, share_of_reference_topic, centroid_cosine) |>
            arrange(period_bin, topic_uid, desc(n_posts)), file.path(TABLES_DIR, "map_pairs.csv"), na = "")
write_csv(mpt |> mutate(across(c(best_share, second_share, nearest_cosine), r4)), file.path(TABLES_DIR, "map_period_topics.csv"), na = "")
write_csv(rtop |> mutate(across(c(coverage, starts_with("share_")), r4)), file.path(TABLES_DIR, "reference_topics.csv"), na = "")
log_msg("tables written to %s: map_pairs (%d rows), map_period_topics (%d), reference_topics (%d)", TABLES_DIR, nrow(pairs), nrow(mpt), nrow(rtop))

# ---------------------------------------------------------------------------
# 9. Figure G1: how cleanly the period topics fall into reference topics
# ---------------------------------------------------------------------------
g1 <- mpt |> filter(n_both > 0) |> mutate(bin = factor(period_bin, levels = 1:9), period_type = BIN_TYPE[period_bin])
tab_g1 <- g1 |> group_by(period_bin, period_type) |>
  summarise(n_topics = n(), median_best_share = median(best_share), q1 = quantile(best_share, 0.25), q3 = quantile(best_share, 0.75),
            share_best_above_half = mean(best_share > 0.5), share_methods_agree = mean(agree, na.rm = TRUE), .groups = "drop")
save_tab(tab_g1 |> mutate(across(c(median_best_share, q1, q3, share_best_above_half, share_methods_agree), r4)), "G1_reference_best_share_by_bin")
p <- ggplot(g1, aes(bin, best_share, fill = period_type)) +
  geom_boxplot(outlier.size = 0.4, outlier.alpha = 0.4, linewidth = 0.3, colour = INK2, width = 0.6) +
  scale_fill_manual(values = c(tournament = SERIES[1], interim = SERIES[2]), breaks = c("tournament", "interim")) +
  scale_y_continuous(labels = percent, limits = c(0, 1)) +
  labs(title = "How cleanly the period topics fall into the reference topics",
       subtitle = sprintf(paste0("Share of a period topic's counted posts (cluster + reassigned, K1) that fall into its best-matching reference topic, by period bin.\n",
                                 "%s period topics with counted overlap; median best-match share %.0f %%.\n",
                                 "The reference topic holding most of a period topic's posts is also the nearest by centroid for %.0f %% of the period topics."),
                          format(nrow(g1), big.mark = ","), 100 * median(g1$best_share), 100 * mean(g1$agree, na.rm = TRUE)),
       caption = "Reference model: one BERTopic model on the whole corpus (session 12, 2026-10-30; guard off, decision W1). Table: tab_G1_reference_best_share_by_bin.csv.",
       x = "Period bin", y = "Best-match share") +
  theme_fifa() + theme(panel.grid.major.x = element_blank())
save_fig(p, "G1_reference_best_share_by_bin")

# ---------------------------------------------------------------------------
if (!all(unlist(results))) stop("one or more checks FAILED — see above")
log_msg("ALL CHECKS PASSED")
