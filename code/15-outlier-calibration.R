##############################################################################
# FIFA COMMUNITIES CHAPTER
## 15. OUTLIER CALIBRATION  (the coded judgement samples of the outlier tiers)
##############################################################################
#
# PURPOSE
# -------
# The precision of the outlier tiers of decision D9 (design §6.3), measured by the
# co-authors' judgement samples: for every period bin the pipeline wrote four
# 100-post samples of raw HDBSCAN outliers, one per similarity band (posts just
# above the bin's 35th, 50th, 65th and 80th percentile of outlier-to-nearest-topic
# similarity), and a coder judged for each post whether it belongs with the topic
# it would receive (`belongs` = 1/0).  This script turns the 36 coded files into
# the tables and figures of record and is the independent check of the Python
# analysis made during the assessment of 2026-10-30
# (docs/session-11_2026-10-30/01_outlier-calibration-results.md).
#
# WHY
# ---
# The percentiles of D9 are a rank, not a quality level; the samples say how
# often an admitted post belongs.  Alan Nemirovski's decisions K1-K7 of
# 2026-10-30 (docs/session-11_2026-10-30/00_approvals.md) rest on these numbers:
# the per-post tiers stand, but every topic-level aggregate counts cluster +
# reassigned posts only (K1); the headline precision is the plain pooled average
# over all judgements, each counting once (K2); the samples are single-coded and
# the coders differ, so the per-coder table stands beside the pooled one.
#
# DESIGN PRINCIPLES
# -----------------
# * Every CSV is read as text; nothing in results/analysis/ is modified.
# * Each check prints PASS/FAIL and the script stops on a FAIL (pipeline-utils).
# * Coded files are recognised by `coded` anywhere in the name (`_coded_BA`,
#   `-coded-AM`, `-coded-AN`, with or without spaces), any column order.
# * 95 % intervals are Wilson intervals; precision = share of `belongs` = 1.
#
# HOW TO RUN (repository root)
# ----------------------------
#   Rscript code/15-outlier-calibration.R          # ~30 s
#
# OUTPUTS  (results/descriptives/)
# -------
#   tab_F1_outlier_precision_bin_band.csv       bin x band: coder, n, belongs, precision, Wilson CI, posts of the band in the corpus
#   tab_F2_outlier_precision_pooled.csv         pooled by band, tier, coder, period type and coder x tier: plain pooling (the
#                                               figure of record), with the corpus-weighted and equal-coder variants beside it
#   tab_F3_outlier_precision_by_similarity.csv  precision by absolute similarity band, per coder and pooled
#   tab_F4_candidate_quantiles_by_bin.csv       what each candidate floor would leave as outliers, per bin
#   tab_F4b_candidate_tiers_corpus.csv          corpus-wide tiers under candidate (floor, single-topic) quantile pairs
#   fig_F1_outlier_precision_by_band.png/.pdf   precision by band and bin, with intervals, coder named
#   fig_F2_outlier_precision_by_similarity.png/.pdf  precision against absolute similarity, one line per coder
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr); library(ggplot2); library(scales); library(stringr) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                     # shared helpers (code/R/pipeline-utils.R): log_msg, check, read_chr, find_root
ROOT <- find_root()
setwd(ROOT)
source(file.path(dirname(.utils), "plot-utils.R"))   # palette, theme_fifa, save_fig / save_tab

ANALYSIS <- "results/analysis"
OUT_DIR  <- "results/descriptives"
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)

BINS <- tibble(bin = 1:9,
               period_type = rep(c("tournament", "interim"), length.out = 9),
               label = c("Bin 1 · WC 2010", "Bin 2 · interim", "Bin 3 · WC 2014", "Bin 4 · interim", "Bin 5 · WC 2018",
                         "Bin 6 · interim", "Bin 7 · WC 2022", "Bin 8 · interim", "Bin 9 · WC 2026"))
BAND_LABELS <- c("0.35" = "35th band (T)", "0.5" = "50th band", "0.65" = "65th band", "0.8" = "≥ 80th (reassigned)")
TIER_OF     <- function(q) if_else(q >= 0.8, "reassigned", "dual")
CANDIDATE_Q <- c(0.35, 0.50, 0.65, 0.80, 0.90, 0.95)

wilson_lo <- function(k, n, z = 1.96) { p <- k / n; d <- 1 + z^2 / n; (p + z^2 / (2 * n)) / d - z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d }
wilson_hi <- function(k, n, z = 1.96) { p <- k / n; d <- 1 + z^2 / n; (p + z^2 / (2 * n)) / d + z * sqrt(p * (1 - p) / n + z^2 / (4 * n^2)) / d }

# ---------------------------------------------------------------------------
# 1. Read the 36 coded samples, the band tables, the nearest-topic tables and the topic sizes
# ---------------------------------------------------------------------------
judgements <- list(); bands <- list(); nearest <- list(); n_modelled <- integer(9)
for (b in 1:9) {
  bdir <- file.path(ANALYSIS, sprintf("bin-%d", b)); cdir <- file.path(bdir, "outlier_calibration")
  tab <- read_chr(file.path(cdir, "calibration_table.csv")) |>
    mutate(bin = b, quantile = as.numeric(quantile), threshold = as.numeric(threshold),
           n_outliers = as.integer(n_outliers), n_reassigned_at_T = as.integer(n_reassigned_at_T)) |>
    arrange(quantile) |>
    mutate(n_corpus_band = n_reassigned_at_T - lead(n_reassigned_at_T, default = 0L))   # posts of the band in the corpus
  bands[[b]] <- tab
  near <- read_chr(file.path(bdir, "outlier_nearest.csv")) |>
    mutate(similarity = as.numeric(similarity), similarity_2 = as.numeric(similarity_2))
  nearest[[b]] <- near |> mutate(bin = b)
  n_modelled[b] <- sum(read_chr(file.path(bdir, "documents.csv"), col_select = "topic_id_raw")$topic_id_raw != "")
  sizes <- read_chr(file.path(bdir, "topics.csv"), col_select = c("topic_id", "n_cluster")) |>
    transmute(nearest_topic = topic_id, n_cluster = as.integer(n_cluster))
  files <- list.files(cdir, pattern = "^sample_T[0-9.]+.*coded.*\\.csv$", ignore.case = TRUE, full.names = TRUE)
  for (f in files) {
    nm <- basename(f)
    T_chr <- str_match(nm, "^sample_T([0-9.]+)")[, 2]
    coder <- str_match(nm, "coded[-_ ]*([A-Za-z]+)\\.csv$")[, 2]
    s <- read_chr(f)
    blank <- read_chr(file.path(cdir, sprintf("sample_T%s.csv", T_chr)))
    check(sprintf("bin %d %s: 100 rows, ids as in the blank sample, belongs in {0, 1}", b, nm),
          nrow(s) == 100 && identical(s$id, blank$id) && all(str_trim(s$belongs) %in% c("0", "1")),
          sprintf("%d rows", nrow(s)))
    q <- tab$quantile[abs(tab$threshold - as.numeric(T_chr)) < 1e-6]
    check(sprintf("bin %d %s: the file's threshold is one of the bin's bands", b, nm), length(q) == 1)
    judgements[[length(judgements) + 1]] <- s |>
      transmute(bin = b, coder = coder, quantile = q, T = as.numeric(T_chr), id, nearest_topic,
                nearest_topic_name, similarity = as.numeric(similarity), belongs = as.integer(str_trim(belongs))) |>
      left_join(near |> select(id, nearest_topic_2, similarity_2), by = "id") |>
      left_join(sizes, by = "nearest_topic")
  }
}
J <- bind_rows(judgements) |>
  left_join(BINS, by = "bin") |>
  mutate(band = factor(BAND_LABELS[as.character(quantile)], levels = BAND_LABELS), tier = TIER_OF(quantile),
         margin = similarity - similarity_2)
B <- bind_rows(bands)
check("36 coded files, 3,600 judgements", length(judgements) == 36 && nrow(J) == 3600, sprintf("%d files, %d rows", length(judgements), nrow(J)))
check("one coder per bin", all((J |> distinct(bin, coder) |> count(bin))$n == 1))
check("every judgement joined to its nearest-topic row and its topic's size", !anyNA(J$similarity_2) && !anyNA(J$n_cluster))
log_msg("coders: %s", paste(sprintf("%s = bins %s", unique(J$coder), sapply(unique(J$coder), function(c) paste(sort(unique(J$bin[J$coder == c])), collapse = ","))), collapse = "; "))

# ---------------------------------------------------------------------------
# 2. F1: precision per bin and band (the table behind the first figure)
# ---------------------------------------------------------------------------
f1 <- J |>
  group_by(bin, period_type, coder, quantile, band, tier, T) |>
  summarise(n = n(), belongs = sum(belongs), .groups = "drop") |>
  mutate(precision = belongs / n, ci_lo = wilson_lo(belongs, n), ci_hi = wilson_hi(belongs, n)) |>
  left_join(B |> select(bin, quantile, n_corpus_band), by = c("bin", "quantile")) |>
  arrange(bin, quantile)
save_tab(f1 |> mutate(across(c(precision, ci_lo, ci_hi), ~ round(.x, 4))), "F1_outlier_precision_bin_band")

# ---------------------------------------------------------------------------
# 3. F2: pooled precision — plain pooling (K2, the figure of record) with the two other weightings beside it
# ---------------------------------------------------------------------------
pooled <- function(d, group_type, group) {
  # plain pooling: every judgement counts once
  k <- sum(d$belongs); n <- nrow(d)
  # corpus weighting: each bin x band precision weighted by the posts of that band in the corpus
  cells <- d |> group_by(bin, quantile) |> summarise(p = mean(belongs), .groups = "drop") |>
    left_join(B |> select(bin, quantile, n_corpus_band), by = c("bin", "quantile"))
  # equal weight per coder: each coder's own pooled precision, then the mean over the coders present
  per_coder <- d |> group_by(coder) |> summarise(p = mean(belongs), .groups = "drop")
  tibble(group_type = group_type, group = group, n = n, belongs = k, precision = k / n,
         ci_lo = wilson_lo(k, n), ci_hi = wilson_hi(k, n)) |>
    mutate(precision_corpus_weighted = sum(cells$p * cells$n_corpus_band) / sum(cells$n_corpus_band),
           precision_equal_coder = if (nrow(per_coder) > 1) mean(per_coder$p) else NA_real_,
           n_coders = nrow(per_coder))
}
f2 <- bind_rows(
  lapply(levels(J$band), function(x) pooled(filter(J, band == x), "band", x)),
  pooled(filter(J, tier == "dual"), "tier", "dual (35th–80th)"),
  pooled(filter(J, tier == "reassigned"), "tier", "reassigned (≥ 80th)"),
  pooled(J, "tier", "all admitted (≥ 35th)"),
  lapply(sort(unique(J$coder)), function(c) pooled(filter(J, coder == c), "coder", c)),
  lapply(c("tournament", "interim"), function(t) pooled(filter(J, period_type == t), "period_type", t)),
  lapply(sort(unique(J$coder)), function(c) bind_rows(
    pooled(filter(J, coder == c, tier == "dual"), "coder_tier", paste(c, "dual")),
    pooled(filter(J, coder == c, tier == "reassigned"), "coder_tier", paste(c, "reassigned"))))
)
check("pooled n: 900 per band, 2,700 dual, 900 reassigned, 3,600 in all",
      all(f2$n[f2$group_type == "band"] == 900) && f2$n[f2$group == "dual (35th–80th)"] == 2700 && f2$n[f2$group == "all admitted (≥ 35th)"] == 3600)
save_tab(f2 |> mutate(across(c(precision, ci_lo, ci_hi, precision_corpus_weighted, precision_equal_coder), ~ round(.x, 4))), "F2_outlier_precision_pooled")
hd <- f2 |> filter(group_type == "tier")
log_msg("precision of record (plain pooling): %s", paste(sprintf("%s %.1f %%", hd$group, 100 * hd$precision), collapse = " · "))
log_msg("by coder: %s", paste(sprintf("%s %.1f %%", f2$group[f2$group_type == "coder"], 100 * f2$precision[f2$group_type == "coder"]), collapse = " · "))

# ---------------------------------------------------------------------------
# 4. F3: precision by absolute similarity, per coder and pooled
# ---------------------------------------------------------------------------
sim_breaks <- c(0, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 1)
sim_labels <- c("≤ 0.55", "0.55–0.60", "0.60–0.65", "0.65–0.70", "0.70–0.75", "0.75–0.80", "> 0.80")
J <- J |> mutate(sim_band = cut(similarity, sim_breaks, labels = sim_labels, include.lowest = TRUE, right = TRUE),
                 sim_mid = (sim_breaks[as.integer(sim_band)] + sim_breaks[as.integer(sim_band) + 1]) / 2)
f3 <- bind_rows(
  J |> group_by(coder, sim_band, sim_mid) |> summarise(n = n(), belongs = sum(belongs), .groups = "drop"),
  J |> group_by(sim_band, sim_mid) |> summarise(n = n(), belongs = sum(belongs), .groups = "drop") |> mutate(coder = "all coders")) |>
  mutate(precision = belongs / n, ci_lo = wilson_lo(belongs, n), ci_hi = wilson_hi(belongs, n)) |>
  arrange(coder, sim_mid)
f3 <- f3 |> mutate(sim_mid = if_else(sim_band == "≤ 0.55", 0.525, if_else(sim_band == "> 0.80", 0.825, sim_mid)))   # end bands: a nominal midpoint for the plot
save_tab(f3 |> select(coder, sim_band, n, belongs, precision, ci_lo, ci_hi) |> mutate(across(c(precision, ci_lo, ci_hi), ~ round(.x, 4))),
         "F3_outlier_precision_by_similarity")

# ---------------------------------------------------------------------------
# 5. F4: what the candidate quantiles would do (from every bin's outlier similarities)
# ---------------------------------------------------------------------------
N <- bind_rows(nearest)
f4 <- bind_rows(lapply(1:9, function(b) {
  s <- N$similarity[N$bin == b]
  tibble(bin = b, n_modelled = n_modelled[b], n_raw_outliers = length(s), raw_outlier_share = length(s) / n_modelled[b],
         quantile = CANDIDATE_Q, threshold = as.numeric(quantile(s, CANDIDATE_Q, type = 7)),
         outliers_left_share = sapply(threshold, function(t) sum(s < t) / n_modelled[b]))
}))
check("F4: the 35th / 80th thresholds equal the pipeline's T and C of every bin",
      all(abs(f4$threshold[f4$quantile %in% c(0.35, 0.8)] - B$threshold[B$quantile %in% c(0.35, 0.8)]) < 1e-3))
save_tab(f4 |> mutate(across(c(raw_outlier_share, outliers_left_share), ~ round(.x, 4)), threshold = round(threshold, 4)), "F4_candidate_quantiles_by_bin")
pairs <- tribble(~floor_q, ~single_q, 0.35, 0.80, 0.50, 0.80, 0.65, 0.80, 0.80, 0.80, 0.35, 0.90, 0.50, 0.90, 0.65, 0.90, 0.80, 0.90, 0.90, 0.90, 0.95, 0.95)
tot_mod <- sum(n_modelled); tot_out <- nrow(N)
f4b <- pairs |> rowwise() |> mutate(tiers = list({
  acc <- c(reassigned = 0, dual = 0, outlier = 0)
  for (b in 1:9) { s <- N$similarity[N$bin == b]; tR <- quantile(s, floor_q, type = 7); tS <- quantile(s, single_q, type = 7)
    acc <- acc + c(sum(s >= tS), sum(s >= tR & s < tS), sum(s < tR)) }
  acc })) |> ungroup() |>
  mutate(n_cluster = tot_mod - tot_out, n_reassigned = sapply(tiers, `[[`, "reassigned"), n_dual = sapply(tiers, `[[`, "dual"),
         n_outlier = sapply(tiers, `[[`, "outlier"), share_reassigned = n_reassigned / tot_mod, share_dual = n_dual / tot_mod,
         share_outlier = n_outlier / tot_mod, n_modelled = tot_mod) |> select(-tiers)
save_tab(f4b |> mutate(across(starts_with("share_"), ~ round(.x, 4))), "F4b_candidate_tiers_corpus")

# ---------------------------------------------------------------------------
# 6. Figures
# ---------------------------------------------------------------------------
CODER_COL <- c(AM = SERIES[2], AN = SERIES[3], BA = SERIES[1], "all coders" = INK)
f1p <- f1 |> left_join(BINS |> select(bin, label), by = "bin") |>
  mutate(panel = factor(sprintf("%s · coder %s", label, coder), levels = sprintf("%s · coder %s", BINS$label, (f1 |> distinct(bin, coder) |> arrange(bin))$coder)))
p <- ggplot(f1p, aes(band, precision, colour = coder)) +
  geom_hline(yintercept = 0.5, colour = AXIS, linewidth = 0.3, linetype = "22") +
  geom_errorbar(aes(ymin = ci_lo, ymax = ci_hi), width = 0.18, linewidth = 0.4) +
  geom_point(size = 1.8) +
  facet_wrap(~ panel, ncol = 3) +
  scale_colour_manual(values = CODER_COL, guide = "none") +
  scale_y_continuous(labels = label_percent(accuracy = 1), limits = c(0, 1), expand = expansion(mult = 0)) +
  scale_x_discrete(labels = c("35th\n(T)", "50th", "65th", "≥ 80th\n(reassigned)")) +
  labs(title = "Share of admitted outlier posts judged to belong to their topic, by similarity band and bin",
       subtitle = sprintf("100 posts per band and bin, 95 %% Wilson intervals; dashed line = 50 %%.\nPooled, every judgement counting once: dual tier %.1f %%, reassigned tier %.1f %%, all admitted %.1f %%.",
                          100 * hd$precision[1], 100 * hd$precision[2], 100 * hd$precision[3]),
       x = "Similarity band of the bin's outliers (percentile of the outlier-to-nearest-topic similarity)", y = "Judged to belong",
       caption = paste("Decision D9 admits raw HDBSCAN outliers at or above the 35th percentile (35th–80th: dual, with a second topic; ≥ 80th: singly reassigned). One coder per bin (BA: bins 1, 3, 5, 7;",
                       "AM: 6, 8, 9; AN: 2, 4), so bin and coder differences cannot be separated. code/15-outlier-calibration.R; decisions K1–K7, docs/session-11_2026-10-30/.", sep = "\n")) +
  theme_fifa() + theme(panel.grid.major.x = element_blank())
save_fig(p, "F1_outlier_precision_by_band", h = 7.5)

p <- ggplot(f3, aes(sim_mid, precision, colour = coder, linetype = coder == "all coders")) +
  geom_hline(yintercept = 0.5, colour = AXIS, linewidth = 0.3, linetype = "22") +
  geom_line(linewidth = 0.6) +
  geom_point(aes(size = n), alpha = 0.9) +
  scale_colour_manual(values = CODER_COL) +
  scale_linetype_manual(values = c(`TRUE` = "42", `FALSE` = "solid"), guide = "none") +
  scale_size_area(max_size = 5, guide = "none") +
  scale_x_continuous(breaks = f3$sim_mid |> unique() |> sort(), labels = sim_labels) +
  scale_y_continuous(labels = label_percent(accuracy = 1), limits = c(0, 1), expand = expansion(mult = 0)) +
  labs(title = "Precision of the admitted posts against their similarity to the nearest topic",
       subtitle = "Share judged to belong, by band of absolute cosine similarity. One line per coder; the dashed line pools all judgements;\npoint size = judgements in the band.",
       x = "Cosine similarity of the post to its nearest topic (all bins)", y = "Judged to belong",
       caption = paste("The bins' cut points differ under the percentile rule (T from 0.497 in bin 1 to 0.616 in bin 8), so the same absolute similarity falls in different tiers in different bins.",
                       "Precision climbs to about 60 % at a similarity of 0.70 and plateaus. code/15-outlier-calibration.R.", sep = "\n")) +
  theme_fifa()
save_fig(p, "F2_outlier_precision_by_similarity")

log_msg("done: tables tab_F1–F4b and figures F1–F2 in %s", OUT_DIR)
