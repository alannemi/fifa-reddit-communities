##############################################################################
# FIFA COMMUNITIES CHAPTER
## 10. SENTIMENT VALIDATION  (agreement of the human coding with VADER)
##############################################################################
#
# PURPOSE
# -------
# Report how well `post_sentiment_label` (VADER compound of the title, cut at
# +/-0.05; design §8) matches human judgement, from the coding sheet(s)
# produced by 09-sentiment-coding-pack.R.  Decision E1 (Alan Nemirovski,
# 2026-10-27), revised the same day: ONE coder (Alan Nemirovski) codes the
# sample, so his codes are the reference and there is no inter-coder
# reliability.  The figures go into codebook §8.2 and limitation 10.
#
# WHAT IS COMPUTED
# ----------------
# One coder (coder_A.csv only — the protocol in force):
# 1. Reference (coder A) vs VADER: percent agreement and Cohen's kappa with a
#    bootstrap 95 % interval; confusion matrix,
# Two coders (only if coder_B.csv exists):
# 1. Coder A vs coder B (the reliability of the human reference); each coder vs
#    VADER.  The reference is the coders' common code; where they disagree, the
#    code in `adjudicated.csv` (id, code) if that file exists, otherwise the post
#    is left out and counted.
# Both cases then report, for the reference against VADER: confusion matrix,
#    precision / recall / F1 per class, macro-F1, and agreement by period bin
#    and by is_self (the sample's strata; small cells — read the n).
# Kappa is computed here (no irr dependency): kappa = (p_o - p_e) / (1 - p_e),
# p_e from the two raters' marginal shares over the three classes.
#
# HOW TO RUN (repository root; after the coding is finished — the output includes agreement with VADER)
# ----------------------------
#   Rscript code/10-sentiment-validation.R
# Input: the coded sheet results/analysis/manual review/sentiment_review_coder_A-coded-AN.csv (coder A, the reference).
# Environment variables (tests): SENT_CODER_A (another coded sheet); SENT_CODING_DIR (default results/analysis/manual review/sentiment_coding)
# holds coder_A.csv (and, with two coders, coder_B.csv and the optional adjudicated.csv); SENT_OUT_DIR (default
# results/descriptives).
# Codes accepted (case-insensitive): positive / neutral / negative, pos / neu / neg, 1 / 0 / -1.
#
# OUTPUTS  (SENT_OUT_DIR)
# -------
#   sentiment_validation_agreement.csv   one row per comparison: n, agreement, kappa, 95 % interval
#   sentiment_validation_confusion.csv   reference x VADER counts
#   sentiment_validation_by_class.csv    precision, recall, F1 per class (VADER against the reference)
#   sentiment_validation_by_stratum.csv  agreement with VADER by period_bin and by is_self
#   sentiment_validation_thresholds.csv  threshold sensitivity: for a neutral band |compound| <= t (t = 0.05 ... 0.75),
#                                        agreement and kappa against the reference, the share of each label in the
#                                        sample and in the whole corpus (data/analysis/sentiment.csv)  [author, 2026-10-27]
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                     # shared helpers (code/R/pipeline-utils.R)
ROOT <- find_root()
setwd(ROOT)

CODING_DIR <- Sys.getenv("SENT_CODING_DIR", "results/analysis/manual review/sentiment_coding")
OUT_DIR    <- Sys.getenv("SENT_OUT_DIR", "results/descriptives")
# The coded reference sheet: the author's file as returned on 2026-10-27 (kept under its own name, decision of the same
# day); a test run that sets SENT_CODING_DIR reads coder_A.csv there instead, SENT_CODER_A overrides both
CODER_A_FILE <- Sys.getenv("SENT_CODER_A", if (nzchar(Sys.getenv("SENT_CODING_DIR"))) file.path(CODING_DIR, "coder_A.csv")
                           else "results/analysis/manual review/sentiment_review_coder_A-coded-AN.csv")
SAMPLE     <- "data/analysis/sentiment_validation_sample.csv"
CLASSES    <- c("negative", "neutral", "positive")
N_BOOT     <- 2000L
SEED       <- 42L

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
norm_code <- function(x) {
  x <- tolower(trimws(x))
  out <- c(positive = "positive", pos = "positive", "1" = "positive", "+1" = "positive",
           neutral = "neutral", neu = "neutral", "0" = "neutral",
           negative = "negative", neg = "negative", "-1" = "negative")[x]
  unname(ifelse(x == "", NA_character_, out))
}

kappa <- function(a, b) {
  a <- factor(a, CLASSES); b <- factor(b, CLASSES)
  po <- mean(a == b)
  pe <- sum(prop.table(table(a)) * prop.table(table(b)))
  if (pe == 1) return(NA_real_)
  (po - pe) / (1 - pe)
}

compare <- function(name, a, b) {
  ok <- !is.na(a) & !is.na(b); a <- a[ok]; b <- b[ok]
  if (length(a) < 2)                 # a sheet not (yet) coded: the row is reported empty, the other comparisons still run
    return(tibble(comparison = name, n = length(a), agreement = NA_real_, kappa = NA_real_, kappa_lo95 = NA_real_, kappa_hi95 = NA_real_))
  set.seed(SEED)
  boot <- replicate(N_BOOT, { i <- sample(length(a), replace = TRUE); kappa(a[i], b[i]) })
  tibble(comparison = name, n = length(a), agreement = round(mean(a == b), 4), kappa = round(kappa(a, b), 4),
         kappa_lo95 = round(unname(quantile(boot, 0.025, na.rm = TRUE)), 4),
         kappa_hi95 = round(unname(quantile(boot, 0.975, na.rm = TRUE)), 4))
}

read_sheet <- function(coder) {
  p <- if (coder == "A") CODER_A_FILE else file.path(CODING_DIR, sprintf("coder_%s.csv", coder))
  s <- read_chr(p)
  bad <- s$code[s$code != "" & is.na(norm_code(s$code))]
  if (length(bad)) stop(p, ": codes not recognised: ", paste(unique(bad), collapse = ", "))
  log_msg("coder %s: %d rows, %d coded", coder, nrow(s), sum(s$code != ""))
  s |> transmute(id, !!paste0("code_", coder) := norm_code(code))
}

# ---------------------------------------------------------------------------
# 1. Inputs
# ---------------------------------------------------------------------------
smp <- read_chr(SAMPLE) |> select(id, period_bin, is_self, post_sentiment, post_sentiment_label)
TWO_CODERS <- file.exists(file.path(CODING_DIR, "coder_B.csv"))
log_msg("protocol: %s", if (TWO_CODERS) "two coders (coder_A.csv + coder_B.csv)" else "one coder (coder_A.csv is the reference)")
d <- smp |> left_join(read_sheet("A"), by = "id")
d <- if (TWO_CODERS) left_join(d, read_sheet("B"), by = "id") else mutate(d, code_B = NA_character_)
stopifnot(nrow(d) == nrow(smp), !anyDuplicated(d$id))

adj_path <- file.path(CODING_DIR, "adjudicated.csv")
adj <- if (file.exists(adj_path)) read_chr(adj_path) |> transmute(id, code_adj = norm_code(code)) else tibble(id = character(), code_adj = character())
d <- d |> left_join(adj, by = "id") |>
  mutate(reference = case_when(!TWO_CODERS ~ code_A,                          # one coder: the codes are the reference
                               !is.na(code_A) & code_A == code_B ~ code_A,
                               !is.na(code_adj) ~ code_adj,
                               TRUE ~ NA_character_))
if (TWO_CODERS) {
  n_dis <- sum(!is.na(d$code_A) & !is.na(d$code_B) & d$code_A != d$code_B)
  log_msg("both coded: %d; disagreements: %d (%d adjudicated); reference label for %d posts",
          sum(!is.na(d$code_A) & !is.na(d$code_B)), n_dis, sum(!is.na(d$code_adj)), sum(!is.na(d$reference)))
} else {
  log_msg("reference label (coder A) for %d of %d posts", sum(!is.na(d$reference)), nrow(d))
}

# ---------------------------------------------------------------------------
# 2. Agreement and kappa
# ---------------------------------------------------------------------------
agree <- if (TWO_CODERS) {
  bind_rows(compare("coder_A vs coder_B", d$code_A, d$code_B),
            compare("coder_A vs VADER", d$code_A, d$post_sentiment_label),
            compare("coder_B vs VADER", d$code_B, d$post_sentiment_label),
            compare("reference vs VADER", d$reference, d$post_sentiment_label))
} else {
  compare("coder_A (reference) vs VADER", d$reference, d$post_sentiment_label)
}

# ---------------------------------------------------------------------------
# 3. Reference vs VADER: confusion matrix, per-class scores, strata
# ---------------------------------------------------------------------------
r <- d |> filter(!is.na(reference))
conf <- r |> count(reference = factor(reference, CLASSES), vader = factor(post_sentiment_label, CLASSES), .drop = FALSE) |>
  pivot_wider(names_from = vader, values_from = n, names_prefix = "vader_")
by_class <- bind_rows(lapply(CLASSES, function(k) {
  tp <- sum(r$reference == k & r$post_sentiment_label == k)
  prec <- tp / sum(r$post_sentiment_label == k); rec <- tp / sum(r$reference == k)
  tibble(class = k, n_reference = sum(r$reference == k), n_vader = sum(r$post_sentiment_label == k),
         precision = round(prec, 4), recall = round(rec, 4), f1 = round(2 * prec * rec / (prec + rec), 4))
}))
by_class <- bind_rows(by_class, tibble(class = "macro", n_reference = nrow(r), n_vader = nrow(r),
                                       precision = round(mean(by_class$precision, na.rm = TRUE), 4),
                                       recall = round(mean(by_class$recall, na.rm = TRUE), 4),
                                       f1 = round(mean(by_class$f1, na.rm = TRUE), 4)))
strata <- bind_rows(
  r |> group_by(stratum = paste0("period_bin ", period_bin)) |> summarise(n = n(), agreement = round(mean(reference == post_sentiment_label), 4), .groups = "drop"),
  r |> group_by(stratum = paste0("is_self ", is_self)) |> summarise(n = n(), agreement = round(mean(reference == post_sentiment_label), 4), .groups = "drop"))

# ---------------------------------------------------------------------------
# 3b. Threshold sensitivity (author's request, 2026-10-27): does the conventional +/-0.05 neutral band suit this corpus?
#     Symmetric bands only (VADER's convention); a check of the encoding against the reference, not a fitted cut-off —
#     one coder and 337 posts would overfit an optimised threshold
# ---------------------------------------------------------------------------
SENTIMENT <- "data/analysis/sentiment.csv"
label_at <- function(x, t) ifelse(x > t, "positive", ifelse(x < -t, "negative", "neutral"))
corpus <- as.numeric(read_chr(SENTIMENT, col_select = "post_sentiment")$post_sentiment)
r$score <- as.numeric(r$post_sentiment)
thresholds <- bind_rows(lapply(c(0.05, seq(0.10, 0.75, by = 0.05)), function(t) {
  lab <- label_at(r$score, t); cl <- label_at(corpus, t)
  tibble(threshold = t, n = nrow(r), agreement = round(mean(lab == r$reference), 4), kappa = round(kappa(lab, r$reference), 4),
         sample_neutral = round(mean(lab == "neutral"), 3), sample_positive = round(mean(lab == "positive"), 3),
         sample_negative = round(mean(lab == "negative"), 3),
         corpus_neutral = round(mean(cl == "neutral"), 3), corpus_positive = round(mean(cl == "positive"), 3),
         corpus_negative = round(mean(cl == "negative"), 3))
}))
thresholds <- thresholds |> mutate(reference_neutral = round(mean(r$reference == "neutral"), 3))
check("threshold 0.05 reproduces post_sentiment_label", all(label_at(r$score, 0.05) == r$post_sentiment_label))

# ---------------------------------------------------------------------------
# 4. Checks and outputs
# ---------------------------------------------------------------------------
check("confusion matrix totals the reference posts", sum(conf[, -1]) == nrow(r), sprintf("%d", nrow(r)))
check("every reference label is one of the three classes", all(r$reference %in% CLASSES))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
write_csv(agree, file.path(OUT_DIR, "sentiment_validation_agreement.csv"), na = "")
write_csv(conf, file.path(OUT_DIR, "sentiment_validation_confusion.csv"), na = "")
write_csv(by_class, file.path(OUT_DIR, "sentiment_validation_by_class.csv"), na = "")
write_csv(strata, file.path(OUT_DIR, "sentiment_validation_by_stratum.csv"), na = "")
write_csv(thresholds, file.path(OUT_DIR, "sentiment_validation_thresholds.csv"), na = "")
print(as.data.frame(agree)); print(as.data.frame(conf)); print(as.data.frame(by_class))
print(as.data.frame(thresholds |> select(threshold, agreement, kappa, sample_neutral, corpus_neutral, corpus_positive, corpus_negative)))
if (!all(unlist(results))) stop("a check failed")
log_msg("written to %s", OUT_DIR)
