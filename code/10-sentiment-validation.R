##############################################################################
# FIFA COMMUNITIES CHAPTER
## 10. SENTIMENT VALIDATION  (agreement of two human coders with each other and with VADER)
##############################################################################
#
# PURPOSE
# -------
# Report how well `post_sentiment_label` (VADER compound of the title, cut at
# +/-0.05; design §8) matches human judgement, from the two coders' sheets
# produced by 09-sentiment-coding-pack.R.  Decision E1 (Alan Nemirovski,
# 2026-10-27).  The figures go into codebook §8.2 and limitation 10.
#
# WHAT IS COMPUTED
# ----------------
# 1. Coder A vs coder B: percent agreement and Cohen's kappa (with a bootstrap
#    95 % interval) — the reliability of the human reference.
# 2. Each coder vs VADER: percent agreement and kappa.
# 3. Reference vs VADER: the reference label is the coders' common code; where
#    they disagree, the code in `adjudicated.csv` (id, code) if that file
#    exists, otherwise the post is left out and counted.  Confusion matrix,
#    precision / recall / F1 per class, macro-F1, and agreement by period bin
#    and by is_self (the sample's strata; small cells — read the n).
# Kappa is computed here (no irr dependency): kappa = (p_o - p_e) / (1 - p_e),
# p_e from the two raters' marginal shares over the three classes.
#
# HOW TO RUN (repository root; after both coders have returned their sheets)
# ----------------------------
#   Rscript code/10-sentiment-validation.R
# Environment variables (tests): SENT_CODING_DIR (default results/analysis/manual review/sentiment_coding)
# holds coder_A.csv, coder_B.csv and the optional adjudicated.csv; SENT_OUT_DIR (default results/descriptives).
# Codes accepted (case-insensitive): positive / neutral / negative, pos / neu / neg, 1 / 0 / -1.
#
# OUTPUTS  (SENT_OUT_DIR)
# -------
#   sentiment_validation_agreement.csv   one row per comparison: n, agreement, kappa, 95 % interval
#   sentiment_validation_confusion.csv   reference x VADER counts
#   sentiment_validation_by_class.csv    precision, recall, F1 per class (VADER against the reference)
#   sentiment_validation_by_stratum.csv  agreement with VADER by period_bin and by is_self
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                     # shared helpers (code/R/pipeline-utils.R)
ROOT <- find_root()
setwd(ROOT)

CODING_DIR <- Sys.getenv("SENT_CODING_DIR", "results/analysis/manual review/sentiment_coding")
OUT_DIR    <- Sys.getenv("SENT_OUT_DIR", "results/descriptives")
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
  set.seed(SEED)
  boot <- replicate(N_BOOT, { i <- sample(length(a), replace = TRUE); kappa(a[i], b[i]) })
  tibble(comparison = name, n = length(a), agreement = round(mean(a == b), 4), kappa = round(kappa(a, b), 4),
         kappa_lo95 = round(unname(quantile(boot, 0.025, na.rm = TRUE)), 4),
         kappa_hi95 = round(unname(quantile(boot, 0.975, na.rm = TRUE)), 4))
}

read_sheet <- function(coder) {
  p <- file.path(CODING_DIR, sprintf("coder_%s.csv", coder))
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
d <- smp |> left_join(read_sheet("A"), by = "id") |> left_join(read_sheet("B"), by = "id")
stopifnot(nrow(d) == nrow(smp), !anyDuplicated(d$id))

adj_path <- file.path(CODING_DIR, "adjudicated.csv")
adj <- if (file.exists(adj_path)) read_chr(adj_path) |> transmute(id, code_adj = norm_code(code)) else tibble(id = character(), code_adj = character())
d <- d |> left_join(adj, by = "id") |>
  mutate(reference = case_when(!is.na(code_A) & code_A == code_B ~ code_A,
                               !is.na(code_adj) ~ code_adj,
                               TRUE ~ NA_character_))
n_dis <- sum(!is.na(d$code_A) & !is.na(d$code_B) & d$code_A != d$code_B)
log_msg("both coded: %d; disagreements: %d (%d adjudicated); reference label for %d posts",
        sum(!is.na(d$code_A) & !is.na(d$code_B)), n_dis, sum(!is.na(d$code_adj)), sum(!is.na(d$reference)))

# ---------------------------------------------------------------------------
# 2. Agreement and kappa
# ---------------------------------------------------------------------------
agree <- bind_rows(compare("coder_A vs coder_B", d$code_A, d$code_B),
                   compare("coder_A vs VADER", d$code_A, d$post_sentiment_label),
                   compare("coder_B vs VADER", d$code_B, d$post_sentiment_label),
                   compare("reference vs VADER", d$reference, d$post_sentiment_label))

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
# 4. Checks and outputs
# ---------------------------------------------------------------------------
check("confusion matrix totals the reference posts", sum(conf[, -1]) == nrow(r), sprintf("%d", nrow(r)))
check("every reference label is one of the three classes", all(r$reference %in% CLASSES))
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
write_csv(agree, file.path(OUT_DIR, "sentiment_validation_agreement.csv"), na = "")
write_csv(conf, file.path(OUT_DIR, "sentiment_validation_confusion.csv"), na = "")
write_csv(by_class, file.path(OUT_DIR, "sentiment_validation_by_class.csv"), na = "")
write_csv(strata, file.path(OUT_DIR, "sentiment_validation_by_stratum.csv"), na = "")
print(as.data.frame(agree)); print(as.data.frame(conf)); print(as.data.frame(by_class))
if (!all(unlist(results))) stop("a check failed")
log_msg("written to %s", OUT_DIR)
