##############################################################################
# FIFA COMMUNITIES CHAPTER
## code/R/pipeline-utils.R — helpers and column specifications shared by the R stages
##############################################################################
#
# PURPOSE
# -------
# One definition each of what 02-data-cleaning.R, 03-data-validation.R, 06-tag-corpus.R and
# 07-validate-analysis.R used to carry separately (session 07, decision H4): logging, the PASS/FAIL check,
# the character-only CSV reader, the per-column fingerprint, locating the repository root, the 42-column
# specification of the cleaned files (codebook v2.1), the 8 analysis columns (codebook §4.9) and the
# period-bin table (design §4).
#
# HOW TO USE — the first lines of each script; works for `Rscript code/<script>.R` run from the repository
# root and for source() in RStudio with fifa-reddit-communities.Rproj open:
#   .utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
#                     if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
#   source(.utils)
#   ROOT <- find_root(); setwd(ROOT)
#
# CONVENTIONS (memory/workflows.md §5): every CSV is read as character (col_types = cols(.default =
# col_character()), na = character()) and written with readr::write_csv(na = ""); nothing is discarded silently.
##############################################################################

suppressPackageStartupMessages({ library(readr); library(digest) })

# ---------------------------------------------------------------------------
# Logging and checks
# ---------------------------------------------------------------------------
log_msg <- function(...) cat(format(Sys.time(), "%H:%M:%S"), "|", sprintf(...), "\n")

results <- list()                                  # filled by check(); the calling script stops on any FALSE
check <- function(name, ok, detail = "") {
  results[[name]] <<- ok
  cat(sprintf("[%s] %-64s %s\n", if (isTRUE(ok)) "PASS" else "FAIL", name, detail))
}

# ---------------------------------------------------------------------------
# Reading and fingerprinting
# ---------------------------------------------------------------------------
read_chr <- function(p, ...) read_csv(p, col_types = cols(.default = col_character()), na = character(),
                                      progress = FALSE, lazy = FALSE, ...)
col_hash <- function(x) digest(as.character(x), algo = "xxhash64", serialize = TRUE)

# ---------------------------------------------------------------------------
# The repository root: the working directory first (Rscript from the root; the .Rproj in RStudio), then the
# running script's own location (code/<script>.R -> ..).  No machine-specific fallback: the repository is
# wherever it was cloned, and the data folder is copied into it (data/README.md).
# ---------------------------------------------------------------------------
find_root <- function(sentinel = "code/R/pipeline-utils.R") {
  if (file.exists(sentinel)) return(normalizePath("."))
  a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
  if (length(f) == 1) {
    cand <- normalizePath(file.path(dirname(f), ".."), mustWork = FALSE)
    if (file.exists(file.path(cand, sentinel))) return(cand)
  }
  stop("repository root not found: run `Rscript code/<script>.R` from the repository root, or open fifa-reddit-communities.Rproj")
}

# ---------------------------------------------------------------------------
# Column specifications
# ---------------------------------------------------------------------------
# The 42 columns of the clean master and the yearly files (codebook v2.1 §4; side files add filter_reason).
ORIGINAL_COLS <- c("id","name","title","selftext","author","author_flair_text","author_flair_css_class",
  "author_flair_background_color","author_flair_text_color","author_flair_type","author_flair_richtext",
  "author_flair_template_id","link_flair_text","link_flair_css_class","link_flair_background_color",
  "link_flair_text_color","link_flair_type","link_flair_richtext","link_flair_template_id","score","ups",
  "downs","num_comments","num_crossposts","total_awards_received","gilded","created_datetime","edited",
  "domain","url","permalink","is_self","is_video","is_reddit_media_domain","post_hint","removed_by_category",
  "subreddit_subscribers","media","preview","year","post_type","lang")
N_ORIGINAL <- length(ORIGINAL_COLS)                # 42
# The 8 analysis columns appended by stage S6 (codebook §4.9, in order).
ANALYSIS_COLS <- c("period_bin", "period_type", "topic_id", "topic_assignment", "category_id",
                   "post_sentiment", "post_sentiment_full", "post_sentiment_label")

# ---------------------------------------------------------------------------
# Period bins (design §4).  Mirrored from code/topicmodel/config.py PERIOD_BINS so that R checks stage S0
# independently of the pipeline; 07-validate-analysis.R also compares this table with the pipeline's own
# data/analysis/period_bins_summary.csv when that file is present (check A0).
# ---------------------------------------------------------------------------
BIN_START <- as.Date(c("2010-01-01", "2010-10-01", "2014-02-01", "2014-10-01", "2018-02-01", "2018-10-01", "2022-07-01", "2023-03-01", "2026-02-01"))
BIN_END   <- as.Date(c("2010-09-30", "2014-01-31", "2014-09-30", "2018-01-31", "2018-09-30", "2022-06-30", "2023-02-28", "2026-01-31", "2026-07-31"))
BIN_TYPE  <- c("tournament", "interim", "tournament", "interim", "tournament", "interim", "tournament", "interim", "tournament")
bin_of <- function(dt) { d <- as.Date(substr(dt, 1, 10)); b <- findInterval(d, BIN_START); b[d > BIN_END[pmax(b, 1)] | b < 1] <- NA; b }
