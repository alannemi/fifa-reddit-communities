##############################################################################
# FIFA COMMUNITIES CHAPTER
## 8. VALIDATE THE ANALYSIS VARIABLES  (run after 07-tag-corpus.R)
##############################################################################
#
# PURPOSE
# -------
# Independent checks that the tagged files (codebook v3.0: 42 original + 8
# analysis columns) are complete, internally consistent, consistent with the
# pipeline's own tables, and that the 42 original columns are byte-identical to
# what they were before tagging.  Every check prints PASS/FAIL; the script
# stops with an error at the end if any check failed.  Nothing is modified.
#
# CHECKS (design §9.2)
# --------------------
#  A1  row counts unchanged (per file, against the fingerprint taken by 07)
#  A2  column specification: master and yearly files = 42 + 8 columns in codebook order; side files 43
#  A3  period_bin matches the UTC calendar date of created_datetime (bin table of design §4)
#  A4  every post has one bin in 1..9; bin sizes sum to the file; period_type consistent with the bin
#  A5  topic_id exists in the bin's topics.csv (after overrides); -1 = outlier; empty = not modelled
#  A6  category_id exists in the bin's categories.csv (after overrides); empty iff topic empty or -1
#  A7  topic_assignment consistent with topic_id (outlier <-> -1, not_modelled <-> empty, else cluster/reassigned)
#  A8  sentiment scores within [-1, 1] and label consistent with the +/-0.05 thresholds
#  A9  the 42 original columns are unchanged (per-column xxhash64 equals the pre-write fingerprint)
#  A10 master and post_variables.csv identical on the 8 columns
#  A11 yearly files equal the master's year subset on every column (per-column hashes)
#  A12 the collection was complete (post_variables.json) — reported, not failed, for dry runs
#
# HOW TO RUN (repository root):  Rscript code/08-validate-analysis.R
# Environment: TAG_OUT_ROOT (default ".") = where the tagged files and the fingerprint are;
#              TAG_IN_ROOT (default TAG_OUT_ROOT) = where data/analysis/post_variables.csv is;
#              ANALYSIS_DIR (default <TAG_IN_ROOT>/results/analysis) = the per-bin pipeline folders.
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(jsonlite); library(digest) })

OUT_ROOT <- Sys.getenv("TAG_OUT_ROOT", ".")
IN_ROOT  <- Sys.getenv("TAG_IN_ROOT", OUT_ROOT)
ANALYSIS_DIR <- Sys.getenv("ANALYSIS_DIR", file.path(IN_ROOT, "results/analysis"))
ANALYSIS_COLS <- c("period_bin", "period_type", "topic_id", "topic_assignment", "category_id",
                   "post_sentiment", "post_sentiment_full", "post_sentiment_label")
N_ORIGINAL <- 42

results <- list()
check <- function(name, ok, detail = "") {
  results[[name]] <<- ok
  cat(sprintf("[%s] %-64s %s\n", if (isTRUE(ok)) "PASS" else "FAIL", name, detail))
}
read_chr <- function(p) read_csv(p, col_types = cols(.default = col_character()), na = character(), progress = FALSE, lazy = FALSE)
col_hash <- function(x) digest(as.character(x), algo = "xxhash64", serialize = TRUE)

# the bin table (design §4) — kept in sync with code/topicmodel/config.py PERIOD_BINS
BIN_START <- as.Date(c("2010-01-01", "2010-10-01", "2014-02-01", "2014-10-01", "2018-02-01", "2018-10-01", "2022-07-01", "2023-03-01", "2026-02-01"))
BIN_END   <- as.Date(c("2010-09-30", "2014-01-31", "2014-09-30", "2018-01-31", "2018-09-30", "2022-06-30", "2023-02-28", "2026-01-31", "2026-07-31"))
BIN_TYPE  <- c("tournament", "interim", "tournament", "interim", "tournament", "interim", "tournament", "interim", "tournament")
bin_of <- function(dt) { d <- as.Date(substr(dt, 1, 10)); b <- findInterval(d, BIN_START); b[d > BIN_END[pmax(b, 1)] | b < 1] <- NA; b }

# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------
fp_path <- file.path(OUT_ROOT, "data/data artefacts/tagging_fingerprint.json")
if (!file.exists(fp_path)) stop("missing ", fp_path, " — run 07-tag-corpus.R first")
fp <- fromJSON(fp_path, simplifyVector = FALSE)
pv <- read_chr(file.path(IN_ROOT, "data/analysis/post_variables.csv"))
side_path <- file.path(IN_ROOT, "data/analysis/post_variables.json")
side <- if (file.exists(side_path)) fromJSON(side_path) else list(complete = NA)

master_path <- file.path(OUT_ROOT, "data/soccer_data.clean.csv")
yearly <- sort(list.files(file.path(OUT_ROOT, "data/yearly datasets"), pattern = "^soccer\\.data\\.cleaned-[0-9]{4}\\.csv$", full.names = TRUE))
has_master <- file.exists(master_path)
m <- if (has_master) read_chr(master_path) else bind_rows(lapply(yearly, read_chr))   # dry runs may have yearly files only
label <- if (has_master) "soccer_data.clean.csv" else "yearly files (no master)"
cat(sprintf("Loaded %s: %s rows, %d columns; post_variables: %s rows\n", label, format(nrow(m), big.mark = ","), ncol(m), format(nrow(pv), big.mark = ",")))

# per-bin pipeline tables (after overrides, as S5 applies them)
read_opt <- function(p) if (file.exists(p)) read_chr(p) else NULL
bin_tables <- lapply(1:9, function(b) {
  d <- file.path(ANALYSIS_DIR, sprintf("bin-%d", b))
  tp <- read_opt(file.path(d, "topics.csv")); ov <- read_opt(file.path(d, "topic_overrides.csv"))
  ct <- read_opt(file.path(d, "categories.csv")); cov <- read_opt(file.path(d, "category_overrides.csv"))
  topics <- if (is.null(tp)) NULL else {
    ids <- tp$topic_id
    if (!is.null(ov)) { mv <- ov[ov$merge_into != "", ]; ids <- setdiff(ids, mv$topic_id) }   # merged-away topics disappear
    ids
  }
  cats <- if (is.null(ct)) NULL else {
    ids <- ct$category_id
    if (!is.null(cov)) { mv <- cov[cov$merge_into != "", ]; ids <- setdiff(ids, mv$category_id) }
    ids
  }
  list(topics = topics, categories = cats)
})

# ---------------------------------------------------------------------------
# A1, A2
# ---------------------------------------------------------------------------
if (has_master) check("A1 master row count unchanged", nrow(m) == fp$files[["soccer_data.clean.csv"]]$n_rows, format(nrow(m), big.mark = ","))
yrows_ok <- all(vapply(yearly, function(f) nrow(read_chr(f)) == fp$files[[basename(f)]]$n_rows, logical(1)))
check("A1b yearly row counts unchanged", yrows_ok, sprintf("%d files", length(yearly)))
expected_cols <- c(names(fp$files[[1]]$columns), ANALYSIS_COLS)
check("A2a master/yearly = 42 + 8 columns in codebook order", identical(names(m), expected_cols) && ncol(m) == N_ORIGINAL + 8)
ycols_ok <- all(vapply(yearly, function(f) identical(names(read_csv(f, n_max = 0, col_types = cols(.default = col_character()), progress = FALSE)), expected_cols), logical(1)))
check("A2b yearly files share the 50-column spec", ycols_ok)
side_files <- file.path(OUT_ROOT, "data", c("deleted_posts.csv", "removed_posts.csv", "bot_threads.csv", "promotional_posts.csv", "encoding_issue_posts.csv", "non_english_posts.csv"))
if (all(file.exists(side_files))) {
  sok <- all(vapply(side_files, function(f) ncol(read_csv(f, n_max = 0, col_types = cols(.default = col_character()), progress = FALSE)) == N_ORIGINAL + 1, logical(1)))
  check("A2c side files untouched (43 columns)", sok)
} else cat("[SKIP] A2c side files not present in this root\n")

# ---------------------------------------------------------------------------
# A3, A4
# ---------------------------------------------------------------------------
b_calc <- bin_of(m$created_datetime)
b_file <- suppressWarnings(as.integer(m$period_bin))
check("A3 period_bin equals the bin of the UTC date", !anyNA(b_calc) && identical(b_calc, b_file))
check("A4a every post has one bin in 1..9", all(b_file %in% 1:9))
check("A4b bin sizes sum to the file", sum(table(b_file)) == nrow(m), paste(sprintf("%d:%s", 1:9, format(tabulate(b_file, 9), big.mark = ",")), collapse = " "))
check("A4c period_type consistent with the bin", all(m$period_type == BIN_TYPE[b_file]))

# ---------------------------------------------------------------------------
# A5–A8
# ---------------------------------------------------------------------------
tid <- m$topic_id; cid <- m$category_id; ass <- m$topic_assignment
a5 <- TRUE; a6 <- TRUE; a5_detail <- c()
for (b in 1:9) {
  rows <- which(b_file == b); if (!length(rows)) next
  bt <- bin_tables[[b]]
  t_b <- tid[rows]; c_b <- cid[rows]
  if (is.null(bt$topics)) { if (any(t_b != "")) { a5 <- FALSE; a5_detail <- c(a5_detail, sprintf("bin %d tagged without topics.csv", b)) }; next }
  bad <- setdiff(unique(t_b[t_b != ""]), bt$topics)
  if (length(bad)) { a5 <- FALSE; a5_detail <- c(a5_detail, sprintf("bin %d: %d unknown topic ids", b, length(bad))) }
  if (is.null(bt$categories)) { if (any(c_b != "")) a6 <- FALSE; next }
  badc <- setdiff(unique(c_b[c_b != ""]), bt$categories)
  if (length(badc)) a6 <- FALSE
  if (any(c_b != "" & (t_b == "" | t_b == "-1"))) a6 <- FALSE
}
check("A5 topic_id valid for its bin", a5, paste(a5_detail, collapse = "; "))
check("A6 category_id valid for its bin and empty for outliers / not modelled", a6)
check("A7 topic_assignment consistent with topic_id",
      all((ass == "outlier") == (tid == "-1")) && all((ass == "not_modelled") == (tid == "")) &&
        all(ass %in% c("cluster", "reassigned", "outlier", "not_modelled")))
s <- suppressWarnings(as.numeric(m$post_sentiment)); sf <- suppressWarnings(as.numeric(m$post_sentiment_full)); lab <- m$post_sentiment_label
has_s <- m$post_sentiment != ""
check("A8a sentiment within [-1, 1]", all(s[has_s] >= -1 & s[has_s] <= 1) && all(sf[has_s] >= -1 & sf[has_s] <= 1), sprintf("%s scored", format(sum(has_s), big.mark = ",")))
lab_exp <- ifelse(s > 0.05, "positive", ifelse(s < -0.05, "negative", "neutral"))
check("A8b label consistent with the thresholds", all(lab[has_s] == lab_exp[has_s]) && all(lab[!has_s] == ""))

# ---------------------------------------------------------------------------
# A9–A12
# ---------------------------------------------------------------------------
if (has_master) {
  fpm <- fp$files[["soccer_data.clean.csv"]]$columns
  same <- vapply(names(fpm), function(cn) identical(col_hash(m[[cn]]), fpm[[cn]]), logical(1))
  check("A9 the 42 original columns are unchanged (master)", all(same), if (all(same)) "" else paste(names(fpm)[!same], collapse = ","))
}
j <- inner_join(m[, c("id", ANALYSIS_COLS)], pv[, c("id", ANALYSIS_COLS)], by = "id", suffix = c(".m", ".pv"))
same8 <- all(vapply(ANALYSIS_COLS, function(cn) identical(j[[paste0(cn, ".m")]], j[[paste0(cn, ".pv")]]), logical(1)))
check("A10 master and post_variables identical on the 8 columns", nrow(j) == nrow(m) && same8)
a11 <- TRUE; a9y <- TRUE
for (f in yearly) {
  y <- read_chr(f); yr <- sub(".*-([0-9]{4})\\.csv$", "\\1", f)
  fpy <- fp$files[[basename(f)]]$columns
  a9y <- a9y && all(vapply(names(fpy), function(cn) identical(col_hash(y[[cn]]), fpy[[cn]]), logical(1)))
  if (has_master) {
    sub <- m[m$year == yr, ]
    a11 <- a11 && nrow(sub) == nrow(y) && all(vapply(names(y), function(cn) identical(col_hash(y[[cn]]), col_hash(sub[[cn]])), logical(1)))
  }
  rm(y); invisible(gc())
}
check("A9b the 42 original columns are unchanged (yearly files)", a9y)
if (has_master) check("A11 yearly files equal the master's year subsets", a11)
if (isTRUE(side$complete)) check("A12 collection complete (all nine bins, sentiment for all)", TRUE) else
  cat(sprintf("[NOTE] A12 collection not complete (bins complete: %s) — acceptable for a dry run only\n", paste(side$bins_complete, collapse = ",")))

cat("\n")
if (!all(unlist(results))) stop("VALIDATION FAILED: ", paste(names(results)[!unlist(results)], collapse = ", "))
cat("ALL CHECKS PASSED\n")
