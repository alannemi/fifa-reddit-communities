##############################################################################
# FIFA COMMUNITIES CHAPTER
## 6. TAG THE CORPUS  (stage S6 of the analytical pipeline — append the analysis variables)
##############################################################################
#
# PURPOSE
# -------
# Append the eight analysis variables of codebook §4.9 (period_bin, period_type,
# topic_id, topic_assignment, category_id, post_sentiment, post_sentiment_full,
# post_sentiment_label) to the clean master `data/soccer_data.clean.csv` and to
# the 17 yearly files, from the join table `data/analysis/post_variables.csv`
# written by stage S5 of code/05-analysis-pipeline.py.  Side files are not
# analysed and stay untouched (43 columns).  Design: docs/session-04_2026-10-22/
# 01_analytical-pipeline-design.md §9.2; approvals: docs/session-05_2026-10-23/.
#
# WHY IN R
# --------
# The data files have exactly one writer and one format (readr::write_csv,
# UTF-8, RFC-4180 quoting, empty cell = missing) since v2.0.  Writing them from
# the same function keeps every existing column byte-identical; the validation
# script (07-validate-analysis.R) proves it with per-column fingerprints taken here before writing.
#
# DESIGN PRINCIPLES
# -----------------
# 1. Fidelity: every column is read as character and written back verbatim;
#    the new columns are appended after the existing 42, in codebook order.
# 2. Gated: the script refuses to write unless post_variables.json says the
#    collection is complete (all nine bins modelled, labelled and grouped;
#    sentiment for every post).  A partial table is accepted only for dry runs
#    (TAG_ALLOW_PARTIAL=1) and only into a different output root.
# 3. Nothing is lost: a per-column xxhash64 fingerprint of the 42 original
#    columns of every file is stored in data/data artefacts/tagging_fingerprint.json
#    before the files are rewritten; 07-validate-analysis.R recomputes it.
# 4. Memory: the master (1.9 GB) is processed first and released; the yearly
#    files follow one at a time (review finding R12).
#
# HOW TO RUN (repository root)
# ----------------------------
#   Rscript code/06-tag-corpus.R                 # in place; needs a complete post_variables.csv
#   TAG_IN_ROOT=<root> TAG_OUT_ROOT=<other root> TAG_ALLOW_PARTIAL=1 Rscript code/06-tag-corpus.R   # dry run
# Environment variables: TAG_IN_ROOT (default: the repository root) holds data/soccer_data.clean.csv,
# data/yearly datasets/ and data/analysis/post_variables.csv; TAG_OUT_ROOT (default = TAG_IN_ROOT) receives
# the tagged files and the artefacts; TAG_ALLOW_PARTIAL=1 accepts an incomplete table (dry runs only).
# Shared helpers and the column specifications: code/R/pipeline-utils.R.
#
# OUTPUTS
# -------
#   <out>/data/soccer_data.clean.csv, <out>/data/yearly datasets/soccer.data.cleaned-YYYY.csv   (50 columns)
#   <out>/data/data artefacts/tagging_fingerprint.json   per file: n_rows, per-column hashes of the 42 columns
#   <out>/data/data artefacts/tagging_summary.csv        per file: rows, posts with topic / category / sentiment
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(jsonlite); library(digest) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                     # shared helpers and column specifications (code/R/pipeline-utils.R)
ROOT <- find_root()
setwd(ROOT)
IN_ROOT  <- Sys.getenv("TAG_IN_ROOT", ROOT)
OUT_ROOT <- Sys.getenv("TAG_OUT_ROOT", IN_ROOT)
ALLOW_PARTIAL <- Sys.getenv("TAG_ALLOW_PARTIAL", "0") == "1"
in_place <- normalizePath(IN_ROOT, mustWork = TRUE) == normalizePath(OUT_ROOT, mustWork = FALSE)

# ---------------------------------------------------------------------------
# 1. Pre-flight: the join table and its sidecar
# ---------------------------------------------------------------------------
pv_path <- file.path(IN_ROOT, "data/analysis/post_variables.csv")
side_path <- file.path(IN_ROOT, "data/analysis/post_variables.json")
if (!file.exists(pv_path)) stop("missing ", pv_path, " — run `--stage collect` first")
side <- if (file.exists(side_path)) fromJSON(side_path) else list(complete = FALSE)
if (!isTRUE(side$complete)) {
  if (!ALLOW_PARTIAL) stop("post_variables.json says the collection is not complete (bins complete: ",
                           paste(side$bins_complete, collapse = ","), "); refusing to tag. Set TAG_ALLOW_PARTIAL=1 for a dry run.")
  if (in_place) stop("a partial table may only be written to a different TAG_OUT_ROOT (dry run), never in place")
  log_msg("WARNING: partial table accepted (dry run): bins complete = %s", paste(side$bins_complete, collapse = ","))
}
pv <- read_chr(pv_path)
if (!all(ANALYSIS_COLS %in% names(pv))) stop("post_variables.csv lacks: ", paste(setdiff(ANALYSIS_COLS, names(pv)), collapse = ", "))
if (any(duplicated(pv$id))) stop("duplicate ids in post_variables.csv")
pv <- pv[, c("id", ANALYSIS_COLS)]
log_msg("post_variables.csv: %s rows, taxonomy version %s", format(nrow(pv), big.mark = ","), side$taxonomy_version %||% "NA")

for (d in c("data", "data/yearly datasets", "data/data artefacts"))
  dir.create(file.path(OUT_ROOT, d), recursive = TRUE, showWarnings = FALSE)

fingerprint <- list(); summary_rows <- list()

# ---------------------------------------------------------------------------
# 2. One file at a time: fingerprint, join, append, write
# ---------------------------------------------------------------------------
tag_file <- function(in_path, out_path, label) {
  d <- read_chr(in_path)
  if (!identical(names(d), ORIGINAL_COLS)) stop(label, ": columns differ from the v2.1 spec (", ncol(d), " columns; already tagged?)")
  if (any(ANALYSIS_COLS %in% names(d))) stop(label, ": analysis columns already present")
  missing_ids <- setdiff(d$id, pv$id)
  if (length(missing_ids)) stop(label, ": ", length(missing_ids), " ids have no row in post_variables.csv")
  fp <- list(n_rows = nrow(d), columns = setNames(lapply(d, col_hash), names(d)))
  tagged <- left_join(d, pv, by = "id")                      # keeps the row order of d
  if (nrow(tagged) != nrow(d)) stop(label, ": join changed the row count")
  if (!identical(tagged$id, d$id)) stop(label, ": join changed the row order")
  write_csv(tagged, out_path, na = "", progress = FALSE)
  summary_rows[[label]] <<- data.frame(file = label, n_rows = nrow(tagged),
                                        n_with_topic = sum(tagged$topic_id != ""), n_with_category = sum(tagged$category_id != ""),
                                        n_with_sentiment = sum(tagged$post_sentiment != ""), n_bins = length(unique(tagged$period_bin)))
  fingerprint[[label]] <<- fp
  log_msg("%-32s %s rows -> %s (topic %s, category %s, sentiment %s)", label, format(nrow(tagged), big.mark = ","), out_path,
          summary_rows[[label]]$n_with_topic, summary_rows[[label]]$n_with_category, summary_rows[[label]]$n_with_sentiment)
  rm(d, tagged); invisible(gc())
}

master_in <- file.path(IN_ROOT, "data/soccer_data.clean.csv")
if (file.exists(master_in)) tag_file(master_in, file.path(OUT_ROOT, "data/soccer_data.clean.csv"), "soccer_data.clean.csv") else
  log_msg("no master at %s (dry run on yearly files only)", master_in)

yearly <- sort(list.files(file.path(IN_ROOT, "data/yearly datasets"), pattern = "^soccer\\.data\\.cleaned-[0-9]{4}\\.csv$", full.names = TRUE))
if (!length(yearly)) stop("no yearly files under ", file.path(IN_ROOT, "data/yearly datasets"))
for (f in yearly) tag_file(f, file.path(OUT_ROOT, "data/yearly datasets", basename(f)), basename(f))

# ---------------------------------------------------------------------------
# 3. Artefacts
# ---------------------------------------------------------------------------
write_json(list(written = format(Sys.time(), "%Y-%m-%d %H:%M:%S"), in_root = IN_ROOT, out_root = OUT_ROOT,
                partial = !isTRUE(side$complete), taxonomy_version = side$taxonomy_version, analysis_columns = ANALYSIS_COLS,
                files = fingerprint),
           file.path(OUT_ROOT, "data/data artefacts/tagging_fingerprint.json"), auto_unbox = TRUE, pretty = TRUE)
write_csv(bind_rows(summary_rows), file.path(OUT_ROOT, "data/data artefacts/tagging_summary.csv"))
log_msg("done: %d file(s) tagged; fingerprint and summary in %s", length(fingerprint), file.path(OUT_ROOT, "data/data artefacts"))
