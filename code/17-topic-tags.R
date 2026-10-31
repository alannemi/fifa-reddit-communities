##############################################################################
# FIFA COMMUNITIES CHAPTER
## 17. TOPIC TAGS  (the analytical status of every topic from the co-authors' QUALTAG coding)
##############################################################################
#
# PURPOSE
# -------
# One table that says, for every topic of the nine period bins, whether it is political, non-political or excluded
# from the political aggregates, and which of the four analytical categories it carries — derived from the
# co-authors' review files (results/analysis/manual review/BIN<N>_topics_labelled_*.csv) by the rule Alan Nemirovski
# gave on 2026-10-31 (decisions V2, V3, V9, V14; docs/session-14_2026-10-31/00_approvals.md).  The review files and
# their qt_ columns (13-recode-qualtag.R) are read only; nothing is written back to them.  18-chapter-figures.R
# aggregates posts across topics from this table.
#
# THE RULE (V3)
# ------------
# Categories: Host Country Domestic -> cat_domestic ("domestic politics"); IR -> cat_ir ("international relations");
# FIFA Body -> cat_fifa ("FIFA governance"); National Identity -> cat_identity ("identity & nationalism").
#   * TP's files (odd bins; the QUALTAG text re-parsed with code/R/qualtag-utils.R):
#       any category tag            -> political  (also when "non-political" or REVIEW stands beside it)
#       no category tag, REVIEW     -> excluded   (out of the political and the non-political aggregates)
#       no category tag, no REVIEW  -> non-political (including a blank QUALTAG and tags such as "WC; player")
#   * BA's files (even bins; the qt_ columns typed in directly, 1 / 0 / blank):
#       any category = 1            -> political
#       qt_political = 1, no category -> political, uncategorised = 1 (counts in the political share, in no category)
#       qt_review = 1, no category, qt_political not 1 -> excluded
#       otherwise (blank or 0)      -> non-political (V9: BA's blank reads as non-political)
# The qualifier tags (WC, Club, Intl non-WC, player, humour) are carried for consultation only and never enter an
# aggregate (V14).  Posts are not touched here: 18-chapter-figures.R applies decision M1's weights to them.
#
# HOW TO RUN (repository root)
# ----------------------------
#   Rscript code/17-topic-tags.R          # seconds
#   TAGS_REVIEW_DIR=<folder> TAGS_OUT_DIR=<folder> TAGS_LABELS=<file> redirect the inputs/outputs for a test.
# The script stops when a bin has two files (remove the superseded one with git rm) or a tag lies outside the scheme;
# it warns when a bin has no file (its topics are absent from the table; script 18 treats their posts as uncoded).
#
# OUTPUTS  (results/analysis/)
# -------
#   topic_tags.csv          one row per topic of every coded bin: topic_uid, period_bin, topic_id, name, tone_category,
#                           coder (TP/BA), file, status (political / non-political / excluded), cat_domestic, cat_ir,
#                           cat_fifa, cat_identity (1/0), n_categories, uncategorised (1/0), review (1/0), wc, club,
#                           intl_non_wc, player, humour (1/0/blank; referential), QUALTAG (TP's text; blank for BA)
#   topic_tags_changes.csv  every topic whose status differs from the file's own summary (TP's qt_political yes / no /
#                           review / blank; BA's 1 / 0 / blank, blank = non-political), and BA's political topics without
#                           a category: topic_uid, coder, name, QUALTAG, file_political, file_status, new_status, reason
#   topic_tags_summary.csv  per bin and overall: topics, political (categorised / uncategorised), non-political,
#                           excluded, topics per category, changes
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(tidyr) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                                          # log_msg, check, read_chr, find_root
source(file.path(dirname(.utils), "qualtag-utils.R"))   # the scheme (POLITICAL_TAGS, TAG_COL, NONPOL, ...) and parse_tags()
ROOT <- find_root()
setwd(ROOT)

REVIEW_DIR <- Sys.getenv("TAGS_REVIEW_DIR", "results/analysis/manual review")
OUT_DIR    <- Sys.getenv("TAGS_OUT_DIR", "results/analysis")
LABELS     <- Sys.getenv("TAGS_LABELS", "results/analysis/topic_labels_all_bins.csv")
FILE_RE    <- "^BIN(\\d)_topics_labelled_.*\\.csv$"
# The four analytical categories: tag (as parse_tags reads it) -> column of this table; and the qt_ column BA types
CATS   <- c("host country domestic" = "cat_domestic", "ir" = "cat_ir", "fifa body" = "cat_fifa", "national identity" = "cat_identity")
CAT_QT <- setNames(unname(POLITICAL_TAGS[names(CATS)]), unname(CATS))      # cat_domestic -> qt_host_country_domestic, ...
# The referential qualifiers (V14): tag -> column; and BA's qt_ column
QUALS   <- c("wc" = "wc", "club" = "club", "intl non-wc" = "intl_non_wc", "player" = "player", "humour" = "humour")
QUAL_QT <- setNames(unname(TAG_COL[names(QUALS)]), unname(QUALS))          # wc -> qt_wc, ...
STATUS  <- c("political", "non-political", "excluded")
as01 <- function(x) as.character(as.integer(x))

# ---------------------------------------------------------------------------
# 1. The files: one per bin; the labelled topics as the spine
# ---------------------------------------------------------------------------
files <- list.files(REVIEW_DIR, pattern = FILE_RE)
bins  <- as.integer(sub(FILE_RE, "\\1", files))
if (anyDuplicated(bins)) stop("more than one file for bin(s) ", paste(unique(bins[duplicated(bins)]), collapse = ", "),
                              ": remove the superseded version first")
files <- files[order(bins)]; bins <- sort(bins)
log_msg("%d review files: %s", length(files), paste(files, collapse = ", "))
missing <- setdiff(1:9, bins)
if (length(missing)) log_msg("WARNING: no review file for bin(s) %s — their topics are absent from topic_tags.csv; 18-chapter-figures.R treats their posts as uncoded",
                             paste(missing, collapse = ", "))
labels <- read_chr(LABELS, col_select = c("topic_uid", "period_bin", "topic_id", "name", "tone_category")) |>
  mutate(period_bin = as.integer(period_bin))

# ---------------------------------------------------------------------------
# 2. One file -> the topics' status, categories, qualifiers, and the reason when the file's own summary differs
# ---------------------------------------------------------------------------
# The file's own summary, read as a status for the comparison: TP's qt_political yes / no / review / blank (unsettled);
# BA's 1 / 0 / blank.  The comparison never changes anything in the files.
file_status <- function(x, coder) {
  if (coder == "TP") c(yes = "political", no = "non-political", review = "excluded")[x] |> coalesce("unsettled")
  else c("1" = "political", "0" = "non-political")[x] |> coalesce("non-political")   # V9: BA's blank = non-political
}

tag_file <- function(f, b) {
  d <- read_chr(file.path(REVIEW_DIR, f))
  if (anyDuplicated(d$topic_id)) stop(f, ": duplicated topic_id")
  coder <- if ("QUALTAG" %in% names(d)) "TP" else "BA"
  n <- nrow(d)
  if (coder == "TP") {
    tags <- lapply(d$QUALTAG, parse_tags)
    unknown <- setdiff(unlist(tags), c(names(TAG_COL), NONPOL))
    if (length(unknown)) stop(f, ": tags outside the scheme: ", paste(unknown, collapse = ", "))
    has <- function(t) vapply(tags, function(x) t %in% x, logical(1))
    cat_m <- sapply(names(CATS), has); colnames(cat_m) <- unname(CATS)
    if (n == 1) cat_m <- matrix(cat_m, nrow = 1, dimnames = list(NULL, unname(CATS)))
    review <- has("review"); nonpol <- has(NONPOL); blank <- lengths(tags) == 0
    any_cat <- rowSums(cat_m) > 0
    status <- ifelse(any_cat, "political", ifelse(review, "excluded", "non-political"))
    uncat  <- rep(FALSE, n)
    qual_m <- sapply(names(QUALS), has); colnames(qual_m) <- unname(QUALS)
    if (n == 1) qual_m <- matrix(qual_m, nrow = 1, dimnames = list(NULL, unname(QUALS)))
    qual_df <- as_tibble(apply(qual_m, 2, as01))
    reason <- case_when(any_cat & nonpol  ~ "category tag beside non-political -> political",
                        any_cat & review  ~ "REVIEW with a category tag -> political",
                        blank             ~ "blank QUALTAG -> non-political",
                        !any_cat & !review & !nonpol ~ "no category tag, no non-political tag -> non-political",
                        TRUE ~ "")
    fpol <- if ("qt_political" %in% names(d)) d$qt_political else rep("", n)
    qualtag <- d$QUALTAG
  } else {
    need <- c("qt_political", "qt_review", unname(CAT_QT), unname(QUAL_QT))
    miss <- setdiff(need, names(d)); if (length(miss)) stop(f, ": BA file without the columns ", paste(miss, collapse = ", "))
    bad <- sapply(need, function(c) any(!trimws(d[[c]]) %in% c("", "0", "1")))
    if (any(bad)) stop(f, ": cells outside 1 / 0 / blank in ", paste(need[bad], collapse = ", "))
    v <- function(c) trimws(d[[c]]) == "1"
    cat_m <- sapply(CAT_QT, v); colnames(cat_m) <- names(CAT_QT)
    if (n == 1) cat_m <- matrix(cat_m, nrow = 1, dimnames = list(NULL, names(CAT_QT)))
    any_cat <- rowSums(cat_m) > 0; pol1 <- v("qt_political"); review <- v("qt_review")
    status <- ifelse(any_cat | pol1, "political", ifelse(review, "excluded", "non-political"))
    uncat  <- pol1 & !any_cat
    qual_df <- as_tibble(lapply(QUAL_QT, function(c) trimws(d[[c]])))        # 1 / 0 / blank as typed
    reason <- case_when(any_cat & !pol1 ~ "category tag without qt_political = 1 -> political",
                        uncat           ~ "qt_political = 1 without a category -> political, uncategorised",
                        review & !any_cat & !pol1 ~ "qt_review = 1 without a category -> excluded",
                        TRUE ~ "")
    fpol <- trimws(d$qt_political)
    qualtag <- rep("", n)
  }
  fstat <- file_status(fpol, coder)                # before the tibble: inside it `coder` would be the new column
  tibble(period_bin = b, topic_id = d$topic_id, topic_uid = paste(b, d$topic_id, sep = "-"), coder = coder, file = f,
         status = status, as_tibble(apply(cat_m, 2, as01)), n_categories = rowSums(cat_m),
         uncategorised = as01(uncat), review = as01(review), qual_df, QUALTAG = qualtag,
         file_political = fpol, file_status = fstat, reason = reason)
}

tags <- bind_rows(Map(tag_file, files, bins))

# ---------------------------------------------------------------------------
# 3. Checks, the three tables
# ---------------------------------------------------------------------------
for (i in seq_along(bins)) {
  b <- bins[i]; spine <- labels$topic_uid[labels$period_bin == b]; got <- tags$topic_uid[tags$period_bin == b]
  check(sprintf("bin %d: the file covers the bin's labelled topics exactly (%s)", b, files[i]),
        setequal(spine, got) && !anyDuplicated(got), sprintf("%d topics", length(got)))
}
check("every status is political / non-political / excluded", all(tags$status %in% STATUS))
check("every categorised political topic carries at least one category; no other topic does",
      all((tags$status == "political" & tags$uncategorised == "0") == (tags$n_categories > 0)))
check("uncategorised political topics come from BA's files only", all(tags$coder[tags$uncategorised == "1"] == "BA"))

tags <- tags |> left_join(labels |> select(topic_uid, name, tone_category), by = "topic_uid") |>
  select(topic_uid, period_bin, topic_id, name, tone_category, coder, file, status, all_of(unname(CATS)), n_categories,
         uncategorised, review, all_of(unname(QUALS)), QUALTAG, file_political, file_status, reason) |>
  arrange(period_bin, as.integer(topic_id))

changes <- tags |> filter(status != file_status | reason != "") |>
  transmute(topic_uid, period_bin, coder, name, QUALTAG, file_political, file_status, new_status = status, reason)

summ <- tags |> group_by(period_bin, coder, file) |>
  summarise(topics = n(), political = sum(status == "political"),
            political_categorised = sum(status == "political" & uncategorised == "0"),
            political_uncategorised = sum(uncategorised == "1"),
            non_political = sum(status == "non-political"), excluded = sum(status == "excluded"),
            across(all_of(unname(CATS)), ~ sum(.x == "1")),
            changes = sum(status != file_status | reason != ""), .groups = "drop")
summ <- bind_rows(summ, summ |> summarise(period_bin = NA_integer_, coder = "all", file = "", across(where(is.numeric), sum)))
print(as.data.frame(summ |> select(-file)), row.names = FALSE)

dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
write_csv(tags |> select(-file_political, -file_status, -reason), file.path(OUT_DIR, "topic_tags.csv"), na = "")
write_csv(changes, file.path(OUT_DIR, "topic_tags_changes.csv"), na = "")
write_csv(summ, file.path(OUT_DIR, "topic_tags_summary.csv"), na = "")
log_msg("%s: topic_tags.csv (%d topics, %d bins), topic_tags_changes.csv (%d), topic_tags_summary.csv", OUT_DIR, nrow(tags), length(bins), nrow(changes))

if (!all(unlist(results))) stop("a check failed: ", paste(names(results)[!unlist(results)], collapse = ", "))
cat("ALL CHECKS PASSED\n")
