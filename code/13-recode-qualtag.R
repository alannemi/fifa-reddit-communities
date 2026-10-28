##############################################################################
# FIFA COMMUNITIES CHAPTER
## 13. QUALTAG RECODING  (the co-authors' qualitative topic tags -> one variable per tag + a political summary)
##############################################################################
#
# PURPOSE
# -------
# Co-author TP (Teresa Pian) tags every topic of the tournament bins in a free-text column `QUALTAG` of her
# review files (results/analysis/manual review/BIN<N>_topics_labelled_*.csv; bin 1 with BA): values separated
# by ';', from the scheme "FIFA Qual Coding" that Alan Nemirovski provided on 2026-10-28.  This script recodes
# the column into separate variables and writes them into the same files, right after QUALTAG, which itself
# stays exactly as TP wrote it.
#
# THE SCHEME (Alan Nemirovski, 2026-10-28)
# ----------------------------------------
#   WC                                     the topic relates to the World Cup (political or not by the other tags)
#   Host Country Domestic, National Identity, FIFA Body, IR      the topic IS political
#   non-political                          the topic is NOT political
#   Club, Intl non-WC, player, humour      qualifiers: club teams; an international competition other than the
#                                          World Cup; a specific individual; humour / jokes
#   REVIEW                                 residual category, borderline political
#
# THE RULES  (decisions of Alan Nemirovski, 2026-10-28: docs/session-10_2026-10-28/00_approvals.md)
# ---------
# One indicator per tag (qt_wc ... qt_review; `non-political` feeds qt_political only) and the summary
# qt_political:
#   * no REVIEW:     a political tag only -> "yes"; non-political only -> "no"; both, or neither -> empty, and
#                    the topic goes on the list for TP (QUALTAG_to_settle.csv).  Indicators: 1 tagged, 0 not.
#   * REVIEW with other tags: qt_political = "review"; the tags given -> 1; every indicator NOT tagged ->
#                    "review" (TP's doubt covers what she did not tag, so no 0 is asserted); qt_review = 1.
#   * REVIEW alone:  qt_political = "review", qt_review = 1, every other column empty (nothing coded).
#   * QUALTAG blank: every qt_ column empty; listed for TP.
# Tags are read without regard to case or surrounding spaces; "humor" is read as "humour".  A tag outside the
# scheme stops the script (nothing is dropped silently).
#
# DESIGN PRINCIPLES
# -----------------
# 1. TP's files stay hers: only the qt_ columns are added.  Before anything is written, the file must survive a
#    read-write round trip byte for byte (same quoting, line endings, final-newline state); the original columns
#    are fingerprinted before and after, and the written file is read back and compared.
# 2. Re-runnable: qt_ columns already in a file are dropped and recomputed from QUALTAG, so the script is simply
#    re-run after TP edits a file or uploads a new bin.
# 3. One file per bin: the script finds BIN<N>_topics_labelled_*.csv and stops if a bin has two (a superseded
#    version is removed with git rm, as BIN1_..._TP:BA.csv and BIN3_..._TP.csv were on 2026-10-28).
#
# HOW TO RUN (repository root)
# ----------------------------
#   Rscript code/13-recode-qualtag.R          # seconds
#   QT_REVIEW_DIR=<folder> for a test on copies of the files.
#
# OUTPUTS  (results/analysis/manual review/)
# -------
#   BIN<N>_topics_labelled_*.csv   the same files with 11 columns after QUALTAG: qt_wc, qt_political,
#                                  qt_host_country_domestic, qt_national_identity, qt_fifa_body, qt_ir, qt_club,
#                                  qt_intl_non_wc, qt_player, qt_humour, qt_review
#   QUALTAG_to_settle.csv          the topics the rules leave without qt_political, for TP:
#                                  bin, topic_id, name, QUALTAG, issue
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                     # shared helpers (code/R/pipeline-utils.R)
ROOT <- find_root()
setwd(ROOT)

REVIEW_DIR <- Sys.getenv("QT_REVIEW_DIR", "results/analysis/manual review")
FILE_RE    <- "^BIN(\\d)_topics_labelled_.*\\.csv$"
SETTLE     <- file.path(REVIEW_DIR, "QUALTAG_to_settle.csv")

# The scheme: tag as read (lower case, single spaces) -> its indicator column
POLITICAL_TAGS <- c("host country domestic" = "qt_host_country_domestic", "national identity" = "qt_national_identity",
                    "fifa body" = "qt_fifa_body", "ir" = "qt_ir")
QUALIFIER_TAGS <- c("club" = "qt_club", "intl non-wc" = "qt_intl_non_wc", "player" = "qt_player", "humour" = "qt_humour")
TAG_COL        <- c("wc" = "qt_wc", POLITICAL_TAGS, QUALIFIER_TAGS, "review" = "qt_review")
NONPOL         <- "non-political"                  # no column of its own: it decides qt_political
SPELLINGS      <- c("humor" = "humour")            # variants found in the files, read as the scheme's spelling
QT_COLS        <- c("qt_wc", "qt_political", unname(POLITICAL_TAGS), unname(QUALIFIER_TAGS), "qt_review")
INDICATOR_COLS <- setdiff(QT_COLS, "qt_political")

# ---------------------------------------------------------------------------
# 1. Helpers: parse a QUALTAG cell, recode it, write a file in its source format
# ---------------------------------------------------------------------------
parse_tags <- function(x) {
  p <- tolower(gsub("\\s+", " ", trimws(strsplit(x, ";", fixed = TRUE)[[1]])))
  p <- p[p != ""]                                  # a trailing ';' leaves an empty part
  unique(unname(ifelse(p %in% names(SPELLINGS), SPELLINGS[p], p)))
}

# One topic's tags -> the 11 qt_ values and the reason it goes on TP's list ("" = settled)
recode_tags <- function(tags) {
  out <- setNames(rep("", length(QT_COLS)), QT_COLS)
  if (length(tags) == 0) return(list(values = out, issue = "blank QUALTAG"))
  review <- "review" %in% tags
  if (review && length(tags) == 1) {               # REVIEW alone: flagged, nothing else coded
    out[c("qt_political", "qt_review")] <- c("review", "1")
    return(list(values = out, issue = ""))
  }
  out[INDICATOR_COLS] <- if (review) "review" else "0"   # untagged: "review" under TP's doubt, else 0
  out[TAG_COL[intersect(tags, names(TAG_COL))]] <- "1"
  if (review) { out["qt_political"] <- "review"; return(list(values = out, issue = "")) }
  pol <- any(tags %in% names(POLITICAL_TAGS)); np <- NONPOL %in% tags
  if (pol && !np) out["qt_political"] <- "yes"
  if (np && !pol) out["qt_political"] <- "no"
  issue <- if (pol && np) "conflict: a political tag and non-political" else
           if (!pol && !np) "untagged: no political tag, no non-political, no REVIEW" else ""
  list(values = out, issue = issue)
}

# The file's own format, so that the rewrite changes nothing but the added columns
source_format <- function(raw) {
  n_crlf <- length(grepRaw("\r\n", raw, fixed = TRUE, all = TRUE))
  n_lf   <- length(grepRaw("\n", raw, fixed = TRUE, all = TRUE))
  if (n_crlf > 0 && n_crlf != n_lf) stop("mixed line endings")
  list(bom = length(raw) >= 3 && identical(raw[1:3], as.raw(c(0xef, 0xbb, 0xbf))),
       eol = if (n_crlf > 0) "\r\n" else "\n",
       final_eol = length(raw) > 0 && raw[length(raw)] == as.raw(0x0a))
}
serialise <- function(d, fmt) {
  head <- sub("\n$", "", format_csv(d[0, ], na = ""))       # readr ends the header with "\n" whatever `eol` says
  s <- paste0(head, fmt$eol, format_csv(d, na = "", eol = fmt$eol, col_names = FALSE))
  if (!fmt$final_eol) s <- sub(paste0(fmt$eol, "$"), "", s)
  out <- charToRaw(enc2utf8(s))
  if (fmt$bom) c(as.raw(c(0xef, 0xbb, 0xbf)), out) else out
}

# ---------------------------------------------------------------------------
# 2. The files: one per bin
# ---------------------------------------------------------------------------
files <- list.files(REVIEW_DIR, pattern = FILE_RE)
bins  <- as.integer(sub(FILE_RE, "\\1", files))
if (anyDuplicated(bins)) stop("more than one file for bin(s) ", paste(unique(bins[duplicated(bins)]), collapse = ", "),
                              ": remove the superseded version first")
files <- files[order(bins)]; bins <- sort(bins)
log_msg("%d review files: %s", length(files), paste(files, collapse = ", "))

src <- lapply(files, function(f) {
  p <- file.path(REVIEW_DIR, f); raw <- readBin(p, "raw", file.size(p))
  d <- read_chr(p, trim_ws = FALSE)                # keep TP's cells exactly (readr trims by default)
  list(path = p, raw = raw, fmt = source_format(raw), d = d)
})

# Every tag must belong to the scheme before anything is written
for (i in seq_along(src)) {
  d <- src[[i]]$d
  if (!"QUALTAG" %in% names(d)) stop(files[i], ": no QUALTAG column")
  if (anyDuplicated(d$topic_id)) stop(files[i], ": duplicated topic_id")
  unknown <- setdiff(unlist(lapply(d$QUALTAG, parse_tags)), c(names(TAG_COL), NONPOL))
  if (length(unknown)) stop(files[i], ": tags outside the scheme: ", paste(unknown, collapse = ", "))
}

# ---------------------------------------------------------------------------
# 3. Recode, insert after QUALTAG, write, read back
# ---------------------------------------------------------------------------
settle <- list(); summary_rows <- list()
for (i in seq_along(src)) {
  s <- src[[i]]; f <- files[i]
  ok <- identical(serialise(s$d, s$fmt), s$raw)
  check(sprintf("bin %d: the file survives a round trip unchanged", bins[i]), ok)
  if (!ok) stop(f, ": its format cannot be reproduced byte for byte; nothing written")

  d0   <- s$d[, setdiff(names(s$d), QT_COLS)]      # a re-run recomputes the qt_ columns
  orig <- vapply(d0, col_hash, "")
  rec  <- lapply(d0$QUALTAG, function(x) recode_tags(parse_tags(x)))
  qt   <- as_tibble(do.call(rbind, lapply(rec, `[[`, "values")))
  k    <- match("QUALTAG", names(d0))
  d1   <- bind_cols(d0[, seq_len(k)], qt, d0[, setdiff(seq_along(d0), seq_len(k)), drop = FALSE])
  check(sprintf("bin %d: original columns unchanged, qt_ columns after QUALTAG", bins[i]),
        identical(vapply(d1[, names(d0)], col_hash, ""), orig) &&
        identical(names(d1), append(names(d0), QT_COLS, after = k)))

  writeBin(serialise(d1, s$fmt), s$path)
  back <- read_chr(s$path, trim_ws = FALSE)
  check(sprintf("bin %d: written file reads back identical, same format", bins[i]),
        identical(names(back), names(d1)) && all(mapply(identical, back, d1)) &&
        identical(source_format(readBin(s$path, "raw", file.size(s$path))), s$fmt))

  issue <- vapply(rec, `[[`, "", "issue")
  if (any(issue != "")) settle[[f]] <- tibble(bin = bins[i], topic_id = d0$topic_id, name = d0$name,
                                              QUALTAG = d0$QUALTAG, issue = issue) |> filter(issue != "")
  pv <- qt$qt_political
  summary_rows[[f]] <- tibble(bin = bins[i], topics = nrow(qt), yes = sum(pv == "yes"), no = sum(pv == "no"),
                              review = sum(pv == "review"), empty = sum(pv == ""), to_settle = sum(issue != ""))
}

# ---------------------------------------------------------------------------
# 4. The list for TP and the summary
# ---------------------------------------------------------------------------
settle <- bind_rows(settle)
if (nrow(settle) == 0) settle <- tibble(bin = integer(), topic_id = character(), name = character(),
                                        QUALTAG = character(), issue = character())
write_csv(settle, SETTLE, na = "")
log_msg("%s: %d topics for TP to settle", SETTLE, nrow(settle))

summ <- bind_rows(summary_rows)
print(as.data.frame(bind_rows(summ, summarise(summ, across(-bin, sum)))), row.names = FALSE)   # last row: all bins
check("every topic left without qt_political is on TP's list", sum(summ$empty) == nrow(settle))

if (!all(unlist(results))) stop("a check failed: ", paste(names(results)[!unlist(results)], collapse = ", "))
cat("ALL CHECKS PASSED\n")
