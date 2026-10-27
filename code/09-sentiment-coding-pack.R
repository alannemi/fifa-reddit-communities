##############################################################################
# FIFA COMMUNITIES CHAPTER
## 9. SENTIMENT CODING PACK  (human validation of VADER — the coding sheet)
##############################################################################
#
# PURPOSE
# -------
# Turn the stratified validation sample written by stage S4
# (`data/analysis/sentiment_validation_sample.csv`, 337 posts, 54 strata of
# bin x VADER label x is_self) into a blinded coding sheet.  The human labels
# are compared with VADER by 10-sentiment-validation.R.  Since the author's
# revision of 2026-10-27 one coder (Alan Nemirovski) codes the sample; the
# script can still write a second sheet (CODERS) if a second coder is added.
# Decision E1 (Alan Nemirovski, 2026-10-27): the sentiment calibration is the
# human validation of design §8 / codebook §8.2.
#
# DESIGN PRINCIPLES
# -----------------
# 1. Blind: the sheets carry no VADER score or label and no period bin, and the
#    rows are in a seeded random order (not sorted by stratum), so a coder
#    cannot infer the measure being validated.
# 2. One file per coder (now one: coder_A.csv); with a second coder the files
#    hold the same rows in the same order and are coded independently.
# 3. Same unit as the measure: the code is the sentiment of the TITLE
#    (`post_sentiment` is the compound score of the title, F18); the body of a
#    self post is shown, cut at 1,000 characters like `post_sentiment_full`,
#    as context only.  Guide: docs/session-09_2026-10-27/01_sentiment-coding-guide.md.
# 4. Never overwrites a sheet that has codes in it (a re-run after coding has
#    started stops instead).
#
# HOW TO RUN (repository root)
# ----------------------------
#   Rscript code/09-sentiment-coding-pack.R
#
# OUTPUTS  (results/analysis/manual review/sentiment_coding/ — where the co-authors upload)
# -------
#   coder_A.csv   id, title, body_excerpt, is_self, code (blank), notes (blank)
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr) })

.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                     # shared helpers (code/R/pipeline-utils.R)
ROOT <- find_root()
setwd(ROOT)

SAMPLE   <- "data/analysis/sentiment_validation_sample.csv"
YEARLY   <- "data/yearly datasets"
OUT_DIR  <- "results/analysis/manual review/sentiment_coding"
BODY_MAX <- 1000L                  # = config.BODY_MAX_CHARS, the body cut of post_sentiment_full
SEED     <- 42L                    # = config.SEED
CODERS   <- "A"                    # one coder (author, 2026-10-27); c("A", "B") for two

# ---------------------------------------------------------------------------
# 1. The sample, with the body of each post from the yearly files
# ---------------------------------------------------------------------------
smp <- read_chr(SAMPLE)
stopifnot(!anyDuplicated(smp$id))
log_msg("sample: %d posts", nrow(smp))

body <- bind_rows(lapply(list.files(YEARLY, pattern = "^soccer\\.data\\.cleaned-\\d{4}\\.csv$", full.names = TRUE),
                         function(p) read_chr(p, col_select = c("id", "selftext")) |> filter(id %in% smp$id)))
stopifnot(all(smp$id %in% body$id))

sheet <- smp |>
  select(id, title, is_self) |>
  left_join(body, by = "id") |>
  mutate(body_excerpt = ifelse(is_self == "True", substr(selftext, 1, BODY_MAX), ""),
         code = "", notes = "") |>
  select(id, title, body_excerpt, is_self, code, notes)

set.seed(SEED)
sheet <- sheet[sample(nrow(sheet)), ]

# ---------------------------------------------------------------------------
# 2. The blank sheet(s) (never over a sheet that already has codes)
# ---------------------------------------------------------------------------
dir.create(OUT_DIR, recursive = TRUE, showWarnings = FALSE)
for (coder in CODERS) {
  p <- file.path(OUT_DIR, sprintf("coder_%s.csv", coder))
  if (file.exists(p) && any(read_chr(p)$code != "")) stop(p, " already holds codes; not overwritten")
  write_csv(sheet, p, na = "")
  log_msg("wrote %s (%d rows)", p, nrow(sheet))
}
