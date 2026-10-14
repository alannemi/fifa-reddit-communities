##############################################################################
# FIFA COMMUNITIES CHAPTER
## 3. DATA VALIDATION  (run after 02-data-cleaning.R)
##############################################################################
#
# PURPOSE
# -------
# Independent checks that the rebuilt files are complete, consistent and
# faithful to the raw export.  Every check prints PASS/FAIL; the script stops
# with an error at the end if any check failed, so it can be used in a
# re-run workflow.  Nothing is modified.
#
# CHECKS
# ------
#  C1  row accounting: raw == clean + deleted + removed + bot + promotional
#                              + encoding + non_english (S1 rows, if any, are
#                      dropped and reported by 02-data-cleaning.R)
#  C2  ids are unique within and across all output files
#  C3  yearly files sum to the master and contain only their own year
#  C4  identical column specification in master, yearly and side files
#  C5  date range and continuity (every month 2010-01 .. 2026-07 present)
#  C6  fidelity: a random sample of ids is compared with the raw file
#      byte-for-byte on title and selftext (catches the fread quote bug)
#  C6d/e  every S6 encoding repair reverses to the raw bytes and is present in the clean file
#  C7  JSON columns parse (sample of non-empty flair/media/preview values)
#  C8  clean master contains no removal markers, bot authors, spam accounts,
#      placeholder titles or replacement characters
#
# HOW TO RUN:  Rscript code/03-data-validation.R   (from the repository root)
##############################################################################

suppressPackageStartupMessages({ library(readr); library(dplyr); library(stringr); library(jsonlite) })
set.seed(20261011)

find_root <- function() {
  if (file.exists("data/soccer_posts.csv")) return(normalizePath("."))
  cand <- path.expand("~/OneDrive - McGill University/SHARED FOLDER - FIFA Chapter/fifa-reddit-communities")
  if (file.exists(file.path(cand, "data/soccer_posts.csv"))) return(cand)
  stop("repository root not found")
}
setwd(find_root())
OUT_ROOT <- Sys.getenv("CLEAN_OUT_ROOT", ".")   # same override as 02-data-cleaning.R (dry runs)

results <- list()
check <- function(name, ok, detail = "") {
  results[[name]] <<- ok
  cat(sprintf("[%s] %-60s %s\n", if (isTRUE(ok)) "PASS" else "FAIL", name, detail))
}
read_chr <- function(p, ...) read_csv(p, col_types = cols(.default = col_character()),
                                      na = character(), progress = FALSE, lazy = FALSE, ...)

# ---------------------------------------------------------------------------
# Load outputs (character, verbatim)
# ---------------------------------------------------------------------------
files <- c(clean       = file.path(OUT_ROOT, "data/soccer_data.clean.csv"),
           deleted     = file.path(OUT_ROOT, "data/deleted_posts.csv"),
           removed     = file.path(OUT_ROOT, "data/removed_posts.csv"),
           bot         = file.path(OUT_ROOT, "data/bot_threads.csv"),
           promotional = file.path(OUT_ROOT, "data/promotional_posts.csv"),
           encoding    = file.path(OUT_ROOT, "data/encoding_issue_posts.csv"),
           non_english = file.path(OUT_ROOT, "data/non_english_posts.csv"))
cat("Loading output files ...\n")
d <- lapply(files, read_chr)
n <- sapply(d, nrow)
print(n)

# Raw row count without loading the whole raw file: count ids only.
raw_ids <- read_csv("data/soccer_posts.csv", col_select = c("id"),
                    col_types = cols(.default = col_character()), progress = FALSE)$id
n_raw <- length(raw_ids)
n_s1  <- 0L   # S1 (empty content) rows are dropped; 02-data-cleaning.R reports them (expected 0)

# C1 row accounting ---------------------------------------------------------
check("C1 raw == sum of outputs", sum(n) + n_s1 == n_raw,
      sprintf("raw=%s outputs=%s", format(n_raw, big.mark = ","), format(sum(n), big.mark = ",")))

# C2 unique ids ---------------------------------------------------------------
all_ids <- unlist(lapply(d, `[[`, "id"), use.names = FALSE)
check("C2 ids unique across all outputs", !any(duplicated(all_ids)) && setequal(all_ids, raw_ids))

# C3 yearly files -------------------------------------------------------------
yearly_dir <- file.path(OUT_ROOT, "data/yearly datasets")
yfiles <- list.files(yearly_dir, pattern = "^soccer\\.data\\.cleaned-[0-9]{4}\\.csv$", full.names = TRUE)
ysum <- 0L; yok <- TRUE; ycols_ok <- TRUE
for (f in yfiles) {
  y <- str_extract(basename(f), "[0-9]{4}")
  yd <- read_chr(f)
  ysum <- ysum + nrow(yd)
  if (!all(yd$year == y) || !all(substr(yd$created_datetime, 1, 4) == y)) yok <- FALSE
  if (!identical(names(yd), names(d$clean))) ycols_ok <- FALSE
}
check("C3a yearly rows sum to master", ysum == n[["clean"]], sprintf("%d files, %s rows", length(yfiles), format(ysum, big.mark = ",")))
check("C3b each yearly file holds only its year", yok)
check("C3c no redundant master copy in yearly folder", !file.exists(file.path(yearly_dir, "soccer_data.clean.csv")))

# C4 column spec --------------------------------------------------------------
expected_cols <- c("id","name","title","selftext","author","author_flair_text","author_flair_css_class",
  "author_flair_background_color","author_flair_text_color","author_flair_type","author_flair_richtext",
  "author_flair_template_id","link_flair_text","link_flair_css_class","link_flair_background_color",
  "link_flair_text_color","link_flair_type","link_flair_richtext","link_flair_template_id","score","ups",
  "downs","num_comments","num_crossposts","total_awards_received","gilded","created_datetime","edited",
  "domain","url","permalink","is_self","is_video","is_reddit_media_domain","post_hint","removed_by_category",
  "subreddit_subscribers","media","preview","year","post_type","lang")
check("C4a master has the 42 expected columns", identical(names(d$clean), expected_cols))
check("C4b yearly files share the master spec", ycols_ok)
check("C4c side files = master spec + filter_reason",
      all(sapply(d[-1], function(x) identical(names(x), c(expected_cols, "filter_reason")))))

# C5 dates --------------------------------------------------------------------
dt <- d$clean$created_datetime
check("C5a datetime format YYYY-MM-DD HH:MM:SS", all(str_detect(dt, "^[0-9]{4}-[0-9]{2}-[0-9]{2} [0-9]{2}:[0-9]{2}:[0-9]{2}$")))
check("C5b range within 2010-01-01 .. 2026-07-31", min(dt) >= "2010-01-01" && max(dt) < "2026-08-01", sprintf("%s .. %s", min(dt), max(dt)))
months_present <- unique(substr(dt, 1, 7))
months_expected <- format(seq(as.Date("2010-01-01"), as.Date("2026-07-01"), by = "month"), "%Y-%m")
check("C5c every month 2010-01..2026-07 present", all(months_expected %in% months_present))

# C6 fidelity vs raw ----------------------------------------------------------
# Read the raw rows for a random sample of 5,000 clean ids and compare title /
# selftext byte-for-byte.  Reading the raw file once more (character) is the
# simplest faithful comparison; ~2-3 min.
# Rows whose title/selftext were repaired in stage S6 legitimately differ from
# the raw file; they are excluded from the byte comparison and checked
# separately (C6d) by reversing the recorded transformation.
repairs <- read_csv(file.path(OUT_ROOT, "data/data artefacts/encoding_repairs.csv"),
                    col_types = cols(.default = col_character()), na = character())
sample_ids <- sample(setdiff(d$clean$id, repairs$id), 5000)
raw_sub <- read_csv("data/soccer_posts.csv", col_select = c("id", "title", "selftext", "author_flair_richtext"),
                    col_types = cols(.default = col_character()), na = character(), progress = FALSE) |>
  filter(id %in% sample_ids)
cmp <- d$clean |> filter(id %in% sample_ids) |> select(id, title, selftext) |> inner_join(raw_sub, by = "id", suffix = c(".clean", ".raw"))
check("C6a title identical to raw (5,000-id sample)", nrow(cmp) == 5000 && all(cmp$title.clean == cmp$title.raw))
check("C6b selftext identical to raw (5,000-id sample)", all(cmp$selftext.clean == cmp$selftext.raw))
# A doubled quote ("") may legitimately occur in a raw title; the artefact we
# guard against is a *systematic* doubling, so the sample counts must match.
check("C6c doubled-quote count equals raw (sample)",
      sum(str_detect(cmp$title.clean, '""')) == sum(str_detect(cmp$title.raw, '""')),
      sprintf("%d sampled titles contain a doubled quote in both", sum(str_detect(cmp$title.raw, '""'))))

# C6d: every logged repair must reverse exactly — encoding the repaired text
# as UTF-8 bytes and decoding those bytes with the recorded source encoding
# must give back the original (damaged) text.  This proves no content was
# invented: the repair is a pure re-interpretation of the same bytes.
rt <- vapply(seq_len(nrow(repairs)), function(i) {
  b <- iconv(repairs$repaired[i], from = "UTF-8", to = "UTF-8", toRaw = TRUE)[[1]]
  back <- iconv(list(b), from = repairs$source_encoding[i], to = "UTF-8")
  identical(back, repairs$original[i])
}, logical(1))
check("C6d all S6 repairs reverse to the original bytes", all(rt), sprintf("%d repaired fields", nrow(repairs)))
# C6e: repaired values are what the clean file now contains
rep_clean <- d$clean |> filter(id %in% repairs$id) |> select(id, title, selftext)
rep_ok <- all(vapply(seq_len(nrow(repairs)), function(i) {
  row <- rep_clean[rep_clean$id == repairs$id[i], ]
  nrow(row) == 0 || identical(row[[repairs$field[i]]], repairs$repaired[i])   # nrow 0 = row filtered later (e.g. S7)
}, logical(1)))
check("C6e repaired values present in clean file", rep_ok)

# C7 JSON columns parse ---------------------------------------------------------
json_ok <- function(v) { v <- v[v != "" & v != "[]"]; v <- sample(v, min(1000, length(v)))
  all(vapply(v, function(s) !inherits(try(fromJSON(s, simplifyVector = FALSE), silent = TRUE), "try-error"), logical(1))) }
check("C7a author_flair_richtext parses (sample)", json_ok(d$clean$author_flair_richtext))
check("C7b link_flair_richtext parses (sample)",   json_ok(d$clean$link_flair_richtext))
check("C7c media parses (sample)",                 json_ok(d$clean$media))
check("C7d preview parses (sample)",               json_ok(d$clean$preview))

# C8 clean master is free of filtered material --------------------------------
cl <- d$clean
check("C8a no removed_by_category values", all(cl$removed_by_category == ""))
check("C8b no [removed]/[deleted] selftext", !any(str_trim(cl$selftext) %in% c("[removed]", "[deleted]")))
check("C8c no placeholder titles", !any(str_trim(cl$title) %in% c("[ Removed by moderator ]", "[ Removed by Reddit ]", "[deleted by user]", "[deleted]", "[image processing failed]")))
check("C8d no bot authors", !any(cl$author %in% c("MatchThreadder", "AutoModerator", "2soccer2bot", "soccerbot")))
spam <- read_csv(file.path(OUT_ROOT, "data/data artefacts/spam_accounts.csv"), show_col_types = FALSE)
check("C8e no spam-account authors", !any(cl$author %in% spam$author))
check("C8f no U+FFFD replacement characters", !any(str_detect(paste(cl$title, cl$selftext), "�")))
check("C8g no empty title+selftext", !any(str_trim(cl$title) == "" & str_trim(cl$selftext) == ""))
check("C8h post_type and lang populated", all(cl$post_type %in% c("regular", "match_thread", "daily_thread")) && all(cl$lang != ""))

# Summary ---------------------------------------------------------------------
cat("\nSummary: ", sum(unlist(results)), "passed /", length(results), "checks\n")
if (!all(unlist(results))) stop("VALIDATION FAILED: ", paste(names(results)[!unlist(results)], collapse = ", "))
cat("ALL CHECKS PASSED\n")
