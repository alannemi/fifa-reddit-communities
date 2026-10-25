##############################################################################
# FIFA COMMUNITIES CHAPTER
## 2. DATA CLEANING  (v2 — full rebuild from the raw export)
##############################################################################
#
# PURPOSE
# -------
# Rebuild every cleaned data file of the r/soccer post corpus directly from
# the raw export `data/soccer_posts.csv` (2,047,322 posts, 1 Jan 2010 -
# 31 Jul 2026, Arctic Shift archive), applying a documented, ordered set of
# filters so the retained corpus contains only posts that can contribute to
# an English-language discourse / topic-modelling / sentiment analysis.
#
# WHY A FULL REBUILD (rather than cleaning the earlier cleaned file)
# -----------------------------------------------------------------
# The first version of this script (Aug 2026) loaded the raw file with
# data.table::fread().  On this particular file fread does NOT un-escape
# RFC-4180 doubled quotes, so every literal quotation mark in every text
# field came through as two quotation marks:  "Old carcass"  ->  ""Old carcass""
# and the JSON columns became unparseable ([{""e"": ""text""}]).  The raw file
# itself is correct (it was written by Python's csv module) and readr::read_csv()
# parses it correctly (0 parsing problems).  Session-01 audit (2026-10-11):
# see docs/session-01_2026-10-11/01_data-audit-findings.md.
#
# Rebuilding from raw also lets us fix two other gaps of v1:
#   * removed/deleted posts were detected only through `selftext`, which is
#     empty for link posts (88% of the corpus), so ~203k posts removed by
#     moderators / Reddit's spam filter / AutoModerator survived;
#   * the master file and the yearly files were written by two different
#     writers (readr vs base R) and therefore had different date formats,
#     boolean spellings and missing-value conventions.
#
# DESIGN PRINCIPLES
# -----------------
# 1. Fidelity: every column is read as *character* and written back verbatim.
#    No type coercion, no re-formatting of dates, booleans or JSON.  Numeric
#    tests (e.g. num_comments == 0) are done on temporary integer copies.
# 2. Nothing is destroyed: every filtered row is written to exactly one side
#    file with a `filter_reason` column.  raw == clean + sum(side files).
# 3. One writer for every output (readr::write_csv, UTF-8, RFC-4180 quoting,
#    empty cell = missing), so all files share one format.
# 4. Every rule is explicit, configurable (CONFIG section) and documented
#    inline and in docs/session-01_2026-10-11/02_cleaning-rules-and-rationale.md.
#
# FILTER ORDER (a row is assigned to the FIRST stage that matches)
# ---------------------------------------------------------------
#   S1  empty content          -> dropped (none expected; Reddit requires a title)
#   S2  deleted by the user    -> data/deleted_posts.csv
#   S3  removed by mods/Reddit -> data/removed_posts.csv
#   S4  bot-authored threads   -> data/bot_threads.csv
#   S5  promotional / spam     -> data/promotional_posts.csv
#   S6  encoding-damaged text  -> repaired in place when a strict byte-level
#                                 re-decoding succeeds, else data/encoding_issue_posts.csv
#   S7  non-English text       -> data/non_english_posts.csv
#   --  everything else        -> data/soccer_data.clean.csv (+ yearly splits)
#
# LITERATURE ANCHORS (real works; thresholds are ours, justified empirically)
# -------------------------------------------------------------------------
# * Treating "[deleted]"/"[removed]" markers and removal metadata as signals
#   that content is no longer part of the visible discourse, and excluding
#   bot accounts, follows common practice in Reddit research; see
#   Baumgartner, Zannettou, Keegan, Squire & Blackburn (2020) "The Pushshift
#   Reddit Dataset", ICWSM; and Proferes, Jones, Gilbert, Fiesler & Zimmer
#   (2021) "Studying Reddit: A Systematic Overview of Disciplines, Approaches,
#   Methods, and Ethics", Social Media + Society.
# * Account-level spam features — a high share of posts carrying links,
#   concentration of those links on one domain, and near-total absence of
#   interaction — follow the feature families used by Benevenuto, Magno,
#   Rodrigues & Almeida (2010) "Detecting Spammers on Twitter" (CEAS);
#   Stringhini, Kruegel & Vigna (2010) "Detecting Spammers on Social
#   Networks" (ACSAC); and Lee, Eoff & Caverlee (2011) "Seven Months with
#   the Devils: A Long-Term Study of Content Polluters on Twitter" (ICWSM).
#   Those papers train classifiers; we use the same behavioural signals as
#   transparent rules.  The cut-offs (>=50 posts, >=80% one domain, >=90%
#   zero-comment posts) were chosen from the empirical gap observed in this
#   corpus: legitimate single-outlet accounts (e.g. newspaper accounts) have
#   0-15% zero-comment posts, the content-farm accounts >=95%.
# * Language identification uses Google's Compact Language Detector 2 via
#   the R package `cld2` (J. Ooms), with `cld3` (neural CLD3) as a second
#   opinion when available; disagreement or an "unreliable" verdict keeps
#   the post (conservative).
#
# HOW TO RUN
# ----------
#   From the repository root:   Rscript code/02-data-cleaning.R
#   Needs ~6-8 GB of free RAM and ~15 minutes.  Outputs overwrite in place.
#   Requires: readr, dplyr, stringr, purrr, cld2 (optional: cld3).
#
# OUTPUTS
# -------
#   data/soccer_data.clean.csv                       retained corpus (42 cols)
#   data/yearly datasets/soccer.data.cleaned-YYYY.csv  same, split by year
#   data/deleted_posts.csv, removed_posts.csv, bot_threads.csv,
#   promotional_posts.csv, encoding_issue_posts.csv, non_english_posts.csv
#                                                    (42 cols + filter_reason)
#   data/data artefacts/filter_summary.csv           counts per stage/reason
#   data/data artefacts/spam_accounts.csv            accounts caught by S5a
#   data/data artefacts/piracy_accounts.csv          accounts caught by S5e
#   data/data artefacts/encoding_repairs.csv         every text field repaired in S6 (before/after)
##############################################################################

suppressPackageStartupMessages({
  library(readr)
  library(dplyr)
  library(stringr)
  library(purrr)
})

t_start <- Sys.time()

# ---------------------------------------------------------------------------
# 0. SHARED HELPERS AND THE REPOSITORY ROOT
#    code/R/pipeline-utils.R provides log_msg(), read_chr(), find_root() and the column specifications.
#    Works when run with `Rscript code/02-data-cleaning.R` from the repository root or sourced in RStudio
#    with the .Rproj open; there is no machine-specific fallback (session 07).
# ---------------------------------------------------------------------------
.utils <- local({ a <- commandArgs(trailingOnly = FALSE); f <- sub("^--file=", "", a[grepl("^--file=", a)])
                  if (length(f) == 1) file.path(dirname(normalizePath(f)), "R", "pipeline-utils.R") else "code/R/pipeline-utils.R" })
source(.utils)                     # shared helpers and column specifications (code/R/pipeline-utils.R)
ROOT <- find_root()
setwd(ROOT)
if (!file.exists("data/soccer_posts.csv")) stop("raw export data/soccer_posts.csv not found under ", ROOT, " (copy data/ from the shared folder)")
log_msg("Repository root: %s", ROOT)

# Test overrides (used only for dry runs; leave unset for the real rebuild):
#   CLEAN_NMAX     = read only the first N raw rows
#   CLEAN_OUT_ROOT = write all outputs under this directory instead of ROOT
N_MAX    <- suppressWarnings(as.integer(Sys.getenv("CLEAN_NMAX", NA)))
OUT_ROOT <- Sys.getenv("CLEAN_OUT_ROOT", ROOT)
if (OUT_ROOT != ROOT) log_msg("TEST MODE: outputs redirected to %s", OUT_ROOT)
for (d in c("data", "data/yearly datasets", "data/data artefacts"))
  dir.create(file.path(OUT_ROOT, d), showWarnings = FALSE, recursive = TRUE)

RAW_FILE    <- "data/soccer_posts.csv"                       # input (never modified)
CLEAN_FILE  <- file.path(OUT_ROOT, "data/soccer_data.clean.csv")
YEARLY_DIR  <- file.path(OUT_ROOT, "data/yearly datasets")
ARTEFACTS   <- file.path(OUT_ROOT, "data/data artefacts")
SIDE_FILES  <- c(
  deleted     = file.path(OUT_ROOT, "data/deleted_posts.csv"),
  removed     = file.path(OUT_ROOT, "data/removed_posts.csv"),
  bot         = file.path(OUT_ROOT, "data/bot_threads.csv"),
  promotional = file.path(OUT_ROOT, "data/promotional_posts.csv"),
  encoding    = file.path(OUT_ROOT, "data/encoding_issue_posts.csv"),
  non_english = file.path(OUT_ROOT, "data/non_english_posts.csv")
)

# ---------------------------------------------------------------------------
# 1. CONFIGURATION — every list and threshold used by the filters.
#    Edit here (not in the filter code) if a rule needs to change; the
#    rationale for each entry is given in the comments.
# ---------------------------------------------------------------------------

## 1.1 Columns ---------------------------------------------------------------
# The 39 columns retained in v1 (same list, same order), i.e. the raw 64
# columns minus the 25 dropped below.  Dropped columns are either constant
# for a single-subreddit corpus (subreddit, subreddit_id, subreddit_type,
# quarantine), redundant (created_utc duplicates created_datetime; thumbnail*),
# retrieval-time artefacts (retrieved_on, author_is_blocked, author_premium,
# archived, hide_score, no_follow) or very sparsely populated UI flags.
KEEP_COLS <- c(
  "id", "name", "title", "selftext", "author",
  "author_flair_text", "author_flair_css_class", "author_flair_background_color",
  "author_flair_text_color", "author_flair_type", "author_flair_richtext",
  "author_flair_template_id",
  "link_flair_text", "link_flair_css_class", "link_flair_background_color",
  "link_flair_text_color", "link_flair_type", "link_flair_richtext",
  "link_flair_template_id",
  "score", "ups", "downs", "num_comments", "num_crossposts",
  "total_awards_received", "gilded",
  "created_datetime", "edited",
  "domain", "url", "permalink",
  "is_self", "is_video", "is_reddit_media_domain", "post_hint",
  "removed_by_category", "subreddit_subscribers",
  "media", "preview"
)
DROPPED_COLS <- c(  # documented for the codebook; not read at all
  "author_fullname", "author_premium", "author_is_blocked", "upvote_ratio",
  "created_utc", "is_original_content", "thumbnail", "thumbnail_height",
  "thumbnail_width", "over_18", "spoiler", "stickied", "locked", "pinned",
  "distinguished", "contest_mode", "no_follow", "hide_score", "archived",
  "quarantine", "subreddit", "subreddit_id", "subreddit_type",
  "all_awardings", "retrieved_on"
)

## 1.2 Deleted / removed markers ---------------------------------------------
# Reddit (and the Arctic Shift snapshot) marks content in several ways:
#   selftext "[deleted]"              : author deleted a text post
#   selftext "[removed]"              : moderators / Reddit removed a text post
#   removed_by_category               : set for ANY post type (incl. link posts):
#       deleted, author               -> user-initiated deletion
#       moderator, reddit (spam filter), automod_filtered,
#       copyright_takedown, content_takedown -> platform/mod removal
#   placeholder titles such as "[ Removed by moderator ]" or "[deleted by user]"
#   selftext beginning "[ Removed by Reddit on account of violating ..."
DELETED_RBC  <- c("deleted", "author")
REMOVED_RBC  <- c("moderator", "reddit", "automod_filtered",
                  "copyright_takedown", "content_takedown")
TITLE_DELETED_PLACEHOLDERS <- c("[deleted by user]")
TITLE_REMOVED_PLACEHOLDERS <- c("[ Removed by moderator ]", "[ Removed by Reddit ]",
                                "[deleted]", "[Deleted]", "[image processing failed]")
SELFTEXT_REMOVED_PREFIX    <- "^\\s*\\[\\s*Removed by Reddit"

## 1.3 Bot authors -----------------------------------------------------------
# Accounts that post structural, template-generated threads (match threads,
# daily discussion, free-talk, merchandise threads).  Their *post text* is
# boilerplate; the discourse they host lives in comments, which this corpus
# does not contain (posts only).  Decision approved 2026-10-11.
BOT_AUTHORS <- c("MatchThreadder", "AutoModerator", "2soccer2bot", "soccerbot")

## 1.4 Community-hosting domains ---------------------------------------------
# Domains that r/soccer users routinely use to share clips, images and
# screenshots.  Posting 100% of one's links to these is normal community
# behaviour, so they are NOT counted as "own domain" in the spam-account rule
# and are exempt from the gambling / piracy domain rules.
COMMUNITY_DOMAINS <- c(
  "self.soccer", "reddit.com", "i.redd.it", "v.redd.it", "old.reddit.com",
  "twitter.com", "mobile.twitter.com", "x.com", "pbs.twimg.com", "t.co",
  "youtube.com", "youtu.be", "m.youtube.com",
  "imgur.com", "i.imgur.com", "m.imgur.com",
  "gfycat.com", "giphy.com", "dailymotion.com", "vine.co", "vimeo.com",
  "instagram.com", "facebook.com", "bsky.app", "threads.net", "tiktok.com",
  "en.wikipedia.org", "my.mixtape.moe", "imgtc.com", "i.minus.com",
  "clippituser.tv", "flixtc.com", "juststream.live", "streamgg.com"
)
# ... plus any clip-host domain starting with "stream"/"dubz" (streamable.com,
# streamja.com, streamin.one/.me/.link, streamff.*, streamain.com, streamwo.com,
# streamvi.com, streamion.*, streamsc.co, dubz.link/.co, ...).
COMMUNITY_DOMAIN_REGEX <- "^(stream[a-z]*|dubz)\\."
is_community_domain <- function(d) {
  d <- tolower(d)
  d %in% COMMUNITY_DOMAINS | str_detect(d, COMMUNITY_DOMAIN_REGEX)
}

## 1.5 Spam-account thresholds (ALL THREE must hold) -------------------------
SPAM_MIN_POSTS          <- 50    # enough posts for the shares to be meaningful
SPAM_DOMAIN_SHARE       <- 0.80  # >= 80% of posts link to ONE non-community domain
SPAM_ZERO_COMMENT_SHARE <- 0.90  # >= 90% of posts received zero comments

## 1.6 Gambling / betting-tip spam --------------------------------------------
# Domain tokens.  Most hits are Indonesian "prediksi"/"judi" (prediction /
# gambling) sites, Korean casino terms and bookmaker/affiliate sites.
# `bet` is matched as a token or in known compounds so that e.g. Real BETis
# (realbetisbalompie.es) or "alphabet" are not hit.
GAMBLING_DOMAIN_REGEX <- regex(paste0(
  "(prediksi|judi|taruhan|bandar|togel|casino|poker|sbobet|baccarat|sportsbook|",
  "bola[0-9]|303|slot|tangkas|qq\\.|pkv|",
  "(^|[^a-z])bet([^a-z]|$)|bet(ting|365|fair|host|king|top|win|online|corner|adda|tips|asia|way|vn)|",
  "(sbo|12|w88|aon|uni|sky|speed|war|city|asia|m|fb|188|a)bet|",
  "(bola|bet|judi|agen|toto|poker|casino|slot|sbo|win|king|citi|fb|w|mw)88|88(bet|bola|casino|win|sport|id|bong)|",
  "(^|[.-])agen|agen(bola|judi|sbobet|togel|casino|[0-9])|",
  "toto(strong|bet|[0-9])|",
  "바카라|카지노|먹튀|토토)"), ignore_case = TRUE)
# Title phrases (checked only for non-community domains).
GAMBLING_TITLE_REGEX <- regex(paste0(
  "(prediksi|judi|taruhan|bandar|togel|agen bola|situs judi|situs|",
  "betting tips|free (betting |football |soccer )?tips|sure ?win|",
  "\\bodds\\b.*\\btips\\b|\\btips\\b.*\\bodds\\b|pronostic|",
  "바카라|카지노|먹튀)"), ignore_case = TRUE)

## 1.7 Merchandise spam -------------------------------------------------------
MERCH_REGEX <- regex(paste0(
  "(cheap(est)?[\\w\\s,'-]{0,25}(jersey|jerseys|kit|kits|shirt|shirts|cleats|uniforms?)|",
  "(jersey|jerseys|kit|kits|shirt|shirts)[\\w\\s,'-]{0,25}(cheap|wholesale|for sale|discount|free (shipping|delivery))|",
  "wholesale[\\w\\s,'-]{0,25}(jersey|jerseys|kit|kits|shirt|shirts)|",
  "replica (jersey|jerseys|kit|kits|shirt|shirts)[\\w\\s,'-]{0,15}(sale|shop|cheap|buy|online)|",
  "soccer jerseys? (shop|store|sale|online)|",
  "try (us at )?(www\\.)?four4two|four4two\\.com|jerseysvip|jerseymate|",
  "custom (soccer|football) (uniforms?|jerseys?|kits?))"), ignore_case = TRUE)

## 1.8 URL-only titles --------------------------------------------------------
URL_ONLY_TITLE_REGEX <- regex("^\\s*https?://\\S+\\s*$", ignore_case = TRUE)

## 1.9 Stream-piracy posts (applied at ACCOUNT level: every post must match) --
PIRACY_TITLE_REGEX <- regex(paste0(
  "(live[ -]?stream(ing|s)?[^|]{0,60}\\b(free|online|hd|full hd)\\b|",
  "\\b(free|online|hd|full hd)\\b[^|]{0,60}live[ -]?stream(ing|s)?|",
  "watch[^|]{0,40}\\b(live|online|free)\\b[^|]{0,40}\\b(stream|streaming|hd)\\b|",
  "\\bstream(ing)?\\b[^|]{0,30}\\bfree\\b[^|]{0,30}\\bonline\\b)"), ignore_case = TRUE)

## 1.10 Encoding damage -------------------------------------------------------
# Detection: U+FFFD replacement characters; classic UTF-8-read-as-cp1252
# "mojibake" sequences (Ã + non-alphanumeric, â€, Â + non-alphanumeric);
# C0/C1 control characters other than TAB/LF/CR; titles without a single
# letter or digit.
# Repair (decision 2026-10-11, A. Nemirovski: "fix where possible, do not
# guess"): the only repair applied is a *byte-level re-interpretation* — the
# damaged text is converted back to the single-byte encoding it was wrongly
# decoded with (candidates tried in the order below) and the resulting bytes
# are decoded as UTF-8.  A repair is accepted only if (a) every character is
# representable in the candidate encoding, (b) the bytes form strictly valid
# UTF-8 and (c) no damage marker remains.  Nothing is inferred or replaced
# character-by-character; a row that fails all candidates is filtered out.
# The transformation is reversible and is logged per field in
# data/data artefacts/encoding_repairs.csv (original, repaired, encoding).

MOJIBAKE_REGEX <- regex("�|Ã[^A-Za-z0-9\\s]|â€|Â[^A-Za-z0-9\\s]")
CONTROL_REGEX  <- regex("[\\x{0000}-\\x{0008}\\x{000B}\\x{000C}\\x{000E}-\\x{001F}\\x{007F}-\\x{009F}]")
NO_ALNUM_REGEX <- regex("^[^\\p{L}\\p{N}]*$")
REPAIR_ENCODINGS <- c("WINDOWS-1252", "ISO-8859-1", "WINDOWS-1250")
is_damaged <- function(x) str_detect(x, MOJIBAKE_REGEX) | str_detect(x, CONTROL_REGEX)
# Try to repair a character vector; returns list(text = repaired-or-original,
# encoding = candidate that worked, or NA when no candidate was acceptable).
repair_text <- function(x) {
  out <- x; src <- rep(NA_character_, length(x))
  todo <- is_damaged(x)
  for (enc in REPAIR_ENCODINGS) {
    if (!any(todo)) break
    idx <- which(todo)
    bytes <- iconv(x[idx], from = "UTF-8", to = enc, toRaw = TRUE)   # NULL/NA when not representable
    cand <- vapply(bytes, function(b) {
      if (is.null(b) || length(b) == 0 || anyNA(b) || any(b == as.raw(0))) return(NA_character_)
      s <- rawToChar(b); Encoding(s) <- "UTF-8"
      if (validUTF8(s)) s else NA_character_
    }, character(1))
    ok <- !is.na(cand) & !is_damaged(cand) & cand != x[idx]
    out[idx[ok]] <- cand[ok]; src[idx[ok]] <- enc
    todo[idx[ok]] <- FALSE
  }
  list(text = out, encoding = src)
}

## 1.11 Language identification ----------------------------------------------
# Two evidence levels, because r/soccer titles are full of proper nouns
# (clubs, players, cities) that fool language detectors:
#   * non-Latin script (Cyrillic, Arabic, CJK, Thai, ...): >= 20 letters of
#     which >= 10 are non-Latin is enough;
#   * Latin-script languages (it, es, pt, de, fr, nl, id, ...): require
#     >= 40 letters, >= 6 word tokens, and fewer than half of the word tokens
#     capitalised (a title such as "Cagliari 1-[1] Lazio - Lucas Leiva" is
#     100% capitalised names and is kept), plus cld2/cld3 agreement.
# The decision is made on the TITLE (the text every reader sees and the text
# analysed for 88% of posts); a post whose title is English but whose body is
# a foreign-language quote/translation is kept.  A post is moved only when
# the title is confidently non-English AND the body is not confidently English.
# Review of a first full run (2026-10-11) showed ~20% of combined-text hits
# were such mixed posts (e.g. "[ESPN-BR] Very tense atmosphere ..." + Portuguese
# body), which motivated this title-based rule.
LANG_MIN_ALPHA_NONLATIN <- 15
LANG_MIN_NONLATIN_CHARS <- 8
LANG_MIN_ALPHA_LATIN    <- 30
LANG_MIN_TOKENS_LATIN   <- 5
LANG_MAX_CAP_SHARE      <- 0.50

## 1.12 post_type labels (derived column; informational, no filtering) -------
MATCH_THREAD_TITLE_REGEX <- regex("^\\s*\\[?\\s*(pre|post)?[- ]?match[- ]thread", ignore_case = TRUE)
MATCH_THREAD_FLAIR_REGEX <- regex("match[ _-]?thread|post[- ]?match|pre[- ]?match", ignore_case = TRUE)
DAILY_THREAD_TITLE_REGEX <- regex(paste0(
  "^\\s*(\\[?non-?pl\\]?\\s*)?(daily discussion|daily q&a|daily questions?|free talk friday|",
  "sunday support|monday moan|tactics thursday|trivia tuesday|motd thread|",
  "what to watch|wunderkind watch|change my view|merchandise discussion|",
  "weekly (discussion|questions?)|the \"?in case you missed it\"? thread)"), ignore_case = TRUE)
DAILY_THREAD_FLAIRS <- c("Daily Discussion", "Free Talk", "Monday Moan", "Sunday Support",
                         "Daily Questions Thread", "Today's Matches", "Merchandise",
                         "📺What to Watch", ":n_discussion: Daily Discussion",
                         ":n_freetalk: Free Talk")

# ---------------------------------------------------------------------------
# 2. LOAD THE RAW EXPORT
#    - only KEEP_COLS are read (col_select) to keep memory ~3-4 GB;
#    - every column as character; na = character() keeps empty strings as ""
#      (the raw file uses "" for missing; we preserve that convention).
# ---------------------------------------------------------------------------
log_msg("Reading %s (39 of 64 columns, all as character) ...", RAW_FILE)
raw <- read_csv(
  RAW_FILE,
  col_select = all_of(KEEP_COLS),
  col_types  = cols(.default = col_character()),
  na         = character(),
  locale     = locale(encoding = "UTF-8"),
  progress   = FALSE,
  lazy       = FALSE,
  n_max      = if (is.na(N_MAX)) Inf else N_MAX
)
stopifnot(identical(names(raw), KEEP_COLS))
prob <- problems(raw)
if (nrow(prob) > 0) { print(prob); stop("readr reported parsing problems; aborting.") }
n_raw <- nrow(raw)
log_msg("Loaded %s rows; %d parsing problems.", format(n_raw, big.mark = ","), nrow(prob))
stopifnot(!any(duplicated(raw$id)))

# ---------------------------------------------------------------------------
# 3. DERIVED COLUMNS (appended after the 39 raw columns)
#    year      : first 4 characters of created_datetime (UTC)
#    post_type : match_thread | daily_thread | regular   (title/flair based)
#    lang      : ISO-639-1 code from cld2, "und" when undetermined (added in §5)
# ---------------------------------------------------------------------------
raw <- raw |> mutate(year = substr(created_datetime, 1, 4))
stopifnot(all(raw$year %in% as.character(2010:2026)))

raw <- raw |> mutate(
  post_type = case_when(
    str_detect(title, MATCH_THREAD_TITLE_REGEX) |
      str_detect(link_flair_text, MATCH_THREAD_FLAIR_REGEX)      ~ "match_thread",
    str_detect(title, DAILY_THREAD_TITLE_REGEX) |
      link_flair_text %in% DAILY_THREAD_FLAIRS                    ~ "daily_thread",
    TRUE                                                          ~ "regular"
  )
)
log_msg("post_type: %s", paste(names(table(raw$post_type)), table(raw$post_type), collapse = "; "))

# Temporary numeric helpers (never written out).
num_comments_int <- suppressWarnings(as.integer(raw$num_comments))
num_comments_int[is.na(num_comments_int)] <- 0L
title_trim    <- str_trim(raw$title)
selftext_trim <- str_trim(raw$selftext)

# `reason` holds the first matching filter reason; NA = retained.
reason <- rep(NA_character_, n_raw)
assign_reason <- function(reason, cond, label) {
  hit <- is.na(reason) & cond
  reason[hit] <- label
  log_msg("  %-38s %9s rows", label, format(sum(hit), big.mark = ","))
  reason
}

# ---------------------------------------------------------------------------
# 4. FILTERS  (S1 .. S5; S6/S7 follow after language detection)
# ---------------------------------------------------------------------------

## S1 — empty content --------------------------------------------------------
log_msg("S1 empty content")
reason <- assign_reason(reason, title_trim == "" & selftext_trim == "", "S1_empty_content")

## S2 — deleted by the user ---------------------------------------------------
log_msg("S2 deleted")
reason <- assign_reason(reason, selftext_trim == "[deleted]",                 "S2_selftext_deleted")
reason <- assign_reason(reason, raw$removed_by_category %in% DELETED_RBC,     "S2_removed_by_category_deleted")
reason <- assign_reason(reason, title_trim %in% TITLE_DELETED_PLACEHOLDERS,   "S2_title_deleted_by_user")

## S3 — removed by moderators / Reddit ---------------------------------------
log_msg("S3 removed")
reason <- assign_reason(reason, selftext_trim == "[removed]",                 "S3_selftext_removed")
for (cat_ in REMOVED_RBC) {
  reason <- assign_reason(reason, raw$removed_by_category == cat_, paste0("S3_removed_by_category_", cat_))
}
reason <- assign_reason(reason, title_trim %in% TITLE_REMOVED_PLACEHOLDERS,   "S3_title_placeholder")
reason <- assign_reason(reason, str_detect(raw$selftext, SELFTEXT_REMOVED_PREFIX), "S3_selftext_removed_by_reddit")
# Safety net: any other non-empty removal category we have not listed.
other_rbc <- raw$removed_by_category != "" & !(raw$removed_by_category %in% c(DELETED_RBC, REMOVED_RBC))
if (any(other_rbc & is.na(reason))) {
  log_msg("  NOTE: unexpected removed_by_category values: %s",
          paste(unique(raw$removed_by_category[other_rbc & is.na(reason)]), collapse = ", "))
  reason <- assign_reason(reason, other_rbc, "S3_removed_by_category_other")
}

## S4 — bot-authored structural threads --------------------------------------
log_msg("S4 bot threads")
reason <- assign_reason(reason, raw$author %in% BOT_AUTHORS, "S4_bot_author")

## S5 — promotional / spam ---------------------------------------------------
log_msg("S5 promotional")
community <- is_community_domain(raw$domain)

### S5a spam accounts: statistics are computed over ALL posts of an account
### (behaviour is an account property), excluding the pseudo-author "[deleted]".
acct <- tibble(author = raw$author, domain = tolower(raw$domain),
               zero = num_comments_int == 0L, community = community) |>
  filter(author != "[deleted]") |>
  group_by(author) |>
  summarise(
    n_posts            = n(),
    zero_comment_share = mean(zero),
    top_domain         = { d <- domain[!community]; if (length(d)) names(which.max(table(d))) else NA_character_ },
    top_domain_share   = { d <- domain[!community]; if (length(d)) max(table(d)) / n() else 0 },
    .groups = "drop"
  ) |>
  mutate(is_spam_account = n_posts >= SPAM_MIN_POSTS &
                           top_domain_share >= SPAM_DOMAIN_SHARE &
                           zero_comment_share >= SPAM_ZERO_COMMENT_SHARE)
spam_accounts <- acct |> filter(is_spam_account) |> arrange(desc(n_posts))
log_msg("  spam accounts identified: %d (all three criteria)", nrow(spam_accounts))
reason <- assign_reason(reason, raw$author %in% spam_accounts$author, "S5a_spam_account")

### S5b gambling / betting-tip sites
gambling_hit <- !community & (str_detect(tolower(raw$domain), GAMBLING_DOMAIN_REGEX) |
                              str_detect(raw$title, GAMBLING_TITLE_REGEX))
reason <- assign_reason(reason, gambling_hit, "S5b_gambling")

### S5c merchandise spam
### Genuine community questions ("Does anyone know any cheap football shirt
### websites?") also mention cheap jerseys; a title containing a question mark
### is therefore exempt.  Dry-run review (2026-10-11) showed ~15% of hits were
### such questions before this exemption.
merch_hit <- (str_detect(raw$title, MERCH_REGEX) | str_detect(str_sub(raw$selftext, 1, 500), MERCH_REGEX)) &
             !str_detect(raw$title, fixed("?"))
reason <- assign_reason(reason, merch_hit, "S5c_merch_spam")

### S5d URL-only titles
reason <- assign_reason(reason, str_detect(raw$title, URL_ONLY_TITLE_REGEX), "S5d_url_only_title")

### S5e stream-piracy accounts: EVERY post of the account matches the piracy
### pattern on a non-community domain (accounts that only post piracy links).
piracy_post <- !community & str_detect(raw$title, PIRACY_TITLE_REGEX)
piracy_acct <- tibble(author = raw$author, hit = piracy_post, domain = tolower(raw$domain)) |>
  filter(author != "[deleted]") |>
  group_by(author) |>
  summarise(n_posts = n(), n_piracy = sum(hit),
            top_domain = names(which.max(table(domain))), .groups = "drop") |>
  filter(n_piracy == n_posts) |>
  arrange(desc(n_posts))
log_msg("  piracy-only accounts identified: %d (posts: %d)", nrow(piracy_acct), sum(piracy_acct$n_posts))
reason <- assign_reason(reason, raw$author %in% piracy_acct$author, "S5e_stream_piracy_account")

## S6 — encoding-damaged / unintelligible text --------------------------------
## Step 1: attempt the byte-level repair on damaged titles and bodies of rows
## still in play, logging every change.  Step 2: rows that remain damaged
## (or whose title has no letters/digits) are filtered out.
log_msg("S6 encoding")
in_play <- is.na(reason)
repairs <- list()
for (fld in c("title", "selftext")) {
  dmg <- in_play & is_damaged(raw[[fld]])
  if (any(dmg)) {
    r <- repair_text(raw[[fld]][dmg])
    fixed <- !is.na(r$encoding)
    repairs[[fld]] <- tibble(id = raw$id[dmg][fixed], field = fld,
                             original = raw[[fld]][dmg][fixed], repaired = r$text[fixed],
                             source_encoding = r$encoding[fixed])
    raw[[fld]][dmg] <- r$text                        # repaired values replace the damaged ones
    log_msg("  %-8s damaged: %d, repaired: %d, unrepairable: %d", fld, sum(dmg), sum(fixed), sum(!fixed))
  }
}
repairs <- bind_rows(repairs)
selftext_trim <- str_trim(raw$selftext)             # refresh: used by the language step
txt <- paste(raw$title, raw$selftext)
reason <- assign_reason(reason, str_detect(txt, fixed("\uFFFD")),        "S6_replacement_character")
reason <- assign_reason(reason, str_detect(txt, MOJIBAKE_REGEX),         "S6_mojibake_unrepairable")
reason <- assign_reason(reason, str_detect(txt, CONTROL_REGEX),          "S6_control_characters_unrepairable")
reason <- assign_reason(reason, str_detect(raw$title, NO_ALNUM_REGEX),   "S6_title_without_letters_or_digits")
rm(txt, in_play)

# ---------------------------------------------------------------------------
# 5. LANGUAGE IDENTIFICATION  (column `lang` for every row; filter S7)
#    cld2::detect_language returns NA when it is not confident ("unreliable").
#    We only move a post when: cld2 is confident AND not English AND the text
#    has >= LANG_MIN_ALPHA_CHARS letters AND (if cld3 is installed) cld3 agrees.
#    Short titles made of names and scores ("Bayern 2-1 Dortmund - Kane 45'")
#    are therefore kept — a deliberate, conservative choice.
# ---------------------------------------------------------------------------
log_msg("S7 language identification")
have_cld2 <- requireNamespace("cld2", quietly = TRUE)
have_cld3 <- requireNamespace("cld3", quietly = TRUE)
if (!have_cld2) stop("Package `cld2` is required: install.packages('cld2')")
# Helper: language evidence for one character vector.  Only count-based
# string operations are used (no per-row token lists) so that 2M rows fit in
# memory; a first implementation with str_extract_all() swapped heavily.
lang_evidence <- function(text) {
  n_alpha    <- str_count(text, "\\p{L}")                        # all letters
  n_nonlatin <- n_alpha - str_count(text, "\\p{Latin}")          # letters outside Latin script
  n_tokens   <- str_count(text, "\\p{L}{2,}")                    # word tokens (>= 2 letters)
  n_cap      <- str_count(text, "(?<!\\p{L})\\p{Lu}\\p{L}+")     # tokens starting with a capital
  cap_share  <- ifelse(n_tokens > 0, n_cap / n_tokens, 0)
  cld2 <- cld2::detect_language(text, plain_text = TRUE, lang_code = TRUE)   # NA = unreliable
  cld3 <- if (have_cld3) cld3::detect_language(text) else NA_character_
  agree <- if (have_cld3) (!is.na(cld3) & cld3 == cld2) else TRUE
  list(
    lang   = ifelse(is.na(cld2), "und", cld2),
    non_en = !is.na(cld2) & cld2 != "en" & agree &
             ((n_alpha >= LANG_MIN_ALPHA_NONLATIN & n_nonlatin >= LANG_MIN_NONLATIN_CHARS) |
              (n_alpha >= LANG_MIN_ALPHA_LATIN & n_tokens >= LANG_MIN_TOKENS_LATIN & cap_share < LANG_MAX_CAP_SHARE)),
    en     = !is.na(cld2) & cld2 == "en"
  )
}
log_msg("  cld2 available: %s; cld3 available: %s", have_cld2, have_cld3)
ev_title <- lang_evidence(str_squish(raw$title))
log_msg("  title evidence done")
# Body evidence only where a body exists (~31% of rows); default = not English.
body_en <- rep(FALSE, n_raw)
has_body <- selftext_trim != ""
if (any(has_body)) {
  body_txt <- str_squish(str_sub(raw$selftext[has_body], 1, 2000))   # first 2000 chars are enough
  body_en[has_body] <- !is.na(cld2::detect_language(body_txt, plain_text = TRUE, lang_code = TRUE)) &
                       cld2::detect_language(body_txt, plain_text = TRUE, lang_code = TRUE) == "en"
  rm(body_txt)
}
log_msg("  body evidence done")
# `lang` column: language of title + (truncated) body as a whole (informational, all rows)
lang_all <- cld2::detect_language(str_squish(paste(raw$title, str_sub(raw$selftext, 1, 2000))),
                                  plain_text = TRUE, lang_code = TRUE)
raw$lang <- ifelse(is.na(lang_all), "und", lang_all)
non_english <- ev_title$non_en & !body_en
reason <- assign_reason(reason, non_english, "S7_non_english")
rm(ev_title, body_en, has_body, lang_all)

# ---------------------------------------------------------------------------
# 6. SPLIT AND WRITE
# ---------------------------------------------------------------------------
stage_of <- function(r) str_extract(r, "^S[0-9]")
raw$filter_reason <- reason
stage <- stage_of(reason)

log_msg("Row accounting:")
acc <- tibble(stage = ifelse(is.na(stage), "retained", stage)) |> count(stage) |> arrange(stage)
print(acc, n = Inf)
stopifnot(sum(acc$n) == n_raw)

write_out <- function(df, path, keep_reason) {
  cols <- c(KEEP_COLS, "year", "post_type", "lang", if (keep_reason) "filter_reason")
  write_csv(df[, cols], path, na = "", progress = FALSE)
  log_msg("  wrote %-45s %9s rows", path, format(nrow(df), big.mark = ","))
}

log_msg("Writing side files")
write_out(raw[stage %in% "S2", ], SIDE_FILES[["deleted"]],     TRUE)
write_out(raw[stage %in% "S3", ], SIDE_FILES[["removed"]],     TRUE)
write_out(raw[stage %in% "S4", ], SIDE_FILES[["bot"]],         TRUE)
write_out(raw[stage %in% "S5", ], SIDE_FILES[["promotional"]], TRUE)
write_out(raw[stage %in% "S6", ], SIDE_FILES[["encoding"]],    TRUE)
write_out(raw[stage %in% "S7", ], SIDE_FILES[["non_english"]], TRUE)
if (any(stage %in% "S1")) log_msg("  NOTE: %d S1 rows dropped (no content to keep).", sum(stage %in% "S1"))

clean <- raw[is.na(stage), ]
log_msg("Writing master clean file")
write_out(clean, CLEAN_FILE, FALSE)

log_msg("Writing yearly files (overwriting in place)")
dir.create(YEARLY_DIR, showWarnings = FALSE)
for (y in sort(unique(clean$year))) {
  write_out(clean[clean$year == y, ], file.path(YEARLY_DIR, sprintf("soccer.data.cleaned-%s.csv", y)), FALSE)
}
# The v1 pipeline also wrote a second copy of the master into the yearly folder;
# it is redundant and is removed so that only one master exists.
dup_master <- file.path(YEARLY_DIR, "soccer_data.clean.csv")
if (file.exists(dup_master)) { file.remove(dup_master); log_msg("  removed redundant duplicate %s", dup_master) }

# ---------------------------------------------------------------------------
# 7. AUDIT ARTEFACTS
# ---------------------------------------------------------------------------
dir.create(ARTEFACTS, showWarnings = FALSE)
summary_tbl <- tibble(filter_reason = ifelse(is.na(reason), "retained", reason)) |>
  count(filter_reason, name = "n_rows") |>
  mutate(stage = ifelse(filter_reason == "retained", "retained", stage_of(filter_reason)),
         share_of_raw = round(n_rows / n_raw, 5)) |>
  arrange(stage, desc(n_rows))
write_csv(summary_tbl, file.path(ARTEFACTS, "filter_summary.csv"))
write_csv(spam_accounts |> select(author, n_posts, top_domain, top_domain_share, zero_comment_share),
          file.path(ARTEFACTS, "spam_accounts.csv"))
write_csv(piracy_acct, file.path(ARTEFACTS, "piracy_accounts.csv"))
write_csv(repairs, file.path(ARTEFACTS, "encoding_repairs.csv"), na = "")
log_msg("  encoding repairs logged: %d fields in %d posts", nrow(repairs), n_distinct(repairs$id))
print(summary_tbl, n = Inf)

log_msg("Done. Retained %s of %s raw posts (%.1f%%). Elapsed: %.1f min.",
        format(nrow(clean), big.mark = ","), format(n_raw, big.mark = ","),
        100 * nrow(clean) / n_raw, as.numeric(difftime(Sys.time(), t_start, units = "mins")))
