##############################################################################
# FIFA COMMUNITIES CHAPTER
## code/R/qualtag-utils.R — the QUALTAG scheme and its parser, shared by 13-recode-qualtag.R and 17-topic-tags.R
##############################################################################
#
# The co-authors' qualitative tags ("FIFA Qual Coding", Alan Nemirovski, 2026-10-28; docs/session-10_2026-10-28/
# 00_approvals.md): a ';'-separated list in the QUALTAG column of TP's review files (results/analysis/manual review/).
# Moved out of 13-recode-qualtag.R on 2026-10-31 (session 14) when 17-topic-tags.R began to read the same column,
# so that the two scripts cannot drift.  Sourced after pipeline-utils.R.
##############################################################################

# The scheme: tag as read (lower case, single spaces) -> its indicator column in the review files
POLITICAL_TAGS <- c("host country domestic" = "qt_host_country_domestic", "national identity" = "qt_national_identity",
                    "fifa body" = "qt_fifa_body", "ir" = "qt_ir")
QUALIFIER_TAGS <- c("club" = "qt_club", "intl non-wc" = "qt_intl_non_wc", "player" = "qt_player", "humour" = "qt_humour")
TAG_COL        <- c("wc" = "qt_wc", POLITICAL_TAGS, QUALIFIER_TAGS, "review" = "qt_review")
NONPOL         <- "non-political"                  # no column of its own: it decides qt_political
SPELLINGS      <- c("humor" = "humour")            # variants found in the files, read as the scheme's spelling

# One QUALTAG cell -> its tags: split on ';', trimmed, single-spaced, lower case, spelling variants mapped, duplicates dropped.
# A blank cell gives character(0).
parse_tags <- function(x) {
  p <- tolower(gsub("\\s+", " ", trimws(strsplit(x, ";", fixed = TRUE)[[1]])))
  p <- p[p != ""]                                  # a trailing ';' leaves an empty part
  unique(unname(ifelse(p %in% names(SPELLINGS), SPELLINGS[p], p)))
}
