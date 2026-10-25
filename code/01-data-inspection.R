##########################################
# FIFA COMMUNITIES CHAPTER
## 1. DATA LOADING AND INSPECTION
##########################################
#
# Loads the raw Arctic Shift export of r/soccer posts (2,047,322 posts,
# 1 Jan 2010 - 31 Jul 2026, 64 columns) for interactive inspection.
#
# IMPORTANT — do NOT use data.table::fread() on this file.
# ------------------------------------------------------
# The v1 pipeline (Aug 2026) used fread(), which on this file fails to
# un-escape RFC-4180 doubled quotes: every literal quotation mark in titles,
# selftext and the JSON columns came through doubled ("" instead of ") and
# the JSON flair/media fields became unparseable.  readr::read_csv() parses
# the file correctly (0 problems).  Details: docs/session-01_2026-10-11/.
#
# Everything is read as character so that nothing is coerced or reformatted;
# convert columns explicitly (as.integer(), as.POSIXct(..., tz = "UTC")) when
# an analysis needs numbers or dates.  Expect ~6-8 GB of RAM for the full file;
# use `n_max = 100000` for a quick look.

library(tidyverse)   # readr, dplyr, stringr, ...

# Run from the repository root (open fifa-reddit-communities.Rproj); the data folder is copied into it
# from the shared folder (data/README.md).
if (!file.exists("data/soccer_posts.csv")) stop("data/soccer_posts.csv not found: open the .Rproj or setwd() to the repository root")

soccer.data <- read_csv(
  "data/soccer_posts.csv",
  col_types = cols(.default = col_character()),  # verbatim, no coercion
  na        = character(),                       # keep "" as "" (raw convention)
  progress  = FALSE
  # , n_max = 100000                               # uncomment for a quick look
)

stopifnot(nrow(problems(soccer.data)) == 0)      # readr must report no parsing problems
glimpse(soccer.data)
View(soccer.data)
