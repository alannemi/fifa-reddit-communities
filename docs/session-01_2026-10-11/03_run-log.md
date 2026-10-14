# Session 01 (2026-10-11) — Run log

Chronological record of what was executed, with outputs. Times are local (America/Toronto).
Scratch profiling scripts (Python) were run from a session scratchpad and are summarised in
`01_data-audit-findings.md`; only repository scripts are listed here.

## Environment
R 4.6.1 (arm64 macOS 15, 16 GB RAM); readr 2.2.0, dplyr 1.2.1, stringr 1.6.0, purrr, jsonlite 2.0.0,
commonmark; `cld2` and `cld3` installed from CRAN during the session; Python 3.14 (profiling only);
Google Chrome 154 (headless, PDF rendering). pandoc/quarto not installed.

## Steps

| Time | Step | Result |
|---|---|---|
| 15:50–16:40 | Audit of all files in `data/` (streaming profiles; fread vs readr test on 30k rows) | findings in `01_data-audit-findings.md`; quote corruption confirmed |
| 16:45 | Plan presented and approved with amendments (see `02_cleaning-rules-and-rationale.md` §1) | — |
| 16:57 | `install.packages(c("cld2","cld3"))` | both installed |
| 17:00–17:03 | Dry runs of `code/02-data-cleaning.R` on the first 60,000 raw rows (`CLEAN_NMAX`, `CLEAN_OUT_ROOT`) | logic verified; non-English rule tightened (name-heavy titles were mislabelled) |
| 17:03–17:19 | **Full run #1** | 1,301,888 retained; 16.3 min; 26/26 validation checks passed |
| 17:20 | Review of 25-row samples per rule from run #1 | merch rule caught genuine questions (~15 %); non-English rule caught mixed-language posts (~20 %) → both rules refined |
| 17:24 | Dry run of refined rules | OK |
| 17:24–18:17 | Full run #2 (refined rules) — **aborted** | language step swapped heavily (6.6 GB swap in use) because it built per-row token lists; rewritten with count-based regexes |
| 18:17 | Dry run of optimised language step | identical result, seconds instead of minutes |
| 18:17–18:26 | **Full run #3 (final)** | 1,301,899 retained of 2,047,322 (63.6 %); 9.4 min |
| 18:27 | `Rscript code/03-data-validation.R` | **26/26 checks passed** |
| 18:28 | Independent Python re-profile of the new master | 0 removal markers, 0 placeholder titles, 0 bot authors, 0 U+FFFD, all 199 months present, date range 2010-01-01 02:34:08 – 2026-07-31 23:32:20 |
| 18:35 | Codebook v2.0 written and rendered; docs and memory folders written | — |

| 18:35–18:44 | Follow-up: gambling rule confirmed; S6 rewritten to repair encoding damage by strict byte-level re-decoding (test on the 461 v2.0 S6 rows: 126 of 337 damaged titles repairable — 89 ISO-8859-1, 25 Windows-1252, 12 Windows-1250; 0 of 10 damaged bodies); validation checks C6d/C6e added; dry run OK | code ready |
| 18:45–18:53 | Full run #4 (v2.1) started, **stopped at the user's request before any file was written** (language stage) | disk remains at v2.0 |

## Final row accounting (run #3 = v2.0, current files on disk)

| Stage | filter_reason | Rows | Share of raw |
|---|---|---|---|
| S2 | `S2_selftext_deleted` | 234,217 | 11.44% |
| S2 | `S2_removed_by_category_deleted` | 34,091 | 1.67% |
| S2 | `S2_title_deleted_by_user` | 15,125 | 0.74% |
| S3 | `S3_selftext_removed` | 173,511 | 8.48% |
| S3 | `S3_removed_by_category_moderator` | 109,757 | 5.36% |
| S3 | `S3_removed_by_category_reddit` | 48,733 | 2.38% |
| S3 | `S3_removed_by_category_automod_filtered` | 42,493 | 2.08% |
| S3 | `S3_removed_by_category_copyright_takedown` | 866 | 0.04% |
| S3 | `S3_selftext_removed_by_reddit` | 180 | 0.01% |
| S3 | `S3_removed_by_category_content_takedown` | 72 | 0.00% |
| S3 | `S3_title_placeholder` | 1 | 0.00% |
| S4 | `S4_bot_author` | 48,156 | 2.35% |
| S5 | `S5b_gambling` | 16,709 | 0.82% |
| S5 | `S5a_spam_account` | 16,396 | 0.80% |
| S5 | `S5c_merch_spam` | 926 | 0.04% |
| S5 | `S5e_stream_piracy_account` | 579 | 0.03% |
| S5 | `S5d_url_only_title` | 364 | 0.02% |
| S6 | `S6_control_characters` | 293 | 0.01% |
| S6 | `S6_title_without_letters_or_digits` | 114 | 0.01% |
| S6 | `S6_mojibake` | 54 | 0.00% |
| S7 | `S7_non_english` | 2,786 | 0.14% |
| retained | `retained` | 1,301,899 | 63.59% |

Total: 2,047,322 = raw rows (identity verified by validation check C1).

Output files (all overwritten in place; redundant `yearly datasets/soccer_data.clean.csv` deleted):
`data/soccer_data.clean.csv` (1,301,899 rows, 42 cols), 17 yearly files (same spec), `deleted_posts.csv`
(283,433), `removed_posts.csv` (375,613), `bot_threads.csv` (48,156), `promotional_posts.csv` (34,974),
`encoding_issue_posts.csv` (461), `non_english_posts.csv` (2,786); audit tables in `data/data artefacts/`.

## Precision checks on random samples (25 rows per rule, run #3 unless noted)

| Rule | Genuine hits / sample | Notes |
|---|---|---|
| S5a spam account | 25/25 | content farms, prediction sites, self-promoting blogs with no engagement |
| S5b gambling | ~22/25 | false positives are news items on bookmaker/poker domains (recoverable via `filter_reason`) |
| S5c merch spam | 24/25 | after exempting questions; one 2014 news item about a fan selling shirts |
| S5d url-only title | 25/25 by definition | includes a few genuine link posts with a bare URL as title (accepted per decision D6) |
| S5e piracy account | 24/25 | one local-news "how to watch" item from a one-post account |
| S6 encoding | 25/25 damaged | most are single garbled apostrophes in otherwise readable headlines (kept intact in the side file for possible repair) |
| S7 non-English | 25/25 | run #1 rule had ~20/25; final title-based rule has no false positive in the sample |

## Known residuals in the clean master
* ~107 titles still contain `Ã`/`â€` sequences in positions the mojibake regex deliberately does not
  match (letter follows, e.g. Portuguese "SÃO"); treat as acceptable.
* `lang != en/und` for 80,416 clean posts — mostly proper-noun titles (goal clips) that detectors
  mislabel; kept by design.
