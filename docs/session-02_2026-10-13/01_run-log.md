# Session 02 (2026-10-13) — Run log: completing v2.1

Continuation of session 01 (see `../session-01_2026-10-11/`). Session 01 ended with the v2.1 code in
place (encoding repair in stage S6, validation checks C6d/C6e) but the full run paused before writing.
This session runs it, validates, and finalises the codebook.

## Environment
Same as session 01 (R 4.6.1, readr 2.2.0, cld2/cld3, Python 3.14 for independent profiling, Chrome
headless for PDF). Scratch profiling scripts from session 01 were still available and reused.

## Steps

| Time | Step | Result |
|---|---|---|
| start | Read memory and docs; confirmed disk state = v2.0 (files dated 2026-10-11 18:26), no partial outputs | — |
| start | Asked and received confirmation to run the pipeline and to open a session-02 docs folder | — |
| 12:50–13:00 | `Rscript code/02-data-cleaning.R` (v2.1) | 1,302,021 retained of 2,047,322 (63.6 %); 10.1 min |
| 13:01–13:06 | `Rscript code/03-data-validation.R` | 28/28 passed |
| 13:03 | Python re-profile + extra stats of the new master | consistent with validation |
| 13:08 | Counts filled into codebook v2.1 and session-01 rules document; DRAFT banner removed; PDF re-rendered | — |
| 13:10 | Session-02 docs and memory written; persistent assistant memory updated | — |

## v2.1 row accounting

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
| S6 | `S6_control_characters_unrepairable` | 209 | 0.01% |
| S6 | `S6_title_without_letters_or_digits` | 114 | 0.01% |
| S6 | `S6_mojibake_unrepairable` | 7 | 0.00% |
| S6 | `S6_replacement_character` | 5 | 0.00% |
| S7 | `S7_non_english` | 2,790 | 0.14% |
| retained | `retained` | 1,302,021 | 63.60% |

Total 2,047,322 = raw rows (check C1). Changes vs v2.0: S6 461 → 335 rows (126 repaired and retained); S7 2,786 → 2,790 (4 repaired titles were then confidently non-English); clean master 1,301,899 → **1,302,021**.

## Encoding repair outcome (stage S6)

126 damaged titles (of 337 in play) were repaired by strict byte-level re-decoding; 0 of 10 damaged bodies
(the damage there is a stray control byte with no recoverable source encoding). Source encodings that worked:
ISO-8859-1 89, Windows-1252 25, Windows-1250 12. What the repairs changed, grouped:

| Count | Original fragment | Repaired |
|---|---|---|
| 50 | `â€™` | `’` |
| 18 | `Â` + no-break space | no-break space |
| 12 | `ÂŁ` | `£` |
| 9 | `â€“` | `–` |
| 3 | `â€”` | `—` |
| 2 | `Ã‘` | `Ñ` |
| 4 | `â€˜ … â€™` | `‘ … ’` |
| 1 | `â—` (×5) | `●` |
| 1 | `Ã®` | `î` |
| 1 | Vietnamese mojibake (`ná»\x99i y nÃ³ng …`) | `nội y nóng …` |

Every repair is logged in `data/data artefacts/encoding_repairs.csv` and validation check C6d re-derived all
126 originals from the repaired text. Still filtered (335): 209 titles with an apostrophe followed by stray C1
bytes (`'\x80\x99` — the leading byte was lost upstream, so no re-decoding is possible), 7 unrepairable
mojibake, 5 with U+FFFD, 114 emoji/punctuation-only titles.

## Validation

`Rscript code/03-data-validation.R` — **28 of 28 checks passed** (26 from v2.0 plus C6d repair reversibility and C6e repaired values present). Independent Python re-profile of the master: 0 removal markers, 0 placeholder titles, 0 bot authors, 0 U+FFFD, all 199 months present, date range 2010-01-01 02:34:08 – 2026-07-31 23:32:20.
