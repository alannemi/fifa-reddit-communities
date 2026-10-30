# Manual review

Folder for the co-authors' manual review materials (created at Alan Nemirovski's request, 2026-10-25). Git does not keep an
empty folder, hence this file.

The pipeline itself does not read anything here: the files it applies are the per-bin `results/analysis/bin-N/topic_overrides.csv`
(renames and merges) and, for the record, the coded judgement samples in `results/analysis/bin-N/outlier_calibration/`. Guide:
`docs/session-08_2026-10-25/01_review-guide-bin-1.md`.

## What is here

| File | Content |
|---|---|
| `BIN<N>_topics_labelled_*.csv` | One file per tournament bin (1, 3, 5, 7, 9): the bin's labelled topics with TP's qualitative tags in `QUALTAG` (bin 1 with BA). One file per bin: a superseded version is removed (`BIN1_…_TP:BA.csv` and `BIN3_…_TP.csv` on 2026-10-28; git keeps them) |
| `QUALTAG_to_settle.csv` | The topics whose tags do not decide whether they are political: `bin, topic_id, name, QUALTAG, issue`. Written by `code/13-recode-qualtag.R`; TP settles them in the bin files, then the script is re-run |
| `sentiment_review_coder_A-coded-AN.csv` | Alan Nemirovski's coding of the 337-post VADER validation sample (session 09) |

## The QUALTAG variables (`code/13-recode-qualtag.R`, 2026-10-28)

The script adds 11 columns right after `QUALTAG` in each bin file. `QUALTAG` itself and every other column stay exactly as
the co-authors wrote them. The scheme and the rules are Alan Nemirovski's (`docs/session-10_2026-10-28/00_approvals.md`):

| Column | From the tag(s) | Values |
|---|---|---|
| `qt_wc` | `WC` (topic related to the World Cup) | 1 / 0 / review / empty |
| `qt_political` | `Host Country Domestic`, `National Identity`, `FIFA Body`, `IR` → yes; `non-political` → no; `REVIEW` → review | yes / no / review / empty |
| `qt_host_country_domestic`, `qt_national_identity`, `qt_fifa_body`, `qt_ir` | the four political tags, one column each | 1 / 0 / review / empty |
| `qt_club`, `qt_intl_non_wc`, `qt_player`, `qt_humour` | the qualifiers: club teams; international competition other than the World Cup; a specific individual; humour | 1 / 0 / review / empty |
| `qt_review` | `REVIEW` (residual category, borderline political) | 1 / 0 / empty |

How a value is set:

- **Without REVIEW**, a tag present is 1 and a tag absent is 0. `qt_political` is `yes` with a political tag only and `no` with
  `non-political` only. With both, or with neither, it stays **empty** and the topic goes on `QUALTAG_to_settle.csv`.
- **REVIEW with other tags**: the tags given are 1, every indicator not tagged reads `review` (no 0 is asserted under the
  coder's doubt), and `qt_political` is `review`.
- **REVIEW alone**: `qt_political` = `review`, `qt_review` = 1, every other column empty.
- **Blank QUALTAG**: every column empty; listed for TP.

Tags are read without regard to case or surrounding spaces, and `humor` counts as `humour`. A tag outside the scheme stops the
script. Read the files as text, as all the project's CSVs (`readr::read_csv(…, col_types = cols(.default = col_character()),
na = character())`), and compare with `"1"`, `"0"` and `"review"`.

Re-run after any change to a bin file (it recomputes its own columns and leaves the rest of each file byte for byte as it was):

```bash
Rscript code/13-recode-qualtag.R
```

`code/16-reference-map.R` (session 12, 2026-10-30) reads these files read-only — TP's by their `QUALTAG` column, BA's bin-2 and
bin-4 files by their `qt_` columns (`1` / `0` / blank) — to table the QUALTAG composition of the whole-corpus reference topics
(`results/analysis/reference/reference_topics.csv`); it changes nothing here and writes no tag anywhere (decision W2).

## The outlier judgement samples (coded; session 11, 2026-10-30)

They live with their bin, not here: `results/analysis/bin-N/outlier_calibration/sample_T<threshold>*coded*<initials>.csv`, one
file per similarity band (35th, 50th, 65th, 80th percentile of the bin's outlier-to-nearest-topic similarity; 100 posts each),
beside the blank file the pipeline wrote and the band table `calibration_table.csv`. Coders: BA (bins 1, 3, 5, 7), AM (bins 6, 8,
9), AN (bins 2, 4); `belongs` = 1/0. The pipeline never rewrites a file with `coded` in its name (any separator or suffix).
`code/15-outlier-calibration.R` summarises them (`results/descriptives/tab_F1`–`tab_F4b`, figures F1–F2); the assessment and
the decisions taken on them (K1–K7) are in `docs/session-11_2026-10-30/`. Inter-coder checks are a step of the chapter's editing
process (K2).
