# Session 10 (2026-10-28): the QUALTAG scheme and its recoding

*Participants: Alan Nemirovski (scheme, decisions); Claude (inventory, questions, code, documents). Co-author TP (Teresa Pian)
tagged the topics; her confirmation of the bin-1 file deletion came through Alan.*

## 1. The request

Alan asked for two things:

1. Unpack TP's manually reviewed files.
2. Recode their `QUALTAG` column into separate variables, following the scheme he attached ("FIFA Qual Coding", §2), and store
   them as new columns in the same files. Ask where the delineation is unclear.

The pull before the work was a fast-forward to `f6aceb2b`. It brought two things:
- TP's `BIN3_topics_labelled_TP_FINAL.csv`;
- BA's coded outlier bands of bin 7.

AM's coded bands of bin 9 and TP's bin-9 file were already in the clone (commits of 2026-10-27).

## 2. The scheme (Alan Nemirovski, 2026-10-28; verbatim)

> Variable column: 'QUAL TAG'. All values are separated by ';'. Possible values:
> - 'WC': topic related to World Cup, could be political or non-political depending on additional tags
> - topic *is* political (Political = yes) if it contains one of the following tags: 'Host Country Domestic', 'National Identity', 'FIFA Body', 'IR'
> - topic is not political (Political = no) if it contains the tag 'non-political'
> - additional tags that qualify a topic: 'Club' if topic relates to club teams, 'Intl non-WC' if topic relates to international competition that is not the World Cup, 'player' if topic relates to specific individual, 'humour' if topic relates to humour/jokes
> - 'REVIEW': topic falls into residual category, borderline political

## 3. What the inventory found

The five files of record hold 1,385 topics: bins 1 (65), 3 (474), 5 (448), 7 (172) and 9 (226). Every tag belongs to the scheme
once case and surrounding spaces are ignored; the variants are `Player`, `Humour` and `humor`.

The rule above settles 1,289 topics. The other 96 fall into five groups:

| Case | Topics | Example |
|---|---|---|
| REVIEW alone | 8 | 3-11 `REVIEW` |
| REVIEW with non-political-dimension tags | 33 | 3-24 `WC; REVIEW`, 5-83 `Club; REVIEW` |
| REVIEW with a political tag | 14 | 7-30 `IR; REVIEW`, 9-154 `WC; National Identity; REVIEW` |
| No political tag, no `non-political`, no REVIEW | 36 | 5-237 `Club`, 3-69 `WC; player`, 7-49 `Intl non-WC` |
| A political tag and `non-political` | 3 | 3-275 `non-political; FIFA Body`, 5-222, 9-151 |
| QUALTAG blank | 2 | 5-234, 9-148 |

## 4. Decisions (Alan Nemirovski, 2026-10-28)

| # | Question | Decision |
|---|---|---|
| Q1 | Column layout | **One indicator per tag plus a summary**: `qt_wc`, `qt_political`, `qt_host_country_domestic`, `qt_national_identity`, `qt_fifa_body`, `qt_ir`, `qt_club`, `qt_intl_non_wc`, `qt_player`, `qt_humour`, `qt_review`. Declined: one column per group with `;`-lists; both layouts |
| Q2 | REVIEW alone (8) | **Flag it as review and fill in nothing else**: `qt_political` = `review`, `qt_review` = 1, every other column empty. "Do not guess or fill in values there" |
| Q3 | REVIEW with other tags (14 + 33) | **The tags given go into their indicators (1); every indicator not tagged reads `review`**; `qt_political` = `review`. This was asked in two steps, first for the 14 with a political tag and then whether the rule extends to the 33: **it applies to all 47** |
| Q4 | Topics the rule cannot decide (36 + 3) | **`qt_political` left empty and the topics listed for TP** (`QUALTAG_to_settle.csv`, with the 2 blank rows). Declined: inferring `no` for untagged topics, or letting the political tag win a conflict |
| Q5 | Two bin-3 files | **`BIN3_topics_labelled_TP_FINAL.csv` is the file of record and is recoded; `BIN3_topics_labelled_TP.csv` is removed** (`git rm`; git history keeps it) |
| Q6 | Bin 1's old `BIN1_topics_labelled_TP:BA.csv` (superseded by `…_TP_BA.csv`, whose QUALTAG differs on topics 27, 28, 34 and 64) | **Removed**, with TP's confirmation |

**Defaults stated before the questions (not objected to):**
- `QUALTAG` stays as written, and the new columns follow it directly.
- Tags are read without regard to case or surrounding spaces, and `humor`/`Humour` count as `humour`.
- Indicators are 1/0.
- An R script writes the columns; it can be re-run and keeps each file's own format (CRLF line endings, no final newline).

## 5. What was run

`code/13-recode-qualtag.R` (new) recodes every bin file it finds in `results/analysis/manual review/`. It stops when:
- a bin has two files;
- a tag lies outside the scheme;
- a file cannot be reproduced byte for byte before writing.

It is run with `Rscript code/13-recode-qualtag.R`.

**Checks:**
- **Script:** every check passes:
  - each file survives a round trip unchanged;
  - the 20 original columns have the same fingerprint after the insertion;
  - the written file reads back identical, in the same format;
  - every topic left without `qt_political` is on TP's list.
- **Independent check:** a separate Python implementation of the rules reproduced all 1,385 rows. No cell outside the new columns changed.
- **Re-run:** running the script a second time leaves the five files byte-identical.

| Bin | Topics | yes | no | review | empty (to settle) |
|---|---|---|---|---|---|
| 1 | 65 | 19 | 45 | 1 | 0 |
| 3 | 474 | 87 | 363 | 12 | 12 |
| 5 | 448 | 59 | 356 | 20 | 13 |
| 7 | 172 | 11 | 145 | 10 | 6 |
| 9 | 226 | 39 | 165 | 12 | 10 |
| **All** | **1,385** | **215** | **1,074** | **55** | **41** |

## 6. The World Cup figures (later on 2026-10-28)

**The request.** Alan asked for four figures:
1. A breakdown of the posts of the political topics by the four political categories, with the share the political posts take in their bin.
2. Mean sentiment over time in the five World Cup bins.
3. r/soccer subscribers over time.
4. Frequency ranks of the sentiment labels within each World Cup bin.

He referred to reference figures, but none came through with the message, so the layouts follow the existing figures A2, C1 and C4.

**What the data check found.** `subreddit_subscribers` exists only from March 2018, so the 2010 and 2014 World Cups have no value. It is the count when the archive fetched the post, not when the post was made. Before mid-2023 it moves in steps, and the monthly median falls in 18 of 100 months. The VADER labels rank in the same order in all five World Cup bins (neutral 46–55 %, positive 31–35 %, negative 14–19 %).

**Decisions (Alan Nemirovski, 2026-10-28):**

| # | Question | Decision |
|---|---|---|
| Q7 | Political posts: which posts, which denominator | **Posts in their cluster or singly reassigned to a political topic, as a share of all posts of the bin** (option b). The table also gives the share over the cluster + reassigned posts (a) and the share counting a dual post whose first or second topic is political (c). Declined for the figure: (a), (c) |
| Q8 | Topics with several political tags (42 of 215) | **Their posts count in each category**: side-by-side bars, and a bin's category shares can sum to more than 100 %. Declined: a "multiple" group; split weights |
| Q9 | Which "sentiment labels" to rank | **The topic tones** (`topic_tone`) of cluster + reassigned posts, ranked within each World Cup bin. Declined: the VADER labels, whose ranking is flat |
| Q10 | Subscribers | **The corpus series, March 2018 – July 2026**, with the crawl-time reading in the caption. Declined: an external historical series |

**Defaults stated with the questions (not objected to):**
- Political means `qt_political = yes`; REVIEW topics and the topics not yet settled are excluded, and the table counts them.
- Item 2 shows all posts of the World Cup bins, weekly and aligned on kick-off, one line per World Cup with a 95 % confidence band.
- A new script, `code/14-worldcup-figures.R`, writes the figures and tables to `results/descriptives/` as figures E1–E4.

**Run.**
- Every check of the script passes (join, QUALTAG coverage of every topic of the five bins, population bounds, shares).
- Each figure was rendered and inspected. Two defects were fixed: legend labels in E1, and the axis breaks of E3.
- The palette validator could not run (Node.js absent), so the project's reference palette is used unchanged, in its fixed order.

| World Cup | Posts in bin | Political posts | Share of bin (b) | Of cluster + reassigned (a) | Incl. dual (c) | National identity | FIFA body | Host country domestic | IR |
|---|---|---|---|---|---|---|---|---|---|
| 2010 | 10,002 | 1,793 | 17.9 % | 25.3 % | 23.6 % | 43.6 % | 50.3 % | 13.6 % | 3.5 % |
| 2014 | 83,147 | 9,474 | 11.4 % | 18.3 % | 17.9 % | 55.2 % | 51.7 % | 13.6 % | 5.4 % |
| 2018 | 88,648 | 5,813 | 6.6 % | 10.1 % | 10.6 % | 51.1 % | 53.3 % | 7.5 % | 17.6 % |
| 2022 | 41,438 | 990 | 2.4 % | 3.3 % | 4.0 % | 19.9 % | 52.4 % | 48.4 % | 10.2 % |
| 2026 | 37,626 | 2,799 | 7.4 % | 11.0 % | 11.9 % | 35.2 % | 57.4 % | 27.0 % | 14.5 % |

The four category columns are shares of the political posts.

**Titles only (Alan Nemirovski, 2026-10-28, after seeing the figures).** The E figures carry their bold title only: subtitles and captions were removed. The same was then done for figure A1 (`code/11-descriptives.R`). The other A–D figures keep theirs. The populations and reading notes that the captions held stay in the script header (`code/14-worldcup-figures.R`), in `results/descriptives/README.md` and in this section, for the chapter text. They cover:
- who counts as political and over which denominator;
- the shaded tournament weeks and the 95 % intervals of E2;
- the crawl-time subscriber counts, which start in March 2018;
- the reading of VADER as valence.

**First readings** (descriptive only, for the authors' interpretation):
- **Political share.** It falls from 2010 to 2022 and rises again in 2026.
- **Categories.** FIFA-body topics carry about half of the political posts in every tournament. 2022 is the only World Cup where host-country domestic topics come close to that (48 %), and national identity falls there to 20 %.
- **Sentiment.** In every World Cup except 2022 the mean title valence dips during the tournament weeks (0–4) below the eight weeks before (weighted means: 2010 0.103 → 0.077, 2014 0.117 → 0.073, 2018 0.124 → 0.100, 2026 0.104 → 0.088). In 2022 it is higher during the tournament (0.069 → 0.093). Valence is positive in every week except one 2010 week.
- **Subscribers.** About 0.89 M (March 2018) to 8.73 M (July 2026).
- **Tones.** `excited` ranks first in every World Cup. In 2010 and 2014 `enthusiastic` and `critical` follow. From 2022 `competitive` ranks second.

## 7. Open after this step

1. **For TP:** the 41 topics in `QUALTAG_to_settle.csv`: 36 untagged, 3 conflicts and 2 blank. She settles them in the bin files, and the script is re-run.
2. **Where the variables go next:**
   - The `qt_` columns live in the co-authors' review files only, not yet in `topic_crosswalk.csv`, the join table or the data files.
   - Carrying them onto the posts (via `topic_uid`) is the step that makes the political/theme engagement figures possible (E2, session 09).
   - The interim bins (2, 4, 6, 8) carry no QUALTAG.
3. **Open from earlier sessions:**
   - the outlier tiers and the 0.50 gate: the tournament bins' bands are now all coded (bin 9 by AM, the others by BA), and the interim bins' are not;
   - S6 in place;
   - codebook v3.0 final.
