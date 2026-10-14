# Session 01 (2026-10-11) — Data audit findings

Audit of every file under `data/` before any change was made. All counts were produced
by streaming the CSVs with Python's `csv` module (field-size limit raised; multi-line
`selftext` handled by the parser) and by `readr`/`data.table` tests in R 4.6.1.

## 1. Inventory before the session

| File | Rows | Columns | Writer | Note |
|---|---|---|---|---|
| `data/soccer_posts.csv` (raw) | 2,047,322 | 64 | Python `csv` (`jsonl_to_csv.py`) | correct; 0 duplicate ids |
| `data/soccer_data.clean.csv` | 1,591,577 | 40 (39 + `year`) | `readr::write_csv` | **quote-corrupted** (see §2) |
| `data/yearly datasets/soccer.data.cleaned-2010..2026.csv` | 1,591,577 in total | 40 | base `write.csv` | quote-corrupted; different format from master |
| `data/yearly datasets/soccer_data.clean.csv` | 1,591,577 | 39 (no `year`) | `readr::write_csv` | redundant duplicate of the master |
| `data/removed_posts.csv` | 221,529 | 40 | readr | selftext == "[removed]" only |
| `data/deleted_posts.csv` | 234,216 | 40 | readr | selftext == "[deleted]" only |
| `data/data artefacts/POSTS-*.jsonl` (3 batches) + `soccer_posts.csv.gz` | — | — | collector | untouched source artefacts |

raw = clean + removed + deleted (2,047,322 = 1,591,577 + 221,529 + 234,216) — the v1 split was complete
with respect to its own rule.

## 2. Severe: every literal quotation mark was doubled in all cleaned files

* Raw title (id `88jfkp`): `Ronaldo called his friend Edinho to tell him: "Old carcass, …"`
* Clean/yearly title: `… tell him: ""Old carcass, …""`
* Raw `author_flair_richtext` (id `eucmc`): `[{"e": "text", "t": "Arsenal"}]`
* Clean/yearly: `[{""e"": ""text"", ""t"": ""Arsenal""}]` — invalid JSON.

Cause: `data.table::fread()` (v1.18.4) does not un-escape RFC-4180 doubled quotes on this file
(reproduced on the first 30,000 rows: 1,926 titles contain a quote, 1,926 of them doubled).
`readr::read_csv()` on the same rows: 7 doubled quotes, all genuinely doubled in the raw file; 0 parse problems.
Impact: all quoted speech in titles/selftext (≈6% of titles in the sample), every JSON column
(`author_flair_richtext`, `link_flair_richtext`, `media`, `preview`) unparseable.
Decision: rebuild all cleaned files from the raw export with `readr` (approved by A. Nemirovski, 2026-10-11).

## 3. Moderate: removed/deleted detection missed link posts

The v1 rule keyed only on `selftext`, which is empty for link posts (88% of posts). Still present in
the v1 clean file:

| `removed_by_category` | rows in v1 clean |
|---|---|
| moderator | 109,757 |
| reddit (spam filter) | 48,733 |
| automod_filtered | 42,493 |
| deleted | 933 |
| copyright_takedown | 866 |
| content_takedown | 72 |
| author | 24 |
| **total** | **202,878** |

plus placeholder-only content: title `[ Removed by moderator ]` (294), `[deleted by user]` (241),
`[ Removed by Reddit ]` (70), `[image processing failed]` (≈150 in raw), and selftext beginning
`[ Removed by Reddit on account of violating …` (title shows `[deleted by user]`).

Engagement of these posts (num_comments): automod_filtered median 1 (p90 1); reddit median 0;
moderator median 3 (p90 40); i.e. mostly never part of the visible discourse. Decision: move all of
them to the removed/deleted side files (approved).

## 4. "Both content fields empty": none

0 rows in raw and 0 in clean have an empty `title` and empty `selftext` (Reddit requires a title).
The rule is implemented anyway and reports 0.

## 5. Moderate: two writers, two formats

| Aspect | master (`readr`) | yearly (base `write.csv`) | raw / codebook |
|---|---|---|---|
| `created_datetime` | `2018-03-31T18:02:01Z` | `2010-12-31 23:12:07` (12 rows date-only, e.g. `2012-05-14`) | `2018-03-31 18:02:01` |
| booleans (`is_self`, …) | `TRUE`/`FALSE` | `TRUE`/`FALSE` | `True`/`False` |
| missing numeric (`downs`, `total_awards_received`) | `NA` | `NA` | empty |
| missing text | empty | `""` | empty |
| quoting | minimal | all strings quoted | minimal |

Decision: one writer (`readr::write_csv`, `na = ""`) for every output; all columns read and written
as character so raw conventions pass through verbatim.

## 6. Coverage and completeness — no severe problem

* Monthly post counts are continuous from 2010-01 to 2026-07 with no gaps at the batch boundaries
  (2018-03-31, 2026-06-09). Visible dips correspond to real-world events (e.g. 2020-04/05 = 5.1k/5.3k vs
  ~12k in 2020-03, COVID suspension). 2010 is thin (14k posts; subreddit was small).
* 0 duplicate ids; min date 2010-01-01 02:34:08; max date 2026-07-31 23:59:37 (UTC).
* **The corpus contains posts only. No comment text exists in `data/`, the JSONL batches or the
  collector folder** (`reddit-soccer-collector/data/collected` holds only 3 days of June 2026 scanner
  output, 66 MB). `num_comments` is the only comment signal (97.7M comments referenced; median 6 per post;
  17.7% of posts have 0 comments).
* Only 11.8% of posts have body text (`selftext`); for 88% the analysable text is the title
  (median 62 characters, p90 155). This is a design constraint for topic modelling/sentiment, documented
  in the codebook; it is not a cleaning defect.

## 7. Content worth separating (basis for the promotional / bot rules)

* Bot authors: `MatchThreadder` 38,402 posts, `AutoModerator` 5,312, `2soccer2bot` 3,320, `soccerbot` 1,139.
  Their post text is template boilerplate; 48k posts carry 18% of all referenced comments.
* Account–domain concentration: of 1,619 accounts with ≥100 posts, 102 accounts post ≥80% of their links
  to a single non-community domain **and** have ≥90% zero-comment posts (31,844 posts; e.g. `espnsoccer`
  → espnsoccer.co 5,294 posts, 99.4% zero comments; `evblogs` → e360hubs.com; `newslodge` → newslodge.com.ng;
  `fvhighlights` → footballvideohighlights.com; Indonesian "prediksi" sites). Legitimate single-outlet
  accounts (`TheTelegraph` 571 posts, 0.2% zero comments; `lordjems` → theguardian.com, 0%) are clearly
  separated by the engagement criterion — this gap motivates the three-criteria rule.
* Gambling/prediction spam: ≈18k posts on domains with gambling tokens (bos303.com, prediksibolaterbaik.com,
  kingbola99.org, beritabola88.com, …) + ≈5k with Indonesian betting phrases in the title.
* Merchandise spam: ≈2k (e.g. 1,500+ near-identical `four4two.com` jersey posts; `jerseymate.com`, `skygoal.se`).
* URL-only titles: 610 (mostly betting/affiliate links).
* Stream-piracy titles ("… live stream free online HD"): 1,624 posts.
* Keyword-only rules were tested and rejected: "betting", "advert", "survey", "I made" hit genuine news and
  community content at high rates.

## 8. Minor observations (documented, not filtered)

* 117,793 posts by `[deleted]` authors whose content is intact — kept (content integrity).
* 180 rows with cp1252 mojibake (`Ã©`, `â€™`), 5 with U+FFFD, a few with C1 control characters — the user
  asked for these to be moved to a side file (session decision), see rules doc.
* `link_flair_text` labels drift over time (`Official source` / `Official Source` / `:n_official: Official Source`).
* `edited` holds `False`, `True` (2,262 old rows) or a Unix timestamp.
* `subreddit_subscribers` empty before ~2018; `ups` == `score`; `downs` always 0/empty (Reddit vote fuzzing).
* `num_crossposts > 0` for 95,925 posts means the post was cross-posted elsewhere, not that it is a cross-post.
* Reposts: 33,416 URLs were submitted 2+ times (74,818 posts) — genuine community behaviour, kept.

## 9. Tooling facts

R 4.6.1; readr 2.2.0; dplyr 1.2.1; stringr 1.6.0; data.table 1.18.4 (not used for loading any more);
cld2 and cld3 installed from CRAN during the session; Python 3.14 for independent profiling; 16 GB RAM;
pandoc not installed (codebook PDF rendered via headless Chrome).
