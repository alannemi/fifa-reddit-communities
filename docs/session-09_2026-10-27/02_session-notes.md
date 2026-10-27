# Session 09 (2026-10-27): tone vocabulary, sentiment coding pack and descriptive figures

*Record of what was run and what came out, for the co-authors. Decisions: `00_approvals.md` (E1–E7). Coders' guide: `01_sentiment-coding-guide.md`.*

## 1. Tone vocabulary extended (decision E6)

**What changed in the code:**
- `config.TONE_SET` gains `competitive`, `informative`, `analytical` and `concerned`.
- A new list, `config.TONE_STEMS_ORDERED`, holds their stems: `competit`, `informat` and `informed`, `analyt`, `concern`.
- `agents.tone_category` checks the existing `TONE_STEMS` first, so no earlier category changes. Only then does it check the new stems, choosing the one written first in the tone ("analytical and competitive" → analytical).
- A new stage, `--stage retone`, re-derives `tone_category` in each bin's `topics_labelled.csv` from `labels.jsonl` (`agents.retone_labels`). It makes no model call. It refuses to write if any other column would change, or if a topic would leave a category other than `other`. Its run is recorded in `config.json → retone`, and the label record is untouched.
- The unit checks are now 14, all passing. The new check covers the mapping and the stage's refusals.

**Pre-check.** Rebuilding the nine `topics_labelled.csv` from `labels.jsonl` with the code before the change reproduced every file byte for byte, so the stage's diff is exact.

**Run (local, 13:14 EDT).**

| Bin | Topics moved out of `other` | competitive | informative | analytical | concerned |
|---|---|---|---|---|---|
| 1 | 6 of 65 | 0 | 2 | 3 | 1 |
| 2 | 76 of 916 | 5 | 29 | 17 | 25 |
| 3 | 53 of 474 | 9 | 16 | 14 | 14 |
| 4 | 208 of 1,602 | 35 | 80 | 41 | 52 |
| 5 | 76 of 448 | 24 | 16 | 19 | 17 |
| 6 | 283 of 1,375 | 86 | 109 | 50 | 38 |
| 7 | 47 of 172 | 15 | 22 | 8 | 2 |
| 8 | 213 of 967 | 78 | 81 | 37 | 17 |
| 9 | 42 of 226 | 18 | 14 | 6 | 4 |
| **All** | **1,004 of 6,245** | **270** | **369** | **195** | **170** |

The transitions equal the preview computed before writing. `08-review-pack.py check` is clean for all nine bins, and the review packs were refreshed.

**Collect, run again locally:**
- `complete = true`, 1,302,021 rows, 6,245 topics;
- **labels_version `f2bc460bf46d`** (was `1d6f421094b8`);
- column hashes before and after show that **only `topic_tone` and `topic_tone_2` changed**.

Posts whose first topic has the tone `other` fell from 316,401 to 102,479.

**What is still `other`.** Among the cluster and reassigned posts, 71,299 posts still have the tone `other`. The largest tones are urgent (4.6k posts), ambivalent (3.9k), controversial (3.7k), emotional (2.7k) and nostalgic (2.6k); a long tail follows. Full table: `results/descriptives/tab_C_other_contents.csv`.

**S6 chain test.** `06-tag-corpus.R` and `07-validate-analysis.R` were run into a scratch root on the new join table: 18 files tagged, **ALL CHECKS PASSED** (21 checks). S6 in place has not been run (see §6).

## 2. Sentiment validation (decision E1)

- `code/09-sentiment-coding-pack.R` wrote `results/analysis/manual review/sentiment_coding/coder_A.csv` and `coder_B.csv`. They are identical (same md5) and contain the 337 ids of `data/analysis/sentiment_validation_sample.csv`. The columns are `id, title, body_excerpt, is_self, code, notes`, with no VADER column. 58 rows carry a body excerpt; 10 of the 68 self posts have an empty body.
- `code/10-sentiment-validation.R` reports:
  - agreement and Cohen's κ, with bootstrap 95 % intervals, for A vs B, A vs VADER, B vs VADER, and the reference vs VADER;
  - the confusion matrix;
  - precision, recall and F1 per class;
  - agreement by bin and by `is_self`.
  
  The reference label is the common code; disagreements count only once they are settled in `adjudicated.csv`.
- **Tested on simulated codes** (mixed spellings, half of the disagreements adjudicated). The κ values equal an independent Python computation (A vs B 0.8004; reference vs VADER 0.6916), and the confusion matrix totals the 315 reference posts.
- **Waiting on:** the two coders, assigned by Alan. The guide estimates about 1.5 h per sheet.

## 2b. Sentiment validation result (Alan's coding, returned 2026-10-27 14:37)

**The sheet.** The coded sheet is `results/analysis/manual review/sentiment_review_coder_A-coded-AN.csv`. It was kept under Alan's file name, in the manual-review folder (his decision); `10-sentiment-validation.R` reads it by that name. The old `sentiment_coding/` folder and its blank sheet are gone (git history keeps them).

**Checks.** All 337 ids match the sample; the file is UTF-8 with CRLF line endings; every row is coded; 18 posts carry notes. Two titles and eight bodies lost a leading or trailing line break in the spreadsheet. That is harmless: the script reads only `id` and `code`. One code read `neural` (post `1yx18c`, an infographic on transfer fees) and was **corrected to `neutral` with Alan's confirmation**.

**Result (2026-10-27; coder: Alan Nemirovski; `results/descriptives/sentiment_validation_*.csv`).** The human coding labels 258 of the 337 titles neutral (77 %), 43 positive and 36 negative; VADER labels 159 neutral, 114 positive and 64 negative. Agreement is **56.4 %**, Cohen's **κ = 0.24** (bootstrap 95 % interval 0.17–0.31). By class (VADER against the human reference): neutral precision 0.91, recall 0.56; positive precision 0.25, recall 0.67; negative precision 0.25, recall 0.44 (macro-F1 0.46). The main error is sentiment where the reader sees none: 113 of the 258 neutral titles receive a positive (72) or negative (41) score, mostly news reported in emotive vocabulary ("unbeaten", "champions", "cheer"; "penalty", "allegations", "loser"); 13 negative titles are scored positive (for example a title about a player injured by a *celebrating* fan). Agreement varies between 47 % and 75 % across the bins (12–77 posts each) without a trend, and is the same for link and self posts (56 % and 57 %). The figures are those of the sample, whose VADER-label mix (47 / 34 / 19 % neutral / positive / negative) is close to the corpus's (50 / 34 / 16 %). VADER therefore measures the valence of a title's words rather than the stance of its author.

**Alan's notes point the same way:**
- "Neutral presentation of negative content" (11 posts);
- "Second-hand account" (6);
- "Neutral presentation of mixed content" (5).

The coding rule (code the author's stance, not the event) separates stance from content; VADER scores content words. **Decision E9 (Alan Nemirovski, 2026-10-27):** keep the VADER variables, reframed as the *sentimental valence of posts* and not an authoritative measure of the author's sentiment, with this validation reported. The analysis relies mainly on topic-level and other aggregates, together with the topic tones. No additional validation.

**Threshold sensitivity check (Alan's request, combined with E9; `10-sentiment-validation.R` §3b → `results/descriptives/sentiment_validation_thresholds.csv`):**

| Neutral band | Agreement | κ | Corpus neutral |
|---|---|---|---|
| ±0.05 (in force) | 56.4 % | 0.24 | 50 % |
| ±0.20 | 61.1 % | 0.26 | 55 % |
| ±0.35 | 67.4 % | 0.28 (highest) | 64 % |
| ±0.50 | 72.7 % | 0.23 | 78 % |
| ±0.60–0.75 | 73–76 % | 0.09–0.12 | 87–95 % |

Agreement follows the share labelled neutral (the human majority class, 77 %). κ stays inside the ±0.05 interval (0.17–0.31) up to ±0.45 and falls beyond. No cut-off fixes the disagreement, which lies in the construct (wording valence vs the author's stance). **The ±0.05 encoding is kept**, and the framing as sentimental valence stands. The bands are symmetric (VADER's convention); no cut-off was optimised, since 337 posts coded by one person would overfit it.

## 3. Descriptive figures (`code/11-descriptives.R`, ~1 min)

All figures and tables are in `results/descriptives/` (index: `README.md` there). The script's checks pass:
- every post joined once;
- the monthly counts sum to 1,302,021;
- the tone population is 827,744 posts, each with a tone;
- the tone shares sum to 1 per bin.

Every PNG was inspected.

| Figure | Content | Population |
|---|---|---|
| A1 | Posts per month, 2010–2026, World Cup bins shaded | all 1,302,021 |
| A2 | Posts per week around each World Cup, aligned on kick-off (whole weeks only) | the five World Cup bins |
| B1 | Median comments per post per month, with IQR | all |
| B2 | Median score per month, with IQR; the view is clipped above the second-highest month (December 2022, upper quartile 1,794, is named in the caption) | all |
| B3 | Median and IQR of both metrics per period bin | all |
| C1 | Mean VADER title score per month, with 95 % CI | all |
| C2 | Monthly shares of positive / neutral / negative titles | all |
| C3 | Tone composition per bin (heatmap of shares, all 20 tones) | cluster + reassigned, 827,744 |
| C4 | Rank of the tones that reach the top five in any bin | cluster + reassigned |

**First readings** (descriptive only; for the authors' interpretation):
- **Volume.** (From July 2023 partly an artefact of the crawl regime, §4.) Volume rises to a peak in June 2014 (≈ 17,700 posts) and stays high through 2019. It drops sharply in spring 2020 (COVID suspension) and stays at about 5,000–7,500 posts per month after 2020. The 2010, 2014 and 2018 tournaments show clear weekly peaks at kick-off. The 2022 World Cup (Qatar, November–December) shows no rise, and 2026 rises mildly.
- **Engagement.** Median comments per post go from 2–3 (bins 1–3) to 15–18 (bins 6, 7, 9). Median score goes from 1–4 to 94 in bin 7. These levels depend on the crawl regime (§4).
- **VADER.** The monthly mean title score is positive throughout (0.05–0.15), with a slight decline: 0.105 in 2010–14, 0.096 in 2015–19, 0.093 in 2020–26. The neutral share rises from about 45 % to 55–60 % after 2022, mostly at the expense of positive titles. The negative share is stable at 13–20 %.
- **Topic tones.**
  - `excited` is the most frequent tone in every bin (22–36 %).
  - `competitive` climbs from rank 16–18 (bins 1–3) to rank 2 (bins 6–9).
  - `critical` and `enthusiastic` fall from ranks 2–3 in the early bins to ranks 4–9 later.
  - `neutral` rises from rank 19 to ranks 3–5.

## 3b. Topic network (`code/12-topic-network.R`; author's request: edges by semantic similarity, static figure)

**Method.**
- **Nodes:** the 6,245 labelled topics, sized by their HDBSCAN cluster.
- **Similarity:** the cosine of the topic embeddings (the mean mpnet embedding of each topic's posts). All nine models share one embedding space, so topics can be compared across bins.
- **Edges:** each topic is linked to its 5 nearest topics in the whole corpus, at a similarity of at least 0.65. The cut-off is about the 10th percentile of the 5th neighbour's similarity, and the top 1 % of all pairs.
- **Families:** Louvain communities (seed 42).
- **Layout:** Fruchterman–Reingold on the main connected network, framed on the central 98 % of positions.

**Result.**
- **Links:** 21,964, of which 84 % join topics of different bins.
- **Families:** 77 families of more than one topic, plus 123 isolated topics.
- **Main network:** 6,053 topics (192 outside it).

The largest families are about the same kinds of content in every period:

| Family | Content | Topics |
|---|---|---|
| F1, F3 | transfers and loans | 495 + 469 |
| F2 | league match results and threads | 120 |
| F4 | goals | 312 |
| F5 | managers and star players | 332 |
| F6–F8 | specific leagues and clubs | — |
| F14 | racism, deaths, fines | — |

Most of the large families span all nine bins. Figure D2 shows every period's topics spread over the whole map. **The topics recur across periods: the bins differ in how much there is of each family, not in which families exist.**

**Figures:**
- D1: labelled F1–F8;
- D1b: no labels on the map, larger points (author's request);
- D2: one panel per bin.

**Tables:** families, nodes, edges. The shared palette and theme moved to `code/R/plot-utils.R`; the nine figures of script 11 are byte-identical after the move.

## 4. Finding: the archive's crawl regime changes in July 2023 (engagement figures)

`score` and `num_comments` are the values at the moment the archive (Arctic Shift) retrieved the post (codebook §3). The raw file's `retrieved_on`, set against `created_utc`, gives this crawl lag for the clean posts:

| Year | Median crawl lag | Note |
|---|---|---|
| 2010 | ≈ 2,826 days (7.7 years) | |
| 2012–2014 | 434–715 days | `retrieved_on` missing for part of 2011–2012 |
| 2015–2019 | 25–85 days | |
| 2020–2021 | 62–276 days | missing for part |
| 2022 – June 2023 | 15–135 days (varies by month) | |
| **July 2023 – 2026** | **≈ 0 hours** | 100 % of posts under one day |

The step is visible in the monthly medians. From June to July 2023, the median score falls from 87 to 50 and the median comment count from 19 to 11. The crawl lag stays at about 0 h afterwards, while the medians stay around 30–45.

**It also changes the corpus composition** (checked later the same day, at Alan's question). A post fetched within hours is often archived before moderators remove it, so it carries no removal flag and passes the cleaning filters:

| Months | Raw posts flagged removed/deleted | Raw posts retained in the clean corpus |
|---|---|---|
| Jan – Jun 2023 | 43–56 % | 38–44 % |
| Jul 2023 | 38 % | 59 % |
| Aug 2023 – Jun 2024 | 25–34 % | 64–70 % |

The clean corpus from July 2023 therefore very likely holds posts that moderators later removed. This explains at least part of the jump in monthly volume in mid-2023 (figure A1: about 4,500 → 9,500 posts), part of the lower engagement medians, and a shift in the mix of posts behind the topic and sentiment series. **Decision E8 (Alan Nemirovski, 2026-10-27):** document it in the codebook (limitation 11, §8.5) and revisit if needed.

A `retrieved_on` equal to the creation time is not plausible for scores of this size. The archive probably refreshes scores later without updating `retrieved_on`, but this has not been verified. As decided (E5), the figures do not mark the break; the codebook documents it (E8). The chapter text should mention it: engagement levels after mid-2023 are not directly comparable with earlier years, and comparisons within a period bin are safer than across bins. The project's own 2026 collector (t0 and t48 snapshots, `reddit-soccer-collector/`) could be used to check this later.

## 5. Outlier-tier evidence received today (BA, bins 1 and 5)

Share of posts in each band judged to belong to the topic they were given (`belongs` = 1), from coder BA's samples of 100 posts per band:

| Bin | 35th–50th pct | 50th–65th | 65th–80th | ≥ 80th (reassigned) |
|---|---|---|---|---|
| 1 | 19 % | 19 % | 22 % | 39 % |
| 3 (from session 08) | 41 % | 38 % | 38 % | 51 % |
| 5 | 16 % | 28 % | 32 % | 51 % |

The dual tier (35th–80th percentile) admits mostly posts that do not belong, by this coder's judgement, and the single tier about half. That is why E3 leaves the dual tier out of the tone figures. **The outlier-tier decision itself is still open.** The options, from session 08, are:
- document the precision in the codebook;
- raise `OUTLIER_REASSIGN_QUANTILE`;
- confine topic analyses to `cluster` posts.

A second coder on the same files would give an agreement figure.

## 6. Open items after this session

*Updated after Alan's decisions on the open items (`00_approvals.md` §2b).*

1. **Outlier tiers and the 0.50 gate tolerance:** pending the remaining outlier calibration checks. No second coder. Both must be settled before S6 in place.
2. **Sentiment coding:** done (§2b); the figures are in codebook §8.2 and limitation 10.
2c. **VADER's weak agreement:** decided (E9). The variables are the sentimental valence of posts, reported with the validation; no further validation.
3. **S6 in place** (`06-tag-corpus.R`, then `07-validate-analysis.R`, then `03-data-validation.R`), then codebook v3.0 final.
4. **The political split** (E2): held until at least TP's bin-9 file arrives.
5. **The crawl break:** done (E8, codebook limitation 11).
