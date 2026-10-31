# Session 14 (2026-10-31): the chapter figures from the co-authors' tags — decisions V1–V14

*Participants: Alan Nemirovski (the request, the decisions), Claude (exploration, plan, code, figures, documents). Planned in plan
mode on 2026-10-31, after session 13 had closed; Alan answered three rounds of questions and two follow-ups, and approved the plan
before any file in the repository was changed. Co-authors TP and BA coded the topics (TP: bins 1, 3, 5, 7, 9; BA: bins 2, 4, 6, 8,
the last two uploaded on 2026-10-31).*

## 1. The request

Alan's brief (2026-10-31): map out and implement the code for a fixed set of descriptive visualisations and tables, to replace the
existing figures in `results/descriptives/`, ready to run once the topic labels and tags are mapped onto the topics:

- 1a post volume by month; 1b the average number of comments per post by month; 1c the raw number of comments by month;
- 2a the share of posts tagged political overall (table); 2b political vs non-political posts over time; 2c the share of political
  posts by analytical category — domestic politics, international relations, FIFA governance, identity & nationalism — overall
  (table); 2d the political-categories half of figure E1; 2e figure E1b; 2f the proportion of comments under political posts
  relative to comments under non-political posts;
- 3a a continuous version of E1b across the whole corpus; 3b figure E5; 3c mean sentiment by category over time; 3d, for the
  political posts, the share of content against the share of users (is a small number of users driving the political posting?);
- for 2a–2e and 3a–3d, the main version as stated and, in a subfolder `alternative visualizations`, a version with raw or
  proportional counts, whichever the main version does not show;
- aggregation across topics: a topic tagged "Under Review" (REVIEW) without one of the four categories is excluded; the QUALTAG
  coding of the odd bins must be consistent — a category tag means political, no category tag means non-political unless REVIEW;
- consistent formatting; titles only, no subtitles or captions; ask rather than assume.
- Added during the session: `qt_club`, `qt_intl_non_wc`, `qt_player` have no bearing on the aggregates (referential tags only); all
  the coded bins are on GitHub — pull and merge them and build on them.

## 2. What the exploration established before the questions

- Script 14 (E1–E7) read TP's odd bins by their `qt_political = "yes"` and counted cluster + reassigned posts only (Q7, E3); its
  file glob now matched two bin-5 files and would have stopped. Script 11 (A–C) used medians (E4). The diagnostics D, F, G1 and
  the sentiment validation tables were written into the same folder by scripts 12, 15, 16 and 10.
- BA's files have no `QUALTAG` text, only typed `qt_` columns, mostly blank; in bins 4, 6 and 8 many topics carry `qt_political = 1`
  without any category tag (15, 37, 27). The new files arrived under another name (`topics_labelled_bin6_BA.csv`, `…bin8…`).
- `[deleted]` authors are about 30 % of the posts the archive crawled late (2014: 33,168 of 112,754) and none from 2024.
- `data/` is gitignored: the small topic-tag table belongs under `results/analysis/`; the per-post weights under `data/analysis/`.

## 3. Decisions (Alan Nemirovski, 2026-10-31)

| # | Question | Decision | Why / note |
|---|---|---|---|
| **V1** | Where the figures take the topic tags from, given that bins 6 and 8 had no coding when the question was asked | **The period-bin coding, all nine bins** (BA's even bins included; the two missing files were expected and arrived during the session). The scripts run with a bin missing (its posts `uncoded`, blank months) and unchanged once present | Declined: the reference-model transfer (W2) as the source; the odd bins only |
| **V2** | Where the consistency rule is applied | **A new topic-tag table** (`code/17-topic-tags.R` → `results/analysis/topic_tags.csv`), the co-authors' files and their `qt_` columns untouched; a change list shows every topic whose status differs from the file | Declined: rewriting script 13's rules (Q2–Q4) and the `qt_` columns |
| **V3** | The rule | **Any of the four category tags → political (also beside `non-political` or REVIEW); no category tag → non-political unless REVIEW → excluded.** BA: a category = 1 → political; **`qt_political = 1` without a category → political, uncategorised** (in the political share, in no category); `qt_review = 1` without a category → excluded | The uncategorised option was chosen over "non-political as the rule says" and "excluded until BA categorises them" (79 topics in bins 4, 6, 8 — 90 with bin 2 at the end) |
| **V4** | The post population when aggregating across topics | **The M1 weights** (session 13): cluster / reassigned 1, dual 0.5 in each of its two topics, outlier 0; denominators all posts | Declined: cluster + reassigned only (Q7, E3) |
| **V5** | What non-political means | **Everything not political** (1 − the political weight: outliers, excluded, uncoded included); 2b the exact complement; 2f, 3b, 3c use the same group | Declined: the posts of topics tagged non-political only (E5's reference group) |
| **V6** | 1b's statistic | **The mean** (median and quartiles in the table) | E4 (2026-10-27) had chosen the median for the engagement series; both lines declined |
| **V7** | 2f's form | **A monthly share series** (comments under political posts / all comments) with the overall and per-bin shares in a table | Declined: a ratio series; a table only |
| **V8** | 3a's resolution | **Monthly** | Declined: weekly with a 4-week window |
| **V9** | BA's blank cells | **Blank = non-political** | Declined: blank = uncoded |
| **V10** | What "replace the existing figures" covers | **Retire everything**: every old file in `results/descriptives/` removed, scripts 11 and 14 deleted (git keeps them); the diagnostics (D, F, G1, sentiment validation) move to `results/diagnostics/` and scripts 10, 12, 15, 16 write there | Declined: retiring A–C and E only; keeping the old set beside the new |
| **V11** | 3d's form and the `[deleted]` authors | **One concentration curve for the whole corpus**, `[deleted]` authors excluded from the curve and reported; the table gives the top-1 / 5 / 10 % shares and the Gini index | Declined: one curve per bin; top-user bars |
| **V12** | Alternatives for 3b–3d, where raw vs proportional does not apply | **The post counts behind them** | Declined: none; the share of positive / negative titles |
| **V13** | The two bin-5 review files | **`BIN5_topics_labelled_TP_FINAL.csv` stands**; `BIN5_topics_labelled_TP.csv` removed with `git rm` | TP's revised coding of 2026-10-30 (27 cells changed) |
| **V14** | The qualifier tags | **`qt_club`, `qt_intl_non_wc`, `qt_player` have no bearing on any aggregate** (referential, for consultation); applied to `qt_wc` and `qt_humour` as well, the qualifiers of the same kind | Alan's instruction during the session |
| kept | Q8 (a topic with several categories counts fully in each), E5 (no annotation of the July-2023 crawl break), titles only (2026-10-28) | | |

## 4. Defaults stated with the plan (approved with it)

- Category names as in the request (Domestic politics, International relations, FIFA governance, Identity & nationalism); each
  keeps the colour it had in the E figures. Political = `INK2`, non-political = `MUTED`. Monthly series with the five World Cup
  bins shaded; ids = the chapter numbering (`fig_1a_…`, `tab_2a_…`); the alternatives under the same id with a `_counts` suffix.
- Weighted mean and se for 3b/3c: m = Σ w x / Σ w, se = sqrt(Σ w (x − m)² / Σ w) / sqrt(Σ w). 3a's counts alternative also serves
  as 3c's (the same numbers); 3b's alternative shows the groups on a log scale so that the non-political reference fits.
- The per-post weights are written to `data/analysis/post_political_weights.csv` (untracked, like the join table).
- BA's two new files renamed to the folder convention (`BIN6_…`, `BIN8_…`) at the merge, so that one glob serves scripts 13, 16 and 17.
- Script 13 skips files without `QUALTAG` instead of stopping (BA's), and the scheme + `parse_tags()` moved to `code/R/qualtag-utils.R`,
  shared with 17; `save_fig()` / `save_tab()` take an optional `dir`.
- A new session folder (this one) rather than an addition to session 13's.

## 5. What was changed and run (2026-10-31, after the approval)

- **Merge and housekeeping:** `git pull --no-rebase origin main` (fast-forward `690fdc8a → 26ba54c5`: BA's bins 6 and 8); the
  two files renamed; `BIN5_topics_labelled_TP.csv` removed (V13); `Rscript code/13-recode-qualtag.R` → bin 5 FINAL recoded
  (50 yes / 361 no / 21 review / 16 empty), BA's four files skipped, `QUALTAG_to_settle.csv` 44 topics; all checks PASS.
- **`code/R/qualtag-utils.R`** (new), **`code/17-topic-tags.R`** (new), **`code/18-chapter-figures.R`** (new),
  `code/R/plot-utils.R` (`dir`), scripts 10 / 12 / 15 / 16 (defaults → `results/diagnostics/`); `code/11-descriptives.R` and
  `code/14-worldcup-figures.R` deleted.
- **`Rscript code/17-topic-tags.R`:** 6,245 topics, 9 bins; **539 political** (449 categorised, 90 uncategorised — all BA),
  5,619 non-political, 87 excluded; per category: domestic 53, IR 97, FIFA 160, identity 210 topics; 196 topics on the change
  list (TP: the 41 formerly unsettled, 10 REVIEW-with-category → political, 2 conflicts → political, 1 blank → non-political;
  BA: 11 category-without-political → political, 90 political-without-category, 41 review-without-category → excluded). Checks
  PASS (coverage of every bin's labelled topics both ways; no political topic without a category unless uncategorised).
- **`Rscript code/18-chapter-figures.R`** (45 s): every check PASS; 13 main outputs and 9 alternatives written; political weight
  **72,845 of 1,302,021 posts (5.6 %)**; excluded 0.9 %; uncategorised 0.9 %; per bin 21.5 / 5.4 / 15.9 / 3.7 / 7.1 / 4.4 / 6.1 /
  3.8 / 11.5 %. Categories overall, share of political posts: identity 42 %, FIFA 33 %, IR 15 %, domestic 10 %. Comments under
  political posts 7.3 % of all comments (21 % in bins 1 and 9). 3d: 32,031 named users; the top 1 % write 22 %, the top 10 %
  50 % of the political posts; Gini 0.535; `[deleted]` authors hold 13 % of the political weight.
- **Verification:** the reconciliation with the retired `tab_E1_political_posts.csv` (git) — political share per odd bin
  2010 17.9 → 21.5 %, 2014 11.4 → 15.9, 2018 6.6 → 7.1, 2022 2.4 → 6.1, 2026 7.4 → 11.5 (M1 weights + rule V3); the
  duplicate-bin path stops naming bin 5; the missing-bin path (bin 8 withheld on a scratch copy) runs end to end: a warning,
  35 blank months (Mar 2023 – Jan 2026), the 2a row reporting 86 % uncoded; a visual pass of the main figures (titles only,
  constant category colours, bands and knockout shading).
- **Documentation:** `results/descriptives/README.md` rewritten, `results/diagnostics/README.md` new, `results/README.md`,
  `manual review/README.md`, `results/analysis/reference/README.md`, codebook (two paths; PDF re-rendered), design v2.1 (one path),
  `memory/workflows.md` (§8, §9, §10, §11, new §12), `docs/README.md`, `memory/2026-10-31_session-14.md`.
- Committed and pushed as `9fb9ff51` at Alan's request at the close of the session.
- **Addendum (2026-10-31, after the commit; Alan's request):** figure 3d rebuilt with the non-political posts as the reference curve on
  the same panel, the equality line clipped to the unit square and the Gini of each curve in the corner (an annotation, to keep the
  titles-only rule), in three versions — the whole corpus (`fig_3d_political_user_concentration`), the five tournament bins pooled
  (`…_tournaments_pooled`) and one panel per tournament bin (`…_by_tournament`); "tournament windows" read as the tournament period
  bins, as 2d and 3b. Non-political posts are the more concentrated (Gini 0.80 against 0.54 in the whole corpus; 0.72 against 0.46
  pooled over the tournament bins). Committed separately.
- **Addendum (2026-10-31, Alan's requests):** figure 2b's non-political line dashed. **Figure 2g** added: 2e's layout for the comments
  under each category's posts through each World Cup, both over the centred 7-day window — first as a share of the comments under
  political posts (Alan's choice), then reframed the same day at his request as a share of all comments of the day (the political
  denominator stays in the table as `share_of_political_7d`); the raw comment counts as the counterpart in the alternatives.
- **Addendum (2026-10-31, Alan's requests):** **figure 3b** added — 2g's measure across the whole corpus (the comments under each
  category's posts as a share of the month's comments, four lines, counts counterpart) — and the later figures **renumbered: 3b → 3c
  (VADER by category and World Cup), 3c → 3d (VADER by category per month), 3d → 3e (user concentration)**; the files moved with
  `git mv`. The tournament-bins-pooled version of the concentration curve was removed (the whole-corpus and per-World-Cup versions
  stay). The chapter PDFs moved to `pdfs/` and the tables to `tabs/` in both folders. In the sections above, 3b–3d name the
  figures as planned; the README carries the current ids.

## 6. Open after this session

1. **Commit** the session's changes (the merge is in; `git status` lists them).
2. The chapter text around the figures: the reading notes of `results/descriptives/README.md` (the crawl break, the noisy early
   monthly category lines, the `[deleted]` share in 3d, the uncategorised BA topics).
3. From session 13: the M1 implementation in the pipeline (`recount`, `collect`, script 16) with its own go-ahead — the chapter
   figures already apply the M1 weights (V4); the topic tables still carry the K1 counts.
4. From session 12: the Colab reference run and hand-back, `Rscript code/16-reference-map.R` (figure G1 now in
   `results/diagnostics/`), the QUALTAG transfer rule (W2); S6 + codebook v3.0 final (K5); TP's 44 to-settle topics (the chapter
   figures decide them by rule V3; TP may still settle them in the files); the inter-coder check (K2).
