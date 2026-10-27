# Codebook: r/soccer post corpus (2010–2026)

**Version 3.0 (draft) — 23 October 2026** · r/soccer posts from the Arctic Shift archive, 1 January 2010 – 31 July 2026 · Assembled for the FIFA Communities chapter, McGill University.

This codebook documents every data file in `data/`, how each was produced, the meaning of every variable, and the known limitations of the corpus. It is written for co-authors, reviewers and any researcher who re-uses the data. Version 3.0 adds the **analysis variables** that the analytical pipeline appends to the master and yearly files (Section 4.9) and the methodological use of the data (Section 8), both specified in `docs/session-04_2026-10-22/01_analytical-pipeline-design.md`. **Status:** the data files on disk are still version 2.1 (42 columns) until stage S6 of that pipeline has run; this version is marked *draft* until then.

---

## 1. Overview of the data files

The corpus is a **family of files derived from one raw export**. Use the *clean master* (or its yearly splits) for analysis; the *side files* preserve everything that was filtered out so that any decision can be audited or reversed.

| File | Rows | Columns | Role |
|---|---|---|---|
| `data/soccer_posts.csv` | 2,047,322 | 64 | **Raw export.** Every post retrieved from Arctic Shift, de-duplicated by id. Never modified. |
| `data/soccer_data.clean.csv` | 1,302,021 | 42 | **Clean master.** Posts retained for analysis (63.6% of raw). |
| `data/yearly datasets/soccer.data.cleaned-YYYY.csv` (17 files, 2010–2026) | 1,302,021 in total | 42 | The clean master split by calendar year (UTC). Identical format; convenient for memory-limited work. |
| `data/deleted_posts.csv` | 283,433 | 43 | Side file: posts deleted by their author (stage S2). |
| `data/removed_posts.csv` | 375,613 | 43 | Side file: posts removed by moderators, Reddit's spam filter, AutoModerator or takedowns (S3). |
| `data/bot_threads.csv` | 48,156 | 43 | Side file: template threads posted by bot accounts (S4). |
| `data/promotional_posts.csv` | 34,974 | 43 | Side file: promotional, gambling, merchandise, link-dump and stream-piracy posts (S5). |
| `data/encoding_issue_posts.csv` | 335 | 43 | Side file: posts whose text damage could not be repaired, or whose title is unintelligible (S6). |
| `data/non_english_posts.csv` | 2,790 | 43 | Side file: posts whose title is confidently non-English (S7). |
| `data/data artefacts/filter_summary.csv` | — | — | Count of rows per filter reason (audit table). |
| `data/data artefacts/spam_accounts.csv`, `piracy_accounts.csv` | — | — | Accounts caught by the account-level rules, with the statistics that triggered them. |
| `data/data artefacts/encoding_repairs.csv` | — | — | Every title/body repaired in stage S6: post id, field, original text, repaired text, source encoding (126 fields). |
| `data/data artefacts/POSTS-*.jsonl`, `soccer_posts.csv.gz`, `jsonl_to_csv.py` | — | — | Collection artefacts: the three JSONL batches as downloaded, a compressed copy of the raw CSV, and the conversion script. |
| `data/analysis/post_variables.csv` *(planned, v3.0)* | 1,302,021 | 17 | **Analysis join table.** One row per clean post: the 15 analysis variables of Section 4.9 plus the audit column `topic_id_raw`. Written by stage S5 of the analytical pipeline; the source of the columns appended to the master and yearly files in stage S6. |
| `results/analysis/topic_labels_all_bins.csv` *(planned, v3.0)* | one per topic | — | Two-agent label of every BERTopic topic (name, purpose, emotional tone, description, inclusion cues, debate rounds, agreement) with its `period_bin`, its corpus-wide key `topic_uid`, its sizes and the mean VADER score of its posts (`mean_vader_sentiment`). The unit of analysis since decision D7 (2026-10-25). |
| `results/analysis/topic_crosswalk.csv` *(planned, v3.0)* | one per topic | — | Template for the co-authors' manual cross-bin crosswalk: every topic with a blank `chapter_theme`, filled by hand when the chapter's analysis compares World Cup periods. |

**Version 3.0 (planned).** After stage S6 of the analytical pipeline, the clean master and the yearly files gain the 15 analysis columns of Section 4.9 (57 columns in total); the side files are unchanged (43 columns). The row counts do not change.

**Accounting identity.** raw = clean + deleted + removed + bot + promotional + encoding + non_english (2,047,322 = 1,302,021 + 283,433 + 375,613 + 48,156 + 34,974 + 335 + 2,790). Every post id appears in exactly one file. Verified by `code/03-data-validation.R` (28 checks).

**Pipeline / provenance.**

```
Arctic Shift API
  └─ fetch_posts_range.py ─▶ POSTS-*.jsonl (3 batches)
       └─ jsonl_to_csv.py ─▶ soccer_posts.csv (raw, 64 cols, de-duplicated by id)
            └─ 02-data-cleaning.R ─▶ soccer_data.clean.csv + yearly files
                                     + 6 side files + audit tables
                 └─ 03-data-validation.R ─▶ 28 checks (all pass for v2.1)
```

### 1.1 Clean rows per year

| Year | Raw posts | Clean posts | Retained |
|---|---|---|---|
| 2010 | 17,938 | 13,638 | 76% |
| 2011 | 33,854 | 33,280 | 98% |
| 2012 | 64,395 | 63,649 | 99% |
| 2013 | 79,112 | 78,472 | 99% |
| 2014 | 116,902 | 112,754 | 96% |
| 2015 | 126,399 | 107,461 | 85% |
| 2016 | 137,766 | 85,318 | 62% |
| 2017 | 160,413 | 105,531 | 66% |
| 2018 | 197,865 | 126,743 | 64% |
| 2019 | 185,591 | 118,664 | 64% |
| 2020 | 168,357 | 61,794 | 37% |
| 2021 | 173,373 | 65,286 | 38% |
| 2022 | 155,575 | 64,584 | 42% |
| 2023 | 130,421 | 70,334 | 54% |
| 2024 | 116,843 | 76,473 | 65% |
| 2025 | 108,138 | 74,348 | 69% |
| 2026 | 74,380 | 43,692 | 59% |

The retained share falls from 2020 onwards because Reddit's `removed_by_category` metadata only exists in the archive for posts from mid-2019 on; before that, moderator removals of link posts are not recorded and such posts cannot be identified. The rise from mid-2023 onwards has a different cause: the archive began to fetch posts within hours of posting, often before moderators removed them, so more of them pass the filters (limitation 11). Analyses that compare periods should keep both in mind (Section 7).

---

## 2. Data source and collection

* **Source.** Arctic Shift (https://arctic-shift.photon-reddit.com/), a community-run archive of Reddit that preserves the full post object returned by Reddit's API, including fields not shown in the Reddit interface.
* **Collection.** Three batches fetched on 9 August 2026 with `reddit-soccer-collector/src/fetch_posts_range.py`: 1 Jan 2010 – 31 Mar 2018, 31 Mar 2018 – 9 Jun 2026, 9 Jun 2026 – 31 Jul 2026 (the last batch overlaps the project's own live collector). Overlapping posts were de-duplicated by `id` during conversion to CSV; 2,047,322 unique posts remain.
* **Snapshot semantics.** Each record reflects the state of the post **when Arctic Shift's crawler retrieved it**, not when it was written. `score`, `num_comments`, flairs and removal status are therefore crawl-time values; a post crawled minutes after submission will show low engagement regardless of its later popularity. `retrieved_on` (raw file only) gives the crawl time.
* **Posts only.** The corpus contains submissions (titles, link URLs, self-text bodies). **No comment text is included**; `num_comments` is the only comment-related signal. 97.7 million comments are referenced by the raw posts.
* **Coverage.** Every month from 2010-01 to 2026-07 is present; monthly volumes are continuous across the batch boundaries. Dips reflect real events (e.g. the April–May 2020 COVID suspension).

---

## 3. Cleaning and filtering rules

All cleaning is performed by `code/02-data-cleaning.R` in one pass over the raw file. Rules are applied **in the order below**; a post is assigned to the first rule it matches and written to that rule's side file with the exact `filter_reason` recorded. Counts are for version 2.1.

| Stage | Side file | Rule (what is matched) | Rows |
|---|---|---|---|
| S2 | `deleted_posts.csv` | `selftext == "[deleted]"` (`S2_selftext_deleted`) | 234,217 |
| S2 | `deleted_posts.csv` | `removed_by_category ∈ {deleted, author}` (link posts deleted by the author) (`S2_removed_by_category_deleted`) | 34,091 |
| S2 | `deleted_posts.csv` | title `[deleted by user]` (`S2_title_deleted_by_user`) | 15,125 |
| S3 | `removed_posts.csv` | `selftext == "[removed]"` (`S3_selftext_removed`) | 173,511 |
| S3 | `removed_posts.csv` | `removed_by_category == moderator` (`S3_removed_by_category_moderator`) | 109,757 |
| S3 | `removed_posts.csv` | `removed_by_category == reddit` (Reddit spam filter) (`S3_removed_by_category_reddit`) | 48,733 |
| S3 | `removed_posts.csv` | `removed_by_category == automod_filtered` (`S3_removed_by_category_automod_filtered`) | 42,493 |
| S3 | `removed_posts.csv` | `removed_by_category == copyright_takedown` (`S3_removed_by_category_copyright_takedown`) | 866 |
| S3 | `removed_posts.csv` | `removed_by_category == content_takedown` (`S3_removed_by_category_content_takedown`) | 72 |
| S3 | `removed_posts.csv` | placeholder title (`[ Removed by moderator ]`, `[ Removed by Reddit ]`, …) (`S3_title_placeholder`) | 1 |
| S3 | `removed_posts.csv` | body begins `[ Removed by Reddit on account of …` (`S3_selftext_removed_by_reddit`) | 180 |
| S4 | `bot_threads.csv` | author ∈ {MatchThreadder, AutoModerator, 2soccer2bot, soccerbot} (`S4_bot_author`) | 48,156 |
| S5 | `promotional_posts.csv` | account meets all three spam criteria (≥50 posts, ≥80 % one non-community domain, ≥90 % zero-comment) (`S5a_spam_account`) | 16,396 |
| S5 | `promotional_posts.csv` | gambling/prediction domain tokens or betting-tip title phrases (`S5b_gambling`) | 16,709 |
| S5 | `promotional_posts.csv` | cheap/replica jersey sellers (`four4two.com` etc.); questions exempt (`S5c_merch_spam`) | 926 |
| S5 | `promotional_posts.csv` | title is only a URL (`S5d_url_only_title`) | 364 |
| S5 | `promotional_posts.csv` | account whose every post is a "watch … live stream free/online/HD" link (`S5e_stream_piracy_account`) | 579 |
| S6 | `encoding_issue_posts.csv` | U+FFFD replacement character (information lost upstream) (`S6_replacement_character`) | 5 |
| S6 | `encoding_issue_posts.csv` | mojibake that no byte-level re-decoding repairs (`S6_mojibake_unrepairable`) | 7 |
| S6 | `encoding_issue_posts.csv` | C0/C1 control characters that no byte-level re-decoding repairs (`S6_control_characters_unrepairable`) | 209 |
| S6 | `encoding_issue_posts.csv` | title has no letter or digit (emoji/punctuation only) (`S6_title_without_letters_or_digits`) | 114 |
| S7 | `non_english_posts.csv` | title confidently non-English (cld2 + cld3, evidence thresholds), body not confidently English (`S7_non_english`) | 2,790 |
| — | `soccer_data.clean.csv` | **retained** | **1,302,021** |

### 3.1 Rule definitions and rationale

**S1 Empty content.** `title` and `selftext` both blank after trimming. Reddit requires a title, so no post matches; the rule is kept for completeness.

**S2 Deleted (user-initiated).** `selftext == "[deleted]"`, or `removed_by_category ∈ {deleted, author}`, or title `[deleted by user]`. The Pushshift/Arctic Shift convention is that `[deleted]` marks author deletion and `[removed]` marks moderator/platform removal (Baumgartner et al. 2020). For link posts the body is empty, so `removed_by_category` is the only reliable marker.

**S3 Removed (moderator / platform).** `selftext == "[removed]"`, or `removed_by_category ∈ {moderator, reddit, automod_filtered, copyright_takedown, content_takedown}`, or a placeholder title (`[ Removed by moderator ]`, `[ Removed by Reddit ]`, `[deleted]`, `[image processing failed]`), or a body beginning `[ Removed by Reddit`. Such posts were not part of the visible discourse at crawl time (`automod_filtered` posts were never public; Reddit-spam-filtered posts have a median of 0 comments). Excluding removed content is standard practice in Reddit research (Proferes et al. 2021).

**S4 Bot-authored threads.** Author ∈ {`MatchThreadder`, `AutoModerator`, `2soccer2bot`, `soccerbot`}. These accounts post template threads (match threads with lineup/score blocks, daily discussion, free-talk and merchandise threads). The discourse they host lives in comments, which the corpus does not contain; their own text is boilerplate that would dominate a topic model. Human-posted match/post-match threads are **kept** and labelled `post_type = match_thread`.

**S5 Promotional / non-discourse content.**

* *S5a Spam accounts* — an account is classified as promotional when **all three** hold: (i) ≥ 50 posts in the raw corpus; (ii) ≥ 80 % of its posts link to a single domain outside the community-hosting set (reddit, twitter/x, youtube, imgur, streamable/streamja/streamin/streamff and other clip hosts, instagram, facebook, bsky, wikipedia); (iii) ≥ 90 % of its posts received zero comments. All posts by such accounts are moved. The three behavioural features — link-heavy posting, concentration on one URL/domain, and absence of interaction — are the feature families used in classic spammer-detection work on social platforms (Benevenuto et al. 2010; Stringhini et al. 2010; Lee et al. 2011). Those studies train classifiers; here the same signals are applied as transparent, auditable rules. The cut-offs are ours and were chosen from the empirical separation in this corpus: legitimate single-outlet accounts (e.g. newspaper accounts posting only their own domain) have 0–15 % zero-comment posts, content-farm accounts ≥ 95 %. 99 accounts match; they are listed in `data artefacts/spam_accounts.csv`.
* *S5b Gambling / betting-tip sites* — a non-community domain containing gambling tokens (prediksi, judi, taruhan, bandar, togel, casino, poker, sbobet, bet as a token or in bookmaker compounds, 303/88 gambling-brand patterns, Korean casino terms, …) or a title with betting-tip phrases. Mostly Indonesian "prediksi" (match-prediction) sites. A small number of genuine news items on bookmaker domains are caught (e.g. a poker-tournament result); they can be recovered from the side file by `filter_reason`.
* *S5c Merchandise spam* — "cheap/wholesale/replica … jersey/kit/shirt", `four4two.com`, `jerseymate`, "custom soccer uniforms". Titles containing a question mark are exempt so that genuine "where do I buy…?" questions are kept.
* *S5d URL-only titles* — the title consists solely of a URL (link dumps; mostly betting/affiliate links).
* *S5e Stream-piracy accounts* — accounts **all** of whose posts match the pattern "watch … live stream … free/online/HD" on a non-community domain (502 accounts, most with a single post; listed in `piracy_accounts.csv`). Isolated piracy posts by otherwise normal accounts are kept.

**S6 Encoding-damaged or unintelligible text.** Damage is detected when a title or body contains U+FFFD replacement characters, classic cp1252 mojibake (`Ã©`, `â€™`, `Â£`), or C0/C1 control characters (typically a garbled apostrophe such as `won\x92t`), or when the title has no letter or digit at all (emoji-only). Damaged fields are first **repaired where this can be done without guessing**: the text is converted back to the single-byte encoding it was wrongly decoded with (Windows-1252, ISO-8859-1 or Windows-1250, tried in that order) and the resulting bytes are decoded as UTF-8. A repair is accepted only if every character is representable in the candidate encoding, the bytes form strictly valid UTF-8 and no damage marker remains; the operation is a pure re-interpretation of the same bytes, is reversible, and is logged field by field in `data artefacts/encoding_repairs.csv` (126 fields in 126 posts, e.g. `Lucas Mouraâ€™s` → `Lucas Moura’s`, `ÂŁ13.5million` → `£13.5million`). The validation script re-derives every original from its repaired value. Rows that cannot be repaired this way (for example an apostrophe followed by stray C1 bytes, or a lone U+FFFD) and emoji-only titles are moved to `encoding_issue_posts.csv` with the specific reason. The damage is inherited from Reddit/Arctic Shift, not introduced by the pipeline.

**S7 Non-English titles.** Language identification with Google's Compact Language Detector 2 (`cld2`) and CLD3 (`cld3`) on the **title**. A post is moved only when (a) cld2 is confident, (b) cld3 agrees, (c) the title has enough evidence — ≥ 15 letters with ≥ 8 outside the Latin script, or (for Latin-script languages) ≥ 30 letters, ≥ 5 word tokens and fewer than half of the tokens capitalised — and (d) the body, if present, is not confidently English. Condition (c) keeps goal-clip titles made of proper nouns ("Cagliari 1-[1] Lazio - Lucas Leiva"), which detectors mislabel as Italian/Dutch/Finnish; condition (d) keeps posts with an English title and a quoted foreign-language body. The decision is deliberately conservative: the clean master still contains some non-English posts with short, name-heavy titles, and the `lang` column lets users apply their own threshold.

### 3.2 Content deliberately *not* filtered

* Posts by `[deleted]` authors whose text is intact (author account deleted later): 117,793 in v1 counts. Content integrity is preserved.
* Reposts of the same URL (33,416 URLs submitted 2+ times): normal community behaviour.
* Human-authored templated threads (pre/post/match threads): kept, labelled via `post_type`.
* Surveys, self-promotion of videos/blogs by otherwise engaged community members: keyword rules for these had unacceptable false-positive rates in testing and were not adopted.

### 3.3 Columns dropped from the raw export

The clean and side files keep 39 of the 64 raw columns and add three derived ones. The 25 dropped columns are either constant for a single-subreddit corpus, redundant, crawl-time artefacts or sparsely populated UI flags: `author_fullname`, `author_premium`, `author_is_blocked`, `upvote_ratio`, `created_utc` (duplicates `created_datetime`), `is_original_content`, `thumbnail`, `thumbnail_height`, `thumbnail_width`, `over_18`, `spoiler`, `stickied`, `locked`, `pinned`, `distinguished`, `contest_mode`, `no_follow`, `hide_score`, `archived`, `quarantine`, `subreddit`, `subreddit_id`, `subreddit_type`, `all_awardings`, `retrieved_on`. They remain available in `soccer_posts.csv` and can be joined back by `id`.

---

## 4. Variable definitions (clean master, yearly and side files)

All 42 columns are stored as text; types below are the *intended* types after conversion. "Missing" means an empty cell. Examples are real values.

### 4.1 Identification

| Variable | Type | Description | Example |
|---|---|---|---|
| `id` | string | Reddit post id (base-36). Unique across all files; join key to the raw file. | `88kf9q` |
| `name` | string | Reddit "fullname" (`t3_` + id). | `t3_88kf9q` |
| `permalink` | string | Path of the post on reddit.com. | `/r/soccer/comments/88kf9q/ray_wilkins_…/` |

### 4.2 Content

| Variable | Type | Description | Example |
|---|---|---|---|
| `title` | string | Post title, verbatim (UTF-8, may contain quotes and emoji). Never empty. | `Ray Wilkins: Ex-England captain in critical condition…` |
| `selftext` | string | Body of a self (text) post in Reddit Markdown; empty for link posts and, since 2022, optionally present on link posts. Non-empty for ~10.6% of clean posts. May contain line breaks. | `The votes are in, you decided…` |
| `url` | string | Linked URL for link posts; the post's own Reddit URL for self posts. | `https://www.bbc.co.uk/sport/…` |
| `domain` | string | Domain of `url` (`self.soccer` for self posts). | `twitter.com` |
| `is_self` | boolean (`True`/`False`) | Self (text) post vs link post. | `False` |
| `is_video` | boolean | Post hosts a Reddit-native video (`v.redd.it`). Missing for older posts. | `False` |
| `is_reddit_media_domain` | boolean | Linked content hosted on Reddit's media CDN. Missing for older posts. | `True` |
| `post_hint` | string | Reddit's content-type hint: `link`, `image`, `hosted:video`, `rich:video`, `self`, `gallery`. Missing for ~60 % of posts (older posts, self posts). | `rich:video` |
| `media` | JSON string | Embedded-media metadata (`reddit_video`, oEmbed data). Empty when none. | `{"reddit_video":{…}}` |
| `preview` | JSON string | Preview image URLs and resolutions. Empty when none. | `{"images":[{…}]}` |

### 4.3 Author

| Variable | Type | Description | Example |
|---|---|---|---|
| `author` | string | Username. `[deleted]` when the account was deleted or the post removed after crawl. | `whistledick` |
| `author_flair_text` | string | Author's subreddit flair (usually a club; emoji-code form `:FC_Porto:` in later years). | `Arsenal` |
| `author_flair_css_class` | string | CSS class of the author flair (encodes team/country ids in older years). | `s1 1 team-1 country-eng` |
| `author_flair_background_color`, `author_flair_text_color`, `author_flair_type`, `author_flair_template_id` | string | Styling and template metadata of the author flair; mostly missing before 2018. | `dark`, `richtext` |
| `author_flair_richtext` | JSON string | Rich-text components of the author flair (`[]` when none). | `[{"e": "text", "t": "Arsenal"}]` |

### 4.4 Post flair

| Variable | Type | Description | Example |
|---|---|---|---|
| `link_flair_text` | string | Post category assigned by moderators/author. Labels drift over time (e.g. `Official source`, `Official Source`, `:n_official: Official Source`); normalise before counting. Missing for ~half of posts. | `Media` |
| `link_flair_css_class`, `link_flair_background_color`, `link_flair_text_color`, `link_flair_type`, `link_flair_template_id` | string | Styling and template metadata of the post flair. | `#5b5b5b` |
| `link_flair_richtext` | JSON string | Rich-text components of the post flair (`[]` when none). | `[{"e":"emoji","u":"…"}]` |

### 4.5 Engagement (crawl-time values)

| Variable | Type | Description | Example |
|---|---|---|---|
| `score` | integer | Net votes at crawl time (Reddit "fuzzes" scores). | `10` |
| `ups` | integer | Equals `score` in modern Reddit data. | `10` |
| `downs` | integer | Always 0 or missing (Reddit stopped exposing downvotes). Uninformative. | `0` |
| `num_comments` | integer | Number of comments at crawl time. Only comment-related variable in the corpus. | `142` |
| `num_crossposts` | integer | Times the post was cross-posted *to other subreddits*. Missing for older posts. | `0` |
| `total_awards_received`, `gilded` | integer | Award counts (awards existed 2018–2023). Mostly 0/missing. | `0` |

### 4.6 Time

| Variable | Type | Description | Example |
|---|---|---|---|
| `created_datetime` | datetime (UTC) | Submission time, `YYYY-MM-DD HH:MM:SS`, UTC. Derived from the raw Unix timestamp `created_utc`. | `2018-03-31 18:02:01` |
| `year` | integer (derived) | Calendar year of `created_datetime` (UTC). Used to split the yearly files. | `2018` |
| `edited` | mixed | `False` if never edited; a Unix timestamp (seconds) of the last edit; `True` for some pre-2012 posts where Reddit did not store the time. | `False` / `1522520000` |

### 4.7 Moderation and subreddit

| Variable | Type | Description | Example |
|---|---|---|---|
| `removed_by_category` | string | Removal category recorded by Reddit (`moderator`, `reddit`, `automod_filtered`, `deleted`, `author`, `copyright_takedown`, `content_takedown`). **Always empty in the clean master** (all such posts are in the side files); populated only from mid-2019. | `moderator` |
| `subreddit_subscribers` | integer | r/soccer subscriber count at crawl time. Missing before ~2018. | `903813` |

### 4.8 Derived classification columns (added by the pipeline)

| Variable | Type | Description | Values |
|---|---|---|---|
| `post_type` | string | Structural type of the post from title/flair patterns: `match_thread` (pre-/post-/match threads), `daily_thread` (daily discussion, free-talk, Monday Moan, etc.), `regular` (everything else). Informational only; nothing is filtered on it. | `regular` |
| `lang` | string | ISO-639-1 code of the language detected by cld2 on title + body; `und` when the detector was not confident. Informational; the S7 filter uses a stricter, title-based rule (Section 3.1). | `en` |
| `filter_reason` | string | **Side files only.** The rule that moved the row, e.g. `S3_removed_by_category_moderator`, `S5a_spam_account`, `S7_non_english`. | — |

### 4.9 Analysis variables (appended by the analytical pipeline; version 3.0, planned)

These 15 columns are produced by the analytical pipeline designed in session 04 (`docs/session-04_2026-10-22/01_analytical-pipeline-design.md`, v1.6) and appended, after the 42 columns above, to the clean master and the yearly files in its final stage (S6). They are **model outputs**, not properties of the post: topics come from a BERTopic model fitted separately to each period bin, the topic labels from two Qwen3-1.7B "analyst" agents that must agree on each label (decision D7 of 2026-10-25: the individual topic is the unit of analysis; there is no category layer), and sentiment (the sentimental valence of the title's wording) from the VADER lexicon. Posts that HDBSCAN left unassigned are sorted by their similarity to the nearest topic (decision D9 of 2026-10-25): the most similar join their nearest topic, the less confident ones carry a second topic as well (`topic_assignment = dual`; such a post counts once in each of its two topics), the least similar stay outliers. Side files do not carry them. Provenance of every value is recorded in `data/analysis/post_variables.csv` (plus the audit column `topic_id_raw`) and in the per-bin run folders `results/analysis/bin-N/`.

| Variable | Type | Description | Values |
|---|---|---|---|
| `period_bin` | integer | Analysis period of the post, from the UTC calendar date of `created_datetime`. Bins: 1 = 2010-01-01 – 2010-09-30 (2010 World Cup window); 2 = 2010-10-01 – 2014-01-31; 3 = 2014-02-01 – 2014-09-30 (2014 WC); 4 = 2014-10-01 – 2018-01-31; 5 = 2018-02-01 – 2018-09-30 (2018 WC); 6 = 2018-10-01 – 2022-06-30; 7 = 2022-07-01 – 2023-02-28 (2022 WC); 8 = 2023-03-01 – 2026-01-31; 9 = 2026-02-01 – 2026-07-31 (2026 WC). Contiguous; every post has exactly one bin. | `1`–`9` |
| `period_type` | string | `tournament` for bins 1, 3, 5, 7, 9; `interim` for bins 2, 4, 6, 8. Derivable from `period_bin`; a convenience column. | `tournament` / `interim` |
| `topic_id` | integer | BERTopic topic of the post **within its period bin** (one model per bin; ids are bin-local, so the key is `period_bin` + `topic_id`); for a reassigned or dual post the nearest topic. After the tiered outlier reassignment (D9): `-1` = outlier (no topic); empty = not modelled (fewer than 3 word tokens in title + body). | `-1`, `0`, `1`, … (bin-local); empty |
| `topic_uid` | string | The topic's key across the whole corpus: `period_bin` and `topic_id` joined by a hyphen (`1-26` = topic 26 of bin 1). Empty for outliers and not-modelled posts. | `1-0`, `1-26`, `4-1312`, …; empty |
| `topic_name` | string | The name the two analysts agreed on for the post's topic, after co-author renames and merges (`topic_overrides.csv`); the analysts' original name is kept in `results/analysis/topic_labels_all_bins.csv` (`name_agent`). Empty for outliers and not-modelled posts. | free text, 2–6 words; empty |
| `topic_tone` | string | The topic's emotional tone in the controlled vocabulary (`config.TONE_SET`, 19 terms; `competitive`, `informative`, `analytical` and `concerned` added on 2026-10-27, decision E6), mapped from the analysts' free-text `emotion_tone`; `other` when no term of the vocabulary matched. Empty for outliers and not-modelled posts. | `excited`, `competitive`, `critical`, `informative`, …, `other`; empty |
| `topic_uid_2` | string | The second topic of a **dual** post (`topic_assignment = dual`): the second-nearest topic of an outlier whose similarity to its nearest topic lies between the bin's 35th and 80th percentile. Same form as `topic_uid`, always a different topic; the post counts once in each of the two. Empty for every other post. | `1-37`, …; empty |
| `topic_name_2` | string | The agreed name of the second topic, after co-author renames and merges (a second topic merged into the first disappears, and the post becomes `reassigned`). Empty unless dual. | free text; empty |
| `topic_tone_2` | string | The controlled tone of the second topic. Empty unless dual. | as `topic_tone`; empty |
| `topic_assignment` | string | How `topic_id` was obtained: `cluster` = assigned by HDBSCAN; `reassigned` = outlier attached to its nearest topic alone (similarity at or above the bin's 80th percentile of outlier similarities); `dual` = outlier attached to its nearest topic with the second-nearest topic as a second label (35th to 80th percentile); `outlier` = below the 35th percentile, `topic_id = -1`; `not_modelled`. The percentile values in force are recorded per bin in `results/analysis/bin-N/config.json`. | `cluster` / `reassigned` / `dual` / `outlier` / `not_modelled` |
| `topic_similarity` | float | Cosine similarity between the post's sentence embedding and the embedding of its nearest topic; filled for reassigned, dual and outlier posts (for an outlier: the topic it did not reach). Empty for cluster members and not-modelled posts. 4 decimals. | 0 … 1; empty |
| `topic_similarity_2` | float | The same for the second topic of a dual post. Empty otherwise. 4 decimals. | 0 … 1; empty |
| `post_sentiment` | float | **Sentimental valence of the post**: the VADER compound score of the cleaned **title** (the unit every post has; comparable across bins regardless of body length). It measures the emotional valence of the title's wording, **not an authoritative measure of the author's sentiment** (validation: Section 8.2). 4 decimals. | −1.0000 … 1.0000 |
| `post_sentiment_full` | float | Sentimental valence of title + body: the VADER compound score with the body truncated to 1,000 characters; a second measure for analyses of self posts. Same reading as `post_sentiment`. 4 decimals. | −1.0000 … 1.0000 |
| `post_sentiment_label` | string | Categorised valence of `post_sentiment` with VADER's conventional thresholds: `positive` > 0.05, `negative` < −0.05, otherwise `neutral`. A label of the wording, not of the author's stance (Section 8.2). | `positive` / `neutral` / `negative` |

Audit column in `post_variables.csv` only: `topic_id_raw` (the post's HDBSCAN cluster, re-numbered by cluster size — i.e. the BERTopic topic id before outlier reassignment; HDBSCAN's own labels and the mapping are kept in `results/analysis/bin-N/`). The per-topic labels (`results/analysis/topic_labels_all_bins.csv`) carry, for each `period_bin` × `topic_id` (and `topic_uid`): `name`, `purpose`, `emotion_tone`, `tone_category` (controlled vocabulary), `description`, `inclusion_cues`, `round_iterations` (debate rounds until the two agents agreed), `agreed`, `agreement_mode` (`identical`, `accepted_by_A`, `accepted_by_B`, `stalemate`, `cap`, `no_answer`), `needs_review`, both agents' final answers, sizes (`n_cluster` = HDBSCAN members; `n_first` = posts whose first topic it is; `n_as_second` = dual posts for which it is the second topic; `n_total = n_first + n_as_second`, a dual post counting once in each of its two topics; `share` = `n_total` over the bin's modelled posts, so the shares of a bin may add up to more than 1), `mean_vader_sentiment` with `n_posts_scored` (the mean title compound score of the topic's posts, dual posts included in both topics), keywords and the model/prompt versions.

---

## 5. File format conventions

* **Encoding / layout.** UTF-8, comma-separated, header row, RFC-4180 quoting (fields containing commas, quotes or line breaks are wrapped in double quotes; embedded double quotes are doubled). `selftext` frequently spans multiple lines — use a CSV parser, not line-based tools, to count rows.
* **Missing values** are empty cells. There are no `NA`/`NULL` strings.
* **Booleans** are the strings `True` / `False` (as delivered by Reddit).
* **Timestamps** are UTC.
* **JSON columns** (`author_flair_richtext`, `link_flair_richtext`, `media`, `preview`) hold serialised JSON; parse with `jsonlite::fromJSON()` (R) or `json.loads()` (Python). They are empty or `[]` when no data exists.
* **All columns are text.** Recommended readers:

```r
# R
library(readr)
d <- read_csv("data/soccer_data.clean.csv",
              col_types = cols(.default = col_character()), na = character())
d$num_comments     <- as.integer(d$num_comments)
d$created_datetime <- as.POSIXct(d$created_datetime, tz = "UTC")
```
```python
# Python
import pandas as pd
d = pd.read_csv("data/soccer_data.clean.csv", dtype=str, keep_default_na=False)
```

* **Warning.** Do not load these files with `data.table::fread()` default settings: on this corpus it fails to un-escape doubled quotes and silently corrupts every quoted field (this bug affected version 1 of the cleaned files; see Section 9).
* All files (master, yearly, side files) are written by the same `readr::write_csv()` call and share exactly the same conventions.

---

## 6. Dataset statistics (clean master, v2.1)

| Statistic | Value |
|---|---|
| Posts | 1,302,021 |
| Date range (UTC) | 2010-01-01 02:34:08 – 2026-07-31 23:32:20 |
| Self (text) posts | 12.6% |
| Posts with non-empty `selftext` | 10.6% |
| Median title length (characters) | 62 |
| Posts with zero comments at crawl | 15.2% |
| Posts by `[deleted]` authors (content intact) | 113,928 |
| `post_type`: match_thread / daily_thread / regular | 46,958 / 1,780 / 1,253,283 |
| `lang`: en / und / other | 1,078,542 / 143,057 / 80,422 |

---

## 7. Known limitations for analysis

1. **Posts only.** No comment text. Discourse that takes place in comment threads (the bulk of r/soccer interaction, including inside match threads) is not observable; `num_comments` gives only its volume.
2. **Title-dominated text.** About 88 % of clean posts are link posts whose only text is the title (median 62 characters). Topic models and sentiment measures mostly operate on short, headline-like text; results should be interpreted accordingly.
3. **Crawl-time snapshot.** Scores, comment counts and flairs reflect the moment of archiving. Early-crawled posts under-report engagement. The time between posting and archiving is not constant over the corpus (limitation 11).
4. **Removal metadata is era-dependent.** `removed_by_category` exists only from mid-2019. Before that, moderator-removed *link* posts cannot be identified and remain in the clean master; from 2019 on they are removed. Year-to-year comparisons of volume or of removed shares must account for this, and for the change of July 2023 (limitation 11).
5. **Deleted authors.** 117,793 clean posts carry `author = [deleted]`; author-level analyses lose these posts.
6. **Language.** The S7 filter is conservative (Section 3.1); short, name-heavy non-English titles remain. Use `lang` for stricter subsetting.
7. **Encoding.** 126 posts had mangled characters repaired by byte-level re-decoding (logged in `encoding_repairs.csv`); 335 posts that could not be repaired without guessing, or whose title is emoji-only, are in `encoding_issue_posts.csv`.
8. **Flair drift.** Flair labels and their spelling change over the 16 years; normalise before using them as categories.
9. **Spam rules are rule-based.** They are auditable and reversible (every row carries `filter_reason`) but neither exhaustive nor perfect; small numbers of false positives exist in `promotional_posts.csv` (e.g. genuine news on bookmaker domains) and some spam remains in the master.
10. **Analysis variables are model outputs (v3.0).** `topic_id`, the topic labels (`topic_name`, `topic_tone`) and the sentiment scores (Section 4.9) are produced by a topic model, two small language-model agents and a lexicon, respectively. Topic ids are only meaningful within a period bin (nine separate models; `topic_uid` makes them unique across the corpus); comparisons across bins need the manual topic crosswalk. The number of topics follows the volume of a bin (50–75 topics per 10,000 posts, decision D8), so the interim bins hold many more topics than the tournament bins. The outlier tiers (decision D9) are cut at percentiles of each bin's similarity distribution, a rank rather than a quality level; the co-authors' judgement samples measure how often the admitted posts belong, and a dual post's two labels record ambiguity, not two confirmed memberships. VADER scores headline-length text with a general lexicon and misreads sarcasm and football vocabulary; a validation sample is part of the pipeline, coded by a single author, so the human reference has no inter-coder reliability figure; VADER agrees with it on 56 % of titles (κ = 0.24) and assigns sentiment to many titles the reader judges neutral, so the sentiment variables are read as the sentimental valence of a post's wording, used in aggregate, not as the author's sentiment (Section 8.2). Posts with fewer than three word tokens have no topic.
11. **The archive's crawl regime changes in July 2023.** Until June 2023 Arctic Shift fetched posts weeks to years after they were posted (median delay ≈ 7.7 years for 2010, 1–2 years for 2012–2014, 1–3 months for 2015–2019, 2 weeks to 9 months in 2020 – June 2023; `retrieved_on` minus `created_utc` in the raw file). From July 2023 it fetched them within hours. This most likely followed Reddit's API changes of mid-2023 (not verified). It has two consequences.
    - **Engagement.** `score` and `num_comments` are measured differently on either side of the change. The monthly median score falls from 87 (June 2023) to 50 (July 2023), and the median comment count from 19 to 11; both stay at the lower level afterwards. Scores of this size are too high for posts only minutes old, so the archive probably refreshes engagement values after the first fetch at a delay it does not record.
    - **Corpus composition.** A post fetched within hours is often archived before moderators remove it, so it carries no removal flag and passes the cleaning filters (Section 3). Among raw posts, the share flagged as removed or deleted drops from 43–56 % (January–June 2023) to 25–34 % (August 2023 – June 2024). The share retained in the clean corpus rises from 38–44 % to 64–70 %, and monthly clean volume roughly doubles in mid-2023. From July 2023 the clean corpus therefore very likely contains posts that moderators later removed. This affects volume, engagement and, through the mix of posts, topic shares and sentiment.

    The change falls inside period bin 8 (March 2023 – January 2026; four months before it). Bin 9 (the 2026 World Cup) lies entirely after it, and bins 1–7 entirely before it. Comparisons within a bin are unaffected by the change itself. Comparisons across it (for example the 2022 and 2026 World Cups, or any trend through mid-2023) are not like for like (Section 8.5). The project's own collector (`reddit-soccer-collector/`, posts snapshotted at posting and 48 hours later, June–July 2026) could quantify both effects for bin 9; this has not been done (decision of Alan Nemirovski, 2026-10-27: document now, revisit if needed).

---

## 8. Methodological use of the data

Full design, settings with rationale, prompts and flagged limitations: `docs/session-04_2026-10-22/01_analytical-pipeline-design.md` (session 04, 2026-10-22; decisions by the author recorded there). Summary:

**8.1 Corpus construction for topic modelling.** Unit of analysis: one post = one document. Text: the cleaned `title`, a newline, and the cleaned `selftext` truncated to 1,000 characters (cleaning removes Markdown link syntax, URLs, Reddit sprite codes, HTML entities and extra whitespace; it never changes words). Posts with fewer than three word tokens are not modelled. All retained posts are used, including human match/daily threads (`post_type`) and posts whose `lang` is not `en`. The corpus is divided into nine period bins (Section 4.9, `period_bin`) and a separate BERTopic model is fitted per bin: `all-mpnet-base-v2` sentence embeddings → UMAP (15 neighbours, 5 dimensions, cosine, seed 42) → HDBSCAN (`min_cluster_size` from 25 upward, adjusted by a guard to 50–75 topics per 10,000 modelled posts, decision D8; `min_samples` 10) → c-TF-IDF keywords (1–2-grams, English and scaffolding stop-words, KeyBERT-inspired re-ranking with maximal marginal relevance). Posts left unassigned by HDBSCAN are sorted by their embedding similarity to the nearest topic against the bin's own distribution (decision D9): at or above the 80th percentile they join the nearest topic, between the 35th and the 80th they join it and carry the second-nearest topic as a second label, below the 35th they stay outliers (`topic_assignment`). Each topic is then labelled by two Qwen3-1.7B analyst agents that must agree (multi-turn debate, explicit acceptance, at most five rounds; unresolved topics are flagged for the co-authors); the individual topic is the unit of analysis (decision D7: no category layer); co-author renames and merges are recorded in an editable table and applied by the pipeline.

**8.2 Sentiment analysis.** The post-level sentiment variables measure the **sentimental valence of posts**, the emotional weight of their wording as scored by a lexicon. They are not an authoritative measure of the author's sentiment (decision of Alan Nemirovski, 2026-10-27, after the validation below). The chapter uses them mainly in aggregate: mean valence by topic (`mean_vader_sentiment` in the topic table), by period and by theme, read together with the topics' analyst-assigned tones (`topic_tone`). The topic-level measures are not validated separately (same decision). VADER (Hutto & Gilbert 2014) compound scores: `post_sentiment` on the title (comparable across bins and post types), `post_sentiment_full` on title + body, `post_sentiment_label` at the conventional ±0.05 thresholds. A stratified sample of 337 posts (period bin × VADER label × `is_self`, 54 strata) is hand-coded blind (no scores shown, random order) by one author (Alan Nemirovski) as the human reference, on the title. Against it the validation reports VADER's percent agreement and Cohen's κ (bootstrap 95 % interval), the confusion matrix, precision and recall per class, and agreement by period bin and post type (`code/10-sentiment-validation.R`). With a single coder the reliability of the human reference itself is not measured (decision of 2026-10-27).

**Result (2026-10-27; coder: Alan Nemirovski; `results/descriptives/sentiment_validation_*.csv`).** The human coding labels 258 of the 337 titles neutral (77 %), 43 positive and 36 negative; VADER labels 159 neutral, 114 positive and 64 negative. Agreement is **56.4 %**, Cohen's **κ = 0.24** (bootstrap 95 % interval 0.17–0.31). By class (VADER against the human reference): neutral precision 0.91, recall 0.56; positive precision 0.25, recall 0.67; negative precision 0.25, recall 0.44 (macro-F1 0.46). The main error is sentiment where the reader sees none: 113 of the 258 neutral titles receive a positive (72) or negative (41) score, mostly news reported in emotive vocabulary ("unbeaten", "champions", "cheer"; "penalty", "allegations", "loser"); 13 negative titles are scored positive (for example a title about a player injured by a *celebrating* fan). Agreement varies between 47 % and 75 % across the bins (12–77 posts each) without a trend, and is the same for link and self posts (56 % and 57 %). The figures are those of the sample, whose VADER-label mix (47 / 34 / 19 % neutral / positive / negative) is close to the corpus's (50 / 34 / 16 %). VADER therefore measures the valence of a title's words rather than the stance of its author, which is how the variables are defined and used (above). For aggregate use, the errors matter less than they would post by post: what matters is whether they are stable across the groups being compared, and the agreement by bin and by post type shows no systematic difference. **Threshold sensitivity** (`sentiment_validation_thresholds.csv`). The neutral band was widened symmetrically from ±0.05 to ±0.75. Agreement rises with the band (to 67 % at ±0.35, 73 % at ±0.50) only because more titles become neutral, the human coding's majority class. Cohen's κ, which corrects for that, stays at 0.24–0.28 up to ±0.45, inside the interval of the conventional cut-off, and falls beyond it (0.09–0.12 at ±0.60–0.75, where 87–95 % of the corpus would be neutral). No cut-off brings VADER meaningfully closer to the human reading. The disagreement lies in what is measured, the valence of the wording rather than the author's stance, not in where the line is drawn. The conventional ±0.05 encoding is therefore kept. Title-only posts (88 %) are scored on their title; the lexicon's known weaknesses on sports vocabulary and sarcasm are documented in the design (F10).

**8.3 Discourse analysis.** The coding frame is the set of labelled topics of each period bin (`results/analysis/topic_labels_all_bins.csv`: the agreed name, purpose, tone, description and cues of every topic, with its sizes and the mean VADER score of its posts); cross-bin comparison of World Cup periods uses a manual crosswalk of topics to chapter-level themes (`topic_crosswalk.csv`). `post_type`, `lang` and flair are covariates for checking topics (share of match threads, share of English titles per topic), not inputs to the model. Time windows are the period bins; topic prevalence is reported weekly in tournament bins and monthly in interim bins.

**8.4 Reproducibility.** Scripts `code/05-analysis-pipeline.py` (+ `code/topicmodel/`), `code/05b-analysis-pipeline-colab.ipynb`, `code/06-tag-corpus.R`, `code/07-validate-analysis.R` with the shared R helpers `code/R/pipeline-utils.R` (implemented in session 05, consolidated into one pipeline and renumbered in session 07, 2026-10-25; the earlier names are `06-…`, `06b-…`, `07-…`, `08-…` in the session documents of October 2026). Pinned package versions (`code/requirements-topicmodel.txt`) and model revisions; seed 42; greedy decoding; every run folder holds `config.json` (settings, prompts, backend, versions), the intermediate arrays (UMAP output, raw HDBSCAN labels) and the full agent transcripts, so every downstream output can be replayed exactly.

**8.5 Comparisons over time.** The corpus is not a constant-condition measurement across 2010–2026 (limitations 3, 4 and 11). Two changes in how the data were produced shift the composition of the clean corpus: removal metadata exists only from mid-2019, and the archive fetched posts within hours from July 2023. The second change also shifts the meaning of `score` and `num_comments`. Comparisons within a period bin (for example between topics, themes or political and non-political posts during one World Cup) are the most robust. Comparisons across bins should be read as indicative, and those that cross July 2023 (bin 9 against bins 1–7; trends through mid-2023; the two parts of bin 8) should state the change. Engagement levels before and after July 2023 are not directly comparable. Subscriber growth (1.1 million in 2018 to 8.6 million in 2026, `subreddit_subscribers`) is a further reason not to compare raw engagement levels across the 16 years.

---

## 9. Version history

| Version | Date | Change |
|---|---|---|
| 3.0 (draft) | 2026-10-23, revised 2026-10-25 | Analysis variables specified (Section 4.9: `period_bin`, `period_type`, `topic_id`, `topic_uid`, `topic_name`, `topic_tone`, `topic_assignment`, `post_sentiment`, `post_sentiment_full`, `post_sentiment_label`), methodological use documented (Section 8) and limitation 10 added, from the session-04 pipeline design; on 2026-10-25 the category layer was removed and the topic's key, label and tone were put on every post (decisions D7, D8), and the same night the outlier handling became tiered (decision D9: `topic_uid_2`, `topic_name_2`, `topic_tone_2`, `topic_similarity`, `topic_similarity_2` added; 15 analysis columns); on 2026-10-27 four tone categories were added to the controlled vocabulary (decision E6; `topic_tone`, `topic_tone_2`; join table labels version `f2bc460bf46d`), and the archive's change of crawl regime in July 2023 was documented as limitation 11 and in Section 8.5. **The data files remain v2.1 until stage S6 appends the 15 columns to the master and yearly files**; this row is then finalised with the run date and labels version. The pipeline code was implemented and dry-run on bin 1 in session 05 (2026-10-23, `docs/session-05_2026-10-23/`). |
| 1.0 | 2026-08-09 | Raw export (`soccer_posts.csv`, 2,047,322 posts, 64 columns) documented. First cleaned files produced with `data.table::fread()`: 25 columns dropped, posts with `selftext` `[removed]`/`[deleted]` split off (1,591,577 retained). |
| 2.1 | 2026-10-13 | Gambling rule confirmed by the team. Stage S6 now **repairs** encoding damage by strict byte-level re-decoding before filtering (126 posts repaired, 335 filtered); repairs logged in `encoding_repairs.csv` and verified by two new validation checks (28 in total). Clean master: 1,302,021 posts. |
| 2.0 | 2026-10-11 | **Full rebuild from the raw export.** (1) Fixed a quote-doubling corruption introduced by `fread()` in v1 (every literal `"` had become `""`; JSON columns unparseable). (2) Removal detection extended to link posts via `removed_by_category` and placeholder titles. (3) New side files for bot threads, promotional content, encoding issues and non-English titles. (4) Derived columns `post_type` and `lang` added. (5) One writer and one format for all files; redundant duplicate master in the yearly folder removed. (6) Validation script added (26 checks). Details: `docs/session-01_2026-10-11/`. |

---

## 10. References

* Baumgartner, J., Zannettou, S., Keegan, B., Squire, M., & Blackburn, J. (2020). The Pushshift Reddit Dataset. *Proceedings of the International AAAI Conference on Web and Social Media (ICWSM)*, 14, 830–839.
* Benevenuto, F., Magno, G., Rodrigues, T., & Almeida, V. (2010). Detecting spammers on Twitter. *Collaboration, Electronic Messaging, Anti-Abuse and Spam Conference (CEAS)*.
* Lee, K., Eoff, B. D., & Caverlee, J. (2011). Seven months with the devils: A long-term study of content polluters on Twitter. *Proceedings of ICWSM*, 5, 185–192.
* Proferes, N., Jones, N., Gilbert, S., Fiesler, C., & Zimmer, M. (2021). Studying Reddit: A systematic overview of disciplines, approaches, methods, and ethics. *Social Media + Society*, 7(2).
* Stringhini, G., Kruegel, C., & Vigna, G. (2010). Detecting spammers on social networks. *Proceedings of the Annual Computer Security Applications Conference (ACSAC)*, 1–9.
* Arctic Shift Reddit archive: https://arctic-shift.photon-reddit.com/ (data source; cite when using this corpus).
* Ooms, J. `cld2` and `cld3` R packages (bindings to Google's Compact Language Detector 2 and 3), CRAN.
* Campello, R. J. G. B., Moulavi, D., & Sander, J. (2013). Density-based clustering based on hierarchical density estimates. *PAKDD 2013*, LNCS 7819, 160–172. (HDBSCAN)
* Grootendorst, M. (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure. arXiv:2203.05794.
* Hutto, C. J., & Gilbert, E. (2014). VADER: A parsimonious rule-based model for sentiment analysis of social media text. *Proceedings of ICWSM*, 8(1), 216–225.
* McInnes, L., Healy, J., & Melville, J. (2018). UMAP: Uniform Manifold Approximation and Projection for dimension reduction. arXiv:1802.03426.
* Qwen Team (2025). Qwen3 Technical Report. arXiv:2505.09388.
* Reimers, N., & Gurevych, I. (2019). Sentence-BERT: Sentence embeddings using Siamese BERT-networks. *Proceedings of EMNLP-IJCNLP 2019*, 3982–3992.

**Citation.** When using this corpus, cite the Arctic Shift project as the data source and this codebook (version 3.0, McGill University, 2026; version 2.1 for the files without analysis variables).
