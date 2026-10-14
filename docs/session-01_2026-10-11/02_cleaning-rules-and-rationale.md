# Session 01 (2026-10-11) — Cleaning rules, rationale and decisions

Companion to `code/02-data-cleaning.R` (where every rule is implemented in the CONFIG section §1 and
the filter section §4–5) and to Section 3 of the codebook. Decisions marked **[AN]** were taken by
Alan Nemirovski during the session after being presented with the evidence in `01_data-audit-findings.md`.

## 1. Decisions

| # | Decision | Rationale | Who |
|---|---|---|---|
| D1 | Rebuild all cleaned files from `soccer_posts.csv` rather than repairing the quote-corrupted v1 files | v1 doubled every literal quote; raw is correct; repair in place cannot distinguish genuine doubled quotes (0.4 % of titles) from the artefact | [AN] |
| D2 | Implement in R with `readr` (all columns as character) | team workflow is R; readr parses the file correctly; verbatim pass-through preserves content integrity | [AN] |
| D3 | Treat any non-empty `removed_by_category` and placeholder titles as removed/deleted and move them to the side files | ~203k removed link posts had survived v1; they were not part of visible discourse (median 0–3 comments) | [AN] |
| D4 | Move bot-authored structural threads to `bot_threads.csv`; keep human-posted match threads with a `post_type` flag | corpus has no comment text; bot post bodies are boilerplate. Condition set by [AN]: keep only if comments were substantively represented — they are not (see audit §6) | assistant proposed, [AN] approved |
| D5 | Spam accounts require **all three** criteria (≥50 posts, ≥80 % one non-community domain, ≥90 % zero-comment posts); anchor in real literature, thresholds justified empirically | [AN] instruction | [AN] |
| D6 | Filter URL-only titles and merchandise spam; filter stream-piracy at **account** level (accounts that only post piracy) | [AN] instruction | [AN] |
| D7 | Move non-English posts to a side file (English-language corpus) | [AN] instruction | [AN] |
| D8 | Encoding damage: **repair where possible without guessing** (strict byte-level re-decoding, logged and reversible); rows that cannot be repaired, and unintelligible titles, go to a side file | [AN] instruction (revised 2026-10-11 after v2.0 review) | [AN] |
| D9 | Keep the gambling rule (largest promotional category) | proposed in plan; **confirmed by [AN] on 2026-10-11** after review of the rule definition and sample | [AN] |
| D10 | Delete the redundant `yearly datasets/soccer_data.clean.csv` | identical to master minus `year`; "no duplicate versions" requirement | plan approved |
| D11 | Codebook as Markdown source + rendered PDF (same file name) | diff-able, editable by all co-authors | plan approved |

## 2. Rules in order of application (v2.1 counts)

| Stage | `filter_reason` | Definition | Rows |
|---|---|---|---|
| S2 | `S2_selftext_deleted` | `selftext == "[deleted]"` | 234,217 |
| S2 | `S2_removed_by_category_deleted` | `removed_by_category ∈ {deleted, author}` (link posts deleted by the author) | 34,091 |
| S2 | `S2_title_deleted_by_user` | title `[deleted by user]` | 15,125 |
| S3 | `S3_selftext_removed` | `selftext == "[removed]"` | 173,511 |
| S3 | `S3_removed_by_category_moderator` | `removed_by_category == moderator` | 109,757 |
| S3 | `S3_removed_by_category_reddit` | `removed_by_category == reddit` (Reddit spam filter) | 48,733 |
| S3 | `S3_removed_by_category_automod_filtered` | `removed_by_category == automod_filtered` | 42,493 |
| S3 | `S3_removed_by_category_copyright_takedown` | `removed_by_category == copyright_takedown` | 866 |
| S3 | `S3_removed_by_category_content_takedown` | `removed_by_category == content_takedown` | 72 |
| S3 | `S3_title_placeholder` | placeholder title (`[ Removed by moderator ]`, `[ Removed by Reddit ]`, …) | 1 |
| S3 | `S3_selftext_removed_by_reddit` | body begins `[ Removed by Reddit on account of …` | 180 |
| S4 | `S4_bot_author` | author ∈ {MatchThreadder, AutoModerator, 2soccer2bot, soccerbot} | 48,156 |
| S5 | `S5a_spam_account` | account meets all three spam criteria (≥50 posts, ≥80 % one non-community domain, ≥90 % zero-comment) | 16,396 |
| S5 | `S5b_gambling` | gambling/prediction domain tokens or betting-tip title phrases | 16,709 |
| S5 | `S5c_merch_spam` | cheap/replica jersey sellers (`four4two.com` etc.); questions exempt | 926 |
| S5 | `S5d_url_only_title` | title is only a URL | 364 |
| S5 | `S5e_stream_piracy_account` | account whose every post is a "watch … live stream free/online/HD" link | 579 |
| S6 | `S6_replacement_character` | U+FFFD replacement character (information lost upstream) | 5 |
| S6 | `S6_mojibake_unrepairable` | mojibake that no byte-level re-decoding repairs | 7 |
| S6 | `S6_control_characters_unrepairable` | C0/C1 control characters that no byte-level re-decoding repairs | 209 |
| S6 | `S6_title_without_letters_or_digits` | title has no letter or digit (emoji/punctuation only) | 114 |
| S7 | `S7_non_english` | title confidently non-English (cld2 + cld3, evidence thresholds), body not confidently English | 2,790 |

Retained: **1,302,021** of 2,047,322 raw posts (63.6%).

## 3. Rationale and literature anchors per rule

### S1 Empty content
`trimws(title) == "" & trimws(selftext) == ""`. Reddit enforces a title, so 0 rows. Kept so the
pipeline documents the check the team asked for.

### S2 / S3 Deleted and removed
Arctic Shift inherits the Pushshift conventions documented by Baumgartner et al. (2020): `[deleted]`
in `selftext` = author deletion, `[removed]` = moderator/platform removal; `removed_by_category`
records the actor for any post type (link posts have an empty body, so this field is the only marker
for them). Excluding removed/deleted content and bot output before analysis is routine in the Reddit
literature surveyed by Proferes et al. (2021). Engagement evidence from this corpus (audit §3):
automod-filtered posts median 1 comment (never publicly visible); Reddit-spam-filtered median 0;
moderator-removed median 3. Deleted and removed are kept in separate files because the *actor*
differs (user vs platform), which matters for some discourse questions.

Mapping: `deleted`, `author` → deleted file; `moderator`, `reddit`, `automod_filtered`,
`copyright_takedown`, `content_takedown` → removed file. Title placeholders and the
`[ Removed by Reddit on account of …]` body text are handled explicitly (they do not carry a category).

### S4 Bot-authored threads
Authors `MatchThreadder`, `AutoModerator`, `2soccer2bot`, `soccerbot`. Evidence: 48k posts carrying
18 % of all referenced comments, but bodies are templates (score lines, rules text). No comment text in
the corpus → nothing analysable remains. Human-posted pre/post/match threads (~43k) are kept and
flagged `post_type = match_thread` because they are written by community members (team line-ups,
statistics, commentary) and co-authors may want them.

### S5a Spam accounts — three-criteria rule
Signals and sources:

| Criterion | Signal in this corpus | Anchor |
|---|---|---|
| ≥ 80 % of posts to one non-community domain | content-farm accounts post only their own site (e.g. `espnsoccer` → espnsoccer.co, 100 %) | "fraction of tweets with URLs" and URL/domain concentration features in Benevenuto et al. (2010); message-similarity and URL-ratio features in Stringhini et al. (2010) |
| ≥ 90 % of posts with zero comments | legitimate single-outlet accounts (`TheTelegraph`, `lordjems`, `theipaper`) have 0–15 % zero-comment posts; farms ≥ 95 % | interaction/engagement features used to separate "content polluters" from legitimate users in Lee, Eoff & Caverlee (2011) |
| ≥ 50 posts | shares are unstable for small accounts; protects occasional posters | our choice |

The cited works build supervised classifiers on labelled data; we do not have labels, so we use the
same behavioural signals as explicit rules whose outputs are listed (`spam_accounts.csv`) and can be
overridden. Thresholds are empirical: see the distribution table in `01_data-audit-findings.md` §7.
The "community-hosting" domain set (reddit/twitter/youtube/imgur/clip hosts) is excluded from the
concentration criterion because posting 100 % goal clips to streamable.com is normal community behaviour.

### S5b Gambling / betting-tip sites
Domain tokens (Indonesian: prediksi, judi, taruhan, bandar, togel, agen, bola88/303 brand patterns;
generic: casino, poker, sbobet, sportsbook, slot; `bet` as a whole token or in bookmaker compounds so
that *Betis* is not hit; Korean 바카라/카지노/먹튀) or title phrases (prediksi, judi, "betting tips",
"sure win", …). Sample review precision ≈ 85–90 %; false positives are news items on bookmaker
domains (e.g. a poker-tournament result on pokerpro.cc). All hits carry `filter_reason = S5b_gambling`
for review.

### S5c Merchandise spam
Patterns for cheap/wholesale/replica jerseys and the recurring `four4two.com` / `jerseymate` sellers.
Titles with a question mark are exempt (genuine "where can I buy…?" questions) — added after the
first full run showed ~15 % of hits were such questions.

### S5d URL-only titles
Title is nothing but a URL. Predominantly affiliate/betting/stream links; a few genuine link posts are
included (user chose to post a bare URL) — accepted per D6.

### S5e Stream-piracy accounts
Pattern "live stream … free/online/HD" on a non-community domain; applied at account level: every
post of the account must match. 502 accounts, 579 posts, most accounts
single-use throwaways (`wAtCh … LiVe Free SOCCER Streaming Online HD TV`).

### S6 Encoding damage — repair first, then filter
Detection: U+FFFD; cp1252 mojibake (`Ã`/`Â` followed by a non-alphanumeric, `â€`); C0/C1 control
characters except TAB/LF/CR; titles without any letter or digit.

Repair (D8, revised): the only transformation applied is a byte-level re-interpretation — the damaged
field is converted back to the single-byte encoding it was wrongly decoded with (Windows-1252, then
ISO-8859-1, then Windows-1250) and those bytes are decoded as UTF-8. The result is accepted only if
every character was representable, the bytes are strictly valid UTF-8 and no damage marker remains.
This is the same principle as the "mojibake undo" of the `ftfy` library but restricted to the
unambiguous case; no character is ever inferred. Every accepted repair is logged
(`data artefacts/encoding_repairs.csv`: id, field, original, repaired, source encoding) and the
validation script proves reversibility by re-deriving each original from its repaired text (check C6d).

Outcome (v2.1): 126 fields in 126 posts repaired (mostly `â€™` → `’`,
`â€“` → `–`, `ÂŁ`/`Â£` → `£`, `Ã©` → `é`); 335 posts filtered, with reasons
`S6_control_characters_unrepairable` (an apostrophe followed by stray C1 bytes — the leading byte was
already lost upstream, so re-decoding is impossible), `S6_mojibake_unrepairable`,
`S6_replacement_character` (U+FFFD carries no information) and `S6_title_without_letters_or_digits`
(emoji/punctuation-only titles, which are unintelligible rather than damaged).

### S7 Non-English titles
Tools: `cld2` (Google CLD2, confidence-aware) and `cld3` (neural CLD3), R bindings by J. Ooms.
Decision on the title; body must not be confidently English. Evidence thresholds (see codebook §3.1)
were tuned on dry runs: the first version (combined text, no capitalisation guard) mislabelled
goal-clip titles made of proper nouns as Italian/Dutch/Finnish; the second (combined text with guards)
moved mixed posts with English titles and quoted foreign bodies. Final rule precision on a 25-row
sample: 25/25 genuinely non-English titles. Recall is deliberately partial; `lang` lets users go further.

## 4. What is not filtered, and why
* Reposts of the same URL (74,818 posts): genuine community behaviour; often removed by mods already.
* `[deleted]`-author posts with intact text: content is what we analyse; the author field is simply unavailable.
* Surveys / self-promotion by engaged members / "I made a video" posts: keyword rules hit genuine content; left in.
* Human templated threads: flagged, not removed.

## 5. Output format decisions
* One writer (`readr::write_csv`, `na = ""`), all values verbatim: `created_datetime` stays
  `YYYY-MM-DD HH:MM:SS` UTC, booleans stay `True`/`False`, missing stays empty, JSON stays valid.
* New columns appended at the end (`year`, `post_type`, `lang`; side files add `filter_reason`) so
  existing column positions are unchanged.

## 6. Open points for the team
1. ~~Confirm the gambling rule (D9) stays~~ — confirmed.
2. ~~Decide whether S6 rows should be repaired~~ — done in v2.1 (D8 revised).
3. The `post_type` regex list and the bot-author list are editable in the CONFIG section; extend if
   further structural threads (e.g. "Monday Moan") should be treated differently.
