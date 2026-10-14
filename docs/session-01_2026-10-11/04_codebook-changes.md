# Session 01 (2026-10-11) — Codebook changes (v1.0 → v2.0)

Source of truth is now `data/codebook_soccer_posts.md` (Markdown); `data/codebook_soccer_posts.pdf`
is rendered from it with `code/04-render-codebook.R` (commonmark → HTML → headless Chrome).

| Area | v1.0 (2026-08-09) | v2.0 (2026-10-11) |
|---|---|---|
| Scope | raw `soccer_posts.csv` only | whole file family: raw, clean master, yearly files, six side files, audit tables |
| Provenance | source + collection date | collection batches, conversion, pipeline diagram, scripts 01–04, accounting identity |
| Cleaning | not documented | every rule with definition, rationale, literature anchor and row count; what is *not* filtered |
| Variables | 64 raw columns, one-line descriptions | 42 clean-file columns with type, description, example, missingness and caveats; list of 25 dropped columns |
| Derived columns | — | `year`, `post_type`, `lang`, `filter_reason` |
| Format | "UTF-8 CSV", booleans "True"/"False" | full conventions (quoting, multi-line fields, missing = empty, UTC, JSON parsing, recommended readers, `fread` warning) |
| Statistics | record count | per-year table, text availability, engagement, `post_type`/`lang` distributions |
| Limitations | missing-value note | posts-only corpus, title-dominated text, crawl-time snapshot, era-dependent removal metadata, deleted authors, language and encoding residuals, flair drift |
| Methods | — | placeholder section 8 for the team |
| Versioning | — | version history table; references list |

Corrections to v1.0 statements:
* `edited` can also be `True` (2,262 pre-2012 raw posts), not only `False`/timestamp.
* `created_datetime` example format is `YYYY-MM-DD HH:MM:SS` UTC in all files (v1 cleaned master had
  ISO `T…Z`).
* `downs` is uninformative (always 0/empty), `ups` equals `score`.
* `subreddit_subscribers` is empty before ~2018; `removed_by_category` populated only from mid-2019.

## Addendum — v2.1 (completed in session 02, 2026-10-13)
* Section 3.1 S6 rewritten: repair-first policy (strict byte-level re-decoding, logged, reversible), then filter.
* File table: `data artefacts/encoding_repairs.csv` added; `encoding_issue_posts.csv` now 335 rows.
* Counts throughout updated to v2.1 (clean master 1,302,021); validation now 28 checks.
* Version history row 2.1 added; gambling rule recorded as confirmed by the team.
