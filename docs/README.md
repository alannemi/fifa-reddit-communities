# docs/

Working documentation for the r/soccer post-corpus project, organised by
working session.  Each session folder contains, in reading order:

1. `01_…` what was found / audited,
2. `02_…` what was decided and why (rules, thresholds, literature),
3. `03_…` what was run and what came out (commands, logs, counts),
4. `04_…` what changed in shared deliverables (codebook, scripts).

| Session | Date | Focus |
|---|---|---|
| [session-01_2026-10-11](session-01_2026-10-11/) | 2026-10-11 | Data audit, full rebuild of the cleaned corpus from raw, filter rules, codebook v2.0; v2.1 prepared (gambling rule confirmed, encoding repair coded) but run paused |
| [session-02_2026-10-13](session-02_2026-10-13/) | 2026-10-13 | Completion of v2.1 (encoding repair, 28 checks, codebook v2.1); BERTopic pipeline built; 2010 test case fitted and labelled |
| [session-03_2026-10-14](session-03_2026-10-14/) | 2026-10-14 | Finishing the 2010 test case: labelled reports, year-2010 labelling, political-content scan, review pack |
| [session-04_2026-10-22](session-04_2026-10-22/) | 2026-10-22/23 | Design of the full analytical pipeline (nine period bins → BERTopic per bin → two Qwen3-1.7B agents labelling topics and building per-bin taxonomies → VADER → tagging of the master CSV); the author's brief (`00_brief.md`), the design document with 21 flagged issues and the decisions (`01_…`), codebook v3.0 draft (§4.9, §8) |

The codebook itself lives with the data: `data/codebook_soccer_posts.md` (source) and `.pdf` (rendered).
Session memories and re-run workflows are in `../memory/`.
