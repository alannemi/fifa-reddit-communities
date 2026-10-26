# docs/

> **Note (2026-10-25, session 07).** The pipeline scripts were renumbered (`06-analysis-pipeline.py` → `05-analysis-pipeline.py`,
> `06b-…-colab.ipynb` → `05b-…-colab.ipynb`, `07-tag-corpus.R` → `06-tag-corpus.R`, `08-validate-analysis.R` →
> `07-validate-analysis.R`), the session-02 exploratory tool (`05-topic-model.py`, `05b-topic-model-colab.ipynb`,
> `topicmodel/label.py`) was removed, the local embedding cache moved to `results/embeddings/`, and the Colab Drive layout
> became the flat folder `MyDrive/fifa-communities/`. The session folders 01–06 below use the names of their time;
> `session-07_2026-10-25/00_approvals.md` (decisions H1–H7) and `01_code-harmonization-notes.md` give the mapping.

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
| [session-05_2026-10-23](session-05_2026-10-23/) | 2026-10-23 | Implementation of the pipeline: approvals of the open flags and the refinements (`00_approvals.md`), implementation notes with the module map and deviations (`01_…`), the bin-1 dry run log (`02_…`), the runbook for the run session (`03_…`), the preliminary report on the test case for the author's review (`04_…`, also a Claude Doc); code `06-analysis-pipeline.py`, `topicmodel/{bins,prompts,agents,tagging}.py`, `06b` notebook, `07-tag-corpus.R`, `08-validate-analysis.R`, unit checks; the 2010 test outputs removed |
| [session-06_2026-10-24](session-06_2026-10-24/) | 2026-10-24 | The 92-topic dry run shows that Qwen3-1.7B cannot partition a whole bin in one prompt (finding D2); the author's decisions (`00_approvals.md`: two-step taxonomy protocol D5, N11/N12 confirmed, prompt v1.2, no Qwen3-4B); implementation of the two-step S3b and the Colab notebook with a session log on Drive (`01_…`); the runbook for the next session (`03_…`). The Colab test batch did not run in this session (the local dry run of the two-step stage was stopped and moved to Colab), so there is no `02_colab-run-log.md` |
| [session-07_2026-10-25](session-07_2026-10-25/) | 2026-10-25 | Harmonization of `code/` into one pipeline: the session-02 exploratory tool removed, one settings file with a path root (`PIPELINE_ROOT`), scripts renumbered, the Colab notebook rooted on `MyDrive/fifa-communities/`, shared R helpers (`code/R/pipeline-utils.R`), `03-data-validation.R` accepting the v3.0 spec; the author's decisions H1–H7 (`00_approvals.md`), the harmonization notes with the verification (`01_…`), the runbook for the Colab run (`02_…`) |
| [session-08_2026-10-25](session-08_2026-10-25/) | 2026-10-25 – 2026-10-26 | The Colab run (A100): bin 1, then decisions D6–D10 (topic as the unit of analysis, density guard, tiered outlier reassignment with a second topic, floor-bound bins), review-as-you-go hand-backs, all nine bins clustered and labelled (6,245 topics), S4 sentiment and S5 collect complete; the run log `00_colab-run-log.md` §1–§16 is the record, `01_review-guide-bin-1.md` the co-author procedure; defects fixed in flight (get_llm, label-record overwrite, nameless topics → salvage) |

The codebook itself lives with the data: `data/codebook_soccer_posts.md` (source) and `.pdf` (rendered).
Session memories and re-run workflows are in `../memory/`.
