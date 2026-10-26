# Review guide — period bin 1 (the 2010 World Cup period, 1 January – 30 September 2010)

**For:** the co-authors reviewing the topics of bin 1. Alan Nemirovski runs the Colab steps and commits.
**Written:** 2026-10-25 (session 08), after the first Colab run of bin 1; **revised the same evening** for decision D7: the
individual topic is the unit of analysis, so the review is one pass over the topics (the category passes of the first
version are gone). The same procedure applies to every later bin with `bin-1` replaced by `bin-N`.

## 0. What you are reviewing, and why it matters

The pipeline found **65 topics** in the 9,809 posts of bin 1 that fit a cluster (the other 3,414 posts are outliers for now).
Two AI analysts (the same language model with two reading instructions) agreed on a label for each topic — a name, a
purpose, an emotional tone, a one-sentence description and a few inclusion cues. These labels are provisional, and they
matter: every post of a topic will carry the topic's final name and tone in the shared data files (`topic_name`,
`topic_tone`), and the chapter's analysis works at the level of these topics. Corrections are made through one small CSV
file, never by editing the outputs: the pipeline applies your file itself, keeps the analysts' original name beside yours
(`name_agent`) and records which topics were merged (`merged_from`). Write the reason for every change in the `notes`
column — the file is part of the chapter's record.

## 1. Where everything is

Repository (GitHub `alannemi/fifa-reddit-communities`, or the folder Alan shares with you), folder `results/analysis/bin-1/`:

| File | What it is | Edit? |
|---|---|---|
| `review_pack.md` | **Start here.** Every topic with its agreed label, both analysts' answers, the keywords, five representative titles and the 40 sample posts the analysts read | no |
| `report.html` | Charts: topic sizes, prevalence over the period (kick-off and final marked), sentiment by topic, the topic table | no |
| `topics.csv`, `topics_labelled.csv` | The topics (sizes, keywords, representative titles) and their agreed labels | no |
| `labels/topic_<id>.json` | The full transcript of one topic's labelling debate (only if you want to see every turn) | no |
| **`topic_overrides.csv`** | your corrections: rename or merge topics | **yes** |
| `outlier_calibration/sample_T<value>.csv` (four files, one per similarity band) | the outlier judgement samples (§4) | **yes** |

The pipeline runs on Google Drive (`MyDrive/fifa-communities/results/bin-1/`, the same files). Alan uploads your filled files
there; you do not need Drive access.

## 2. How to read

Open `review_pack.md` (any Markdown viewer; VS Code's preview, or GitHub's rendering). Section 1 lists the topics with size,
agreed name, tone, how the two analysts reached agreement, and a **review** flag where they did not (none in bin 1). Section
2 has one block per topic: the agreed label, what each analyst said in its last turn, the keywords, five representative
titles and — folded away under "The 40 sample posts" — the posts themselves. Read the posts; the label is right only if it
describes them.

## 3. The topic file (`topic_overrides.csv`)

One row per topic, 65 rows; the first three columns are filled by the pipeline, the last three are yours:

| Column | Meaning |
|---|---|
| `topic_id`, `name`, `needs_review` | the topic, its agreed name, whether the analysts failed to agree |
| `final_name` | a better name, if the agreed one is wrong or too narrow; leave empty to keep |
| `merge_into` | the `topic_id` of the topic this one should be folded into (its posts move there; its own label disappears) |
| `notes` | why |

What to look for: two topics that are the same conversation (merge the smaller into the larger); a name that describes only
part of the posts (rename); posts that are not discussion at all (the three live-stream spam topics 11, 22 and 53 — merge
them into one and name it so, or leave them, your call); a non-English topic (the Italian one, topic 20 — keep, rename, or
note). Candidates seen so far: 26 and 37 (vuvuzelas), 42 and 57 (Mourinho), 24 and 35 (MLS); renames for 0 (these are the
World Cup match threads, not predictions) and 3 (Premier League match threads, not Arsenal).

Example rows (the header stays exactly as it is):
```
topic_id,name,needs_review,final_name,merge_into,notes
0,World Cup 2010 Predictions,False,World Cup match threads,,group-stage match threads and brackets; "predictions" is one strand
3,Arsenal Match Thread,False,Premier League match threads,,Arsenal is one club among many in the sample
37,Vuvuzela Controversy,False,,26,same conversation as topic 26 (FIFA ban debate = the noise complaints)
```
Rules: a `merge_into` target must be another topic of this bin and must not itself be merged away in a cycle (A → B → A); a
chain (A → B, B → C) is allowed and resolves to C. A merged topic cannot also be renamed (its name disappears).

## 4. The outlier judgement samples (`outlier_calibration/`), same sitting

3,414 posts of bin 1 fit no cluster. Since decision D9 (2026-10-25) the pipeline sorts them by their similarity to the nearest
topic, measured against the bin's own distribution: a post at or above the **80th percentile** joins its nearest topic; between
the **35th and the 80th** it joins its nearest topic *and* carries the second-nearest topic as a second label (`dual`); below
the 35th it stays an outlier. Your judgements tell us how often the admitted posts really belong: each `sample_T<value>.csv`
holds 100 outlier posts whose similarity lies just above one cut (35th, 50th, 65th, 80th percentile — four files; the
`calibration_table.csv` beside them names which file is the floor's band), with the nearest topic's id, name and keywords:

| Column | Meaning |
|---|---|
| `id`, `title` | the post |
| `nearest_topic`, `nearest_topic_name`, `nearest_topic_keywords` | the topic it would join: its id, its agreed name (as in the review pack) and its keywords |
| `similarity`, `candidate_threshold`, `band_upper` | where the post sits relative to the cut |
| **`belongs`** | your judgement: `yes` if the post belongs with that topic, `no` if not |
| `notes` | optional |

Start with the lowest band (the 35th percentile file): it is the one the rule admits last, so its "yes" share decides whether
the floor stays or moves up. Then the other three (400 judgements in all, about an hour and a quarter). The same files exist for
the large bins; bin 1 and one large bin are needed. Two coders would allow an agreement figure for the codebook. The names are
the analysts' agreed names at the time the files were written; if you finish the topic file (§3) first, Alan refreshes them with
your renames (`python code/08-review-pack.py pack --bin 1` rewrites the name column and keeps everything you filled in).

## 5. Checking and handing over

1. Save the file as plain CSV, UTF-8, comma-separated, the header unchanged, no rows added or deleted. Excel and Numbers can
   edit it; save as CSV, and check that the topic ids survived unchanged.
2. From the repository root: `python code/08-review-pack.py check --bin 1` — it reports ids that do not exist and merge
   cycles, and counts the renames and merges. Fix anything marked PROBLEM.
3. Commit the file to GitHub (or send it to Alan). Alan's notebook cell 4.0 copies every committed override file to Drive and
   cell 4.1 re-runs the collect stage, which applies the renames and merges to every post and to the topic table and records what
   it applied in `data/analysis/post_variables.json`; the same collect can run on Alan's computer once the sentiment table is there.
   Nothing else has to re-run. A topic without any name (its two analysts both failed — bin 6 topic 369, bin 7 topic 67) must get a
   `final_name` before the final collect.

Time: one to two hours for the 65 topics; about an hour and a quarter for the outlier samples.

## 6. The larger bins

Under the density target of decision D8 the interim bins will hold 1,000–2,500 topics each. A full read is not feasible there:
the design's review is a sample of 20 topics read against their posts plus every topic flagged `needs_review`; the author
decides whether a larger sample is needed.
