# Review guide — period bin 1 (the 2010 World Cup period, 1 January – 30 September 2010)

**For:** the co-authors reviewing the topics and categories of bin 1. Alan Nemirovski runs the Colab steps and commits.
**Written:** 2026-10-25 (session 08), after the first Colab run of bin 1 (`00_colab-run-log.md`). The same procedure applies
to every later bin with `bin-1` replaced by `bin-N`.

## 0. What you are reviewing, and why it matters

The pipeline found **65 topics** in the 9,809 posts of bin 1 that fit a cluster (the other 3,414 posts are outliers for now).
Two AI analysts (the same language model with two reading instructions) agreed on a label for each topic — a name, a
purpose, an emotional tone, a one-sentence description and a few inclusion cues — and then on a set of **34 categories**
into which the topics were sorted. All of this is provisional. The chapter's analysis will use **your** corrected version,
and the method requires that corrections are made through three small CSV files, never by editing the outputs: the
pipeline applies your corrections itself, keeps the analysts' original values beside yours (`*_agent` columns) and records
what was changed. Write the reason for every change in the `notes` column — the files are part of the chapter's record.

## 1. Where everything is

Repository (GitHub `alannemi/fifa-reddit-communities`, or the folder Alan shares with you), folder `results/analysis/bin-1/`:

| File | What it is | Edit? |
|---|---|---|
| `review_pack.md` | **Start here.** Every topic with its agreed label, both analysts' answers, its category, and the 40 sample posts the analysts read; every category with its members and definition | no |
| `report.html` | Charts: topic sizes, prevalence over the period (kick-off and final marked), sentiment by topic, the topic table | no |
| `topics.csv`, `topics_labelled.csv` | The topics (sizes, keywords, five representative titles) and their agreed labels | no |
| `categories.csv`, `topic_categories.csv` | The categories (definitions, members) and the topic → category table with the review flag and the similarity rank | no |
| `labels/topic_<id>.json` | The full transcript of one topic's labelling debate (only if you want to see every turn) | no |
| **`topic_overrides.csv`** | pass 1: rename or merge topics | **yes** |
| **`category_overrides.csv`** | pass 2: rename or merge categories | **yes** |
| **`topic_category_overrides.csv`** | pass 3: move one topic to another category | **yes** |
| `outlier_calibration/sample_T0.5428.csv`, `sample_T0.5789.csv`, `sample_T0.6173.csv` | the outlier judgement samples (§6) | **yes** |

The pipeline runs on Google Drive (`MyDrive/fifa-communities/results/bin-1/`, the same files). Alan uploads your filled
files there before the next run; you do not need Drive access.

## 2. How to read

Open `review_pack.md` (any Markdown viewer; VS Code's preview, or GitHub's rendering). Section 1 is the list of topics with
size, agreed name, category, the similarity rank of that category (1 = the embeddings' nearest; higher = a less obvious
choice) and a **review** flag. Section 2 lists the categories with their member topics and definitions. Section 3 has one
block per topic: the agreed label, what each analyst said in its last turn, the keywords, the category and the ten
candidate categories the analysts chose from, five representative titles, and — folded away under "The 40 sample posts" —
the posts themselves. Read the posts; the label is right only if it describes them.

## 3. Pass 1 — the topics (`topic_overrides.csv`)

One row per topic; the first three columns are filled by the pipeline, the last three are yours:

| Column | Meaning |
|---|---|
| `topic_id`, `name`, `needs_review` | the topic, its agreed name, whether the analysts failed to agree (none in bin 1) |
| `final_name` | a better name, if the agreed one is wrong or too narrow; leave empty to keep |
| `merge_into` | the `topic_id` of the topic this one should be folded into (its posts move there; its own label disappears) |
| `notes` | why |

What to look for: two topics that are the same conversation (merge the smaller into the larger); a name that describes
only part of the posts (rename); posts that are not discussion at all (the three live-stream spam topics — merge them into
one and name it so, or leave them, your call); a non-English topic (the Italian one, topic 20 — keep, rename, or note).
Candidates seen so far: 26 and 37 (vuvuzelas), 42 and 57 (Mourinho), 24 and 35 (MLS), 11, 22 and 53 (live-stream spam);
renames for 0 (these are the World Cup match threads, not predictions) and 3 (Premier League match threads, not Arsenal).

Example rows (the header stays exactly as it is):
```
topic_id,name,needs_review,final_name,merge_into,notes
0,World Cup 2010 Predictions,False,World Cup match threads,,group-stage match threads and brackets; "predictions" is one strand
3,Arsenal Match Thread,False,Premier League match threads,,Arsenal is one club among many in the sample
37,Vuvuzela Controversy,False,,26,same conversation as topic 26 (FIFA ban debate = the noise complaints)
```
Rules: a `merge_into` target must be another topic of this bin and must not itself be merged away in a cycle (A → B → A);
a chain (A → B, B → C) is allowed and resolves to C. A merged topic cannot also be renamed (its name disappears).

**A filled topic file changes the set of topics, so the categories are rebuilt from scratch after it** (Alan: Colab,
`stage('--stage taxonomy --bin 1 --force')`, about 15 minutes, then a new hand-back and a new `review_pack.md`). Passes 2
and 3 therefore wait for that re-run — their category ids will be different.

## 4. Pass 2 — the categories (`category_overrides.csv`), after the re-run

| Column | Meaning |
|---|---|
| `category_id`, `name`, `n_topics`, `needs_review` | the category as the analysts defined it |
| `final_name` | a better name; leave empty to keep |
| `merge_into` | the `category_id` of the category this one should be folded into |
| `notes` | why |

What to look for: near-duplicates (in the first taxonomy: two vuvuzela categories; three streaming categories; two MLS
categories); categories named after a single match or person that the analysts then used generically ("Sevilla vs
Barcelona Match Discussion" holds four match-thread topics — rename it to what it is, or merge it with the other match
category under a generic name); single-topic categories that could join a broader one. There is no target number of
categories: the brief says to group topics only when they are similar, so merging is a judgement, and the record of it is
your `notes`.

## 5. Pass 3 — individual moves (`topic_category_overrides.csv`), after the re-run

| Column | Meaning |
|---|---|
| `topic_id`, `name` | the topic |
| `category_id_agent`, `category_name` | where the analysts put it |
| `category_assignment` | `agent` (the analysts chose) or `repaired` (placed by similarity because the analysts gave no usable answer) |
| `assignment_needs_review` | **True** = read this one first (the analysts did not agree, or their choice ranks low by similarity) |
| `similarity_rank` | where the chosen category ranks among all categories by similarity to the topic (1 = nearest) |
| `final_category_id` | the `category_id` (from `categories.csv`) where the topic belongs; leave empty to keep |
| `notes` | why |

Work through the rows with `assignment_needs_review` = True, then those with rank 2 or 3, then skim the rest against
section 2 of the pack. Use the ids of the current `categories.csv`; a topic moved to a category that you also merge in
pass 2 follows the merge.

## 6. The outlier judgement samples (`outlier_calibration/`), same sitting

3,414 posts of bin 1 fit no cluster. The design re-attaches an outlier to its nearest topic when the similarity between them
is at least a threshold `T`, to be chosen from three candidates. Each `sample_T<value>.csv` holds 100 outlier posts whose
similarity lies just above that candidate, with the nearest topic's keywords:

| Column | Meaning |
|---|---|
| `id`, `title` | the post |
| `nearest_topic`, `nearest_topic_keywords` | the topic it would join |
| `similarity`, `candidate_threshold`, `band_upper` | where the post sits relative to the candidate threshold |
| **`belongs`** | your judgement: `yes` if the post belongs with that topic, `no` if not |
| `notes` | optional |

Fill `belongs` for all 100 rows of all three files (300 judgements, about an hour). The threshold is set where the yes
answers clearly dominate; the same files exist for the large bin 4 and both are needed before `T` is fixed. Two coders
would allow an agreement figure for the codebook.

## 7. Checking and handing over

1. Save each file as plain CSV, UTF-8, comma-separated, the header unchanged, no rows added or deleted. Excel and Numbers
   can edit them; save as CSV, and check that ids such as `cat_10` and `26` survived unchanged.
2. From the repository root: `python code/08-review-pack.py check --bin 1` — it reports ids that do not exist, merge
   cycles, moves of topics that are merged away, and counts what you filled in. Fix anything marked PROBLEM.
3. Commit the files (or send them to Alan). Alan uploads the three files to Drive and runs the next Colab stage:
   after pass 1, `stage('--stage taxonomy --bin 1 --force')` (categories rebuilt, 15 min), then the hand-back and a new
   pack for passes 2 and 3; after passes 2 and 3, `stage('--stage collect --allow-partial')`, which applies all three files
   (topic merges, then moves, then category merges) and records what it applied in `data/analysis/post_variables.json`.

Time: pass 1 about one to two hours for 65 topics; passes 2 and 3 about an hour; the outlier samples about an hour.
