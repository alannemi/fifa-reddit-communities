# Session 02 (2026-10-13) — Topic-model test case: 2010 World Cup

> **Note (2026-10-23, session 05).** The run folders described here (`results/topic-models/window-wc2010/`, `year-2010/`) and the local MiniLM embedding cache were removed from the repository on the author's instruction before the analytical pipeline was implemented (they are in git history up to commit `6eeb0fee`). They can be regenerated with `code/05-topic-model.py` (see `memory/workflows.md` §6); the chapter's analysis uses the period-bin pipeline of session 04/05 instead. This document is kept as the historical record of the test case.

Companion to `02_topic-modelling-design.md` (the approved design). This note records what was run, the
first numbers and first observations, so co-authors can review the outputs in `results/topic-models/`.

## What was run

| Run | Documents | Modelled | Min. topic size | Topics | Outliers before → after reassignment | Time (Mac M4) |
|---|---|---|---|---|---|---|
| `window-wc2010` (11 Apr – 11 Sep 2010) | 7,818 | 7,678 (140 too short) | 19 | 93 | 36.6 % → 0.1 % | 4.4 min |
| `year-2010` (whole year) | 13,638 | 13,376 (262 too short) | 33 | 94 | 39.7 % → 0.1 % | 2.9 min |

Settings as in the design document (all-MiniLM-L6-v2 embeddings; UMAP 15/5; HDBSCAN 0.25 % floor 15;
1–2-grams; KeyBERT-inspired + MMR keywords; VADER sentiment). Both runs also saved a reduced model
(`model_reduced/`, `nr_topics="auto"`: 72 topics for the window) for comparison. Each run folder contains
`report.html` (charts + topic table), `report.md`, `topics.csv`, `documents.csv`, `topics_over_time.csv`,
`sentiment_by_topic.csv`, `sentiment_by_period.csv`, `topic_merge_table.csv`, `config.json`, `log.txt`.

## First observations (before LLM labels)

**The tournament surfaces as expected.** Among the largest window topics: match-thread/how-to-watch
discussion (topic 0), vuvuzelas and the Jabulani ball (4), refereeing and officiating (3; the Lampard
disallowed goal appears in the year model's topic 10 as "disallowed goal"), Spain's title run (7), the
Netherlands (8), England's squad/manager (17, 19, 23), France's training-camp implosion (11), Maradona's
Argentina (20), Paul the octopus (14), "is soccer American" debates (6), MLS and Landon Donovan (29, 45).
Club football continues alongside: Premier League (1), Serie A/La Liga (2), Mourinho to Real Madrid (18),
transfer speculation (28), Joe Cole to Liverpool (49).

**Granularity.** 93 topics, largest 2.8 % of posts, median 73 posts — readable but on the fine side; the
reduced model (72) and the merge table are the tools to coarsen. Human templated threads are a minority
inside topics (`share_match_thread` ≤ 25 % in topic 0, ≈ 0 elsewhere), so they do not dominate.

**Outlier reassignment is aggressive.** HDBSCAN leaves 37–40 % of posts unassigned; the c-TF-IDF
reassignment places almost all of them. This keeps every post in a topic (useful for prevalence over
time) but dilutes topic purity. Options for review: keep the raw outliers (set `REDUCE_OUTLIERS = False`),
or reassign only above a similarity threshold. The before/after shares are in every `config.json`.

**Sentiment baseline.** Window posts: 40.6 % positive, 39.4 % neutral, 20.0 % negative (VADER);
topic means range from −0.14 (refereeing/diving complaints) to +0.54 (celebratory/"treasured memory"
topics). As planned this is a lexicon baseline only.

**Known artefacts to discuss.** Keyword lists still contain some scaffolding ("soccer soccer", "vs",
"match") — candidates for the extra stop-word list; a few topics are bilingual (Italian posts on Serie A,
topic 18 in the year model) because the language filter kept short, name-heavy titles.

## LLM labels

Labelled on 2026-10-13/14 with **Qwen3-1.7B** (the fast fallback; chosen because 93 topics × two analysts × up to
three rounds took ~65 minutes on this Mac, versus an estimated 2–3 h for Qwen3-4B, which is now downloaded and can
relabel the same topics with `--label-only … --llm Qwen/Qwen3-4B` for a comparison). Every topic received valid JSON
from both analysts; the analysts agreed after a mean of 1.37 rounds; **12 of 93 topics are flagged
`needs_review`** (names or tones did not converge; both analysts' answers are kept in `topics.csv`).

Tone categories (controlled vocabulary derived from the free-text tones): enthusiastic 38, frustrated 17, critical 17, excited 4, other 4, humorous 2, angry 2, curious 2, anxious 2, neutral 2, positive 2, negative 1.
Enthusiastic/excited tones dominate the tournament window; frustrated/critical tones concentrate on refereeing, the
vuvuzela, Spain's style, England's campaign and streaming problems.

The 30 largest topics with their labels:

| topic | posts | LLM name | LLM tone (free text) | tone category | review? |
|---|---|---|---|---|---|
| 0 | 218 | World Cup Match Threads and IRC Discussions | enthusiastic, excited | enthusiastic | yes |
| 1 | 218 | Premiership and Champions League Controversies | Anxious, frustrated, and competitive | frustrated |  |
| 3 | 187 | Referee criticism and calls for reform | Critical and frustrated | frustrated |  |
| 2 | 176 | Inter Milan vs Barcelona and Bayern Munich | enthusiastic | enthusiastic |  |
| 5 | 172 | World Cup Predictions and Standings | enthusiastic | enthusiastic |  |
| 6 | 166 | Soccer in America | critical and skeptical | critical |  |
| 4 | 162 | Vuvuzela controversy | frustration and anger | frustrated |  |
| 9 | 160 | World Cup streaming and replays | frustrated and excited | frustrated | yes |
| 7 | 157 | Spain World Cup Controversy and Rivalry | Excitement and frustration | frustrated |  |
| 8 | 138 | Netherlands vs. Uruguay | Disgust and excitement | excited |  |
| 10 | 126 | World Cup posters and football art | funny, humorous, nostalgic | humorous |  |
| 16 | 121 | FIFA & Soccer Fantasy Leagues | enthusiastic | enthusiastic |  |
| 13 | 120 | World Cup upsets and underdog teams | enthusiastic, excited, and frustrated | frustrated |  |
| 12 | 115 | Mexico vs USA Football Discussion | enthusiastic, supportive, and competitive | enthusiastic |  |
| 15 | 111 | South Africa World Cup 2010 | mixed (positive and critical) | critical |  |
| 17 | 105 | England squad and player reactions | enthusiastic, critical, and supportive | enthusiastic |  |
| 20 | 103 | Maradona-related satire and cultural references | Satirical and humorous | humorous |  |
| 19 | 103 | Fabio Capello and England World Cup | Critical and reflective | critical |  |
| 11 | 101 | France World Cup 2010 and Team Dynamics | Critical and frustrated | frustrated | yes |
| 14 | 98 | Paul the Octopus and World Cup Predictions | enthusiastic, passionate, and angry | angry |  |
| 28 | 96 | Barcelona vs Real Madrid Transfer Drama | enthusiastic, competitive | enthusiastic | yes |
| 22 | 93 | Soccer Kit and Team Color Discussions | Inquisitive and curious | curious |  |
| 45 | 93 | MLS match coverage and fan engagement | enthusiastic, excited, and supportive | enthusiastic |  |
| 18 | 91 | Mourinho and Real Madrid | Analytical and critical | critical |  |
| 23 | 89 | England World Cup optimism and criticism | optimistic and critical | critical |  |
| 21 | 87 | World Cup music and chants | enthusiastic, nostalgic, and humorous | enthusiastic |  |
| 29 | 86 | Landon Donovan and Soccer | Excitement, admiration, and nostalgia | excited |  |
| 49 | 86 | Chelsea vs. Liverpool and Player Transfer Drama | frustrated, concerned, and critical | frustrated |  |
| 34 | 85 | World Cup and Football Fan Culture | enthusiastic, nostalgic, and humorous | enthusiastic |  |
| 26 | 84 | Italy World Cup 2022 Critique | Critical and frustrated | frustrated |  |

Flagged for review: 0 (World Cup Match Threads and IRC Discussions); 9 (World Cup streaming and replays); 11 (France World Cup 2010 and Team Dynamics); 28 (Barcelona vs Real Madrid Transfer Drama); 41 (Liverpool ownership and sale); 52 (Sir Alex Ferguson and Manchester United); 55 (Video requests and nostalgic football memories); 78 (World Cup and League Goal Analysis); 39 (Rafa Benitez and Liverpool); 67 (Live streaming of UEFA Champions League matches); 63 (Support for football clubs and leagues); 82 (Jürgen Klinsmann and Joachim Löw's Impact on German Football).

**Reading guide for co-authors.** Open `results/topic-models/window-wc2010/report.html` (charts + full table) and
`topic_merge_table.csv`: set `merge_into` to collapse near-duplicates (e.g. the several England topics 17/19/23, the
Liverpool topics 39/41/79) and `final_name` to overrule a label; re-running with `--apply-merges` records the decision.

