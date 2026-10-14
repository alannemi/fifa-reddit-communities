# Progress report for co-authors (2026-10-14)

The shareable, editable version of this report is the Claude Doc "r/soccer corpus: progress report for
co-authors" (https://claude.ai/code/artifact/7603e652-16d6-4a27-93e7-b1d7df8bb9a3). This file is the
repository record of its content at the time of writing.

## Summary
Clean, documented corpus of 1,302,021 r/soccer posts (Jan 2010 – Jul 2026) and a working BERTopic pipeline
tested on the 2010 World Cup. Needed next: five co-author decisions before the full run.

## The corpus
2,047,322 raw posts from Arctic Shift (retrieved 2026-08-09). Posts only (no comments); 88 % title-only;
crawl-time snapshot; removal metadata only from mid-2019; continuous monthly coverage.

## Data cleaning (v2.1)
Rebuilt from raw with readr after the fread quote-doubling defect. Stages: S1 empty (0), S2 deleted
(283,433), S3 removed (375,613), S4 bot threads (48,156), S5 promotional (34,974), S6 encoding (335; 126
titles repaired), S7 non-English (2,790); retained 1,302,021 (63.6 %). 28 validation checks pass. Retention
96–99 % for 2011–2014 vs 37–42 % for 2020–2022 (removal metadata era effect).

## Codebook and reproducibility
`data/codebook_soccer_posts.md/.pdf` (v2.1); scripts `code/02-data-cleaning.R`, `03-data-validation.R`,
`05-topic-model.py` + `code/topicmodel/`, `05b-topic-model-colab.ipynb`; docs per session; `memory/`.

## Topic-modelling approach
One post = one document (title + ≤1,000 chars body). MiniLM embeddings → UMAP → HDBSCAN (min topic size
0.25 %, floor 15) → c-TF-IDF keywords (KeyBERT + MMR; scaffolding stop-words) → two-analyst LLM labelling
(Qwen3; name, purpose, tone, description, cues; ≤3 rounds) → editable merge table → VADER sentiment baseline
→ year models aligned within WC eras + ±2-month tournament windows.

## 2010 test case
Window 11 Apr–11 Sep 2010: 7,818 posts, 93 topics, fully labelled (12 flagged for review). Year 2010: 13,638
posts, 94 topics, labelling in progress. Largest labelled window topics: match threads/IRC; Premiership and
Champions League controversies; referee criticism; Inter–Barcelona–Bayern; predictions and standings; soccer in
America; vuvuzela controversy; streaming and replays; Spain controversy; Netherlands–Uruguay; posters and art;
upsets and underdogs; Mexico–USA; South Africa 2010; England squad; Maradona satire; France team dynamics; Paul
the octopus. Tones: enthusiastic 38, frustrated 17, critical 17, others ≤4 each. Sentiment: 41 % positive,
39 % neutral, 20 % negative.

## Takeaways from the first round
1. Tournament discourse is event- and team-centred. 2. A strong community/"meta" layer (how to watch, community
artefacts, "soccer in America"); r/soccer 2010 reads as a largely North American, English-speaking community.
3. Enthusiastic by default; frustration aimed at institutions (referees, FIFA/Blatter, conditions, streaming)
and national-team campaigns. 4. Club football does not pause for the World Cup. 5. Politics appears as a lens,
not a topic (~4 % of posts): geopolitics through a team (North Korea), governance/corruption (bids, FIFA,
Triesman), host-nation development/security (South Africa), national identity/race (France, Özil/Khedira/Klose,
Nigeria). 6. The community polices the boundary itself ("keep politics out"), so politicisation is reactive
and incident-tied. 7. Measuring politicisation needs a targeted layer. 8. Labels usable, mostly consistent.
9. Granularity and outlier handling shape results most. 10. Sentiment directionally sensible but shallow.

## Decisions needed
1 granularity; 2 outlier handling; 3 political-content layer (none / keyword flag / guided topics / LLM
classification); 4 models for the full run (MiniLM vs mpnet; Qwen3-1.7B vs 4B); 5 keyword hygiene.
Discussion items: treatment of human match threads; validating sentiment.

## Next steps
Finish year-2010 labels (+ optional 4B comparison) → apply decisions → test Colab on one year → full run
(17 years, 5 windows, 4 eras) → review the 2010–13 era first → write codebook section 8 and the chapter's
methods text.
