# Session 03 (2026-10-14) — Run log: finishing the 2010 test case

Continuation of session 02. Session 02 ended (internet outage on the user's side) with the window-run
labelling in progress; it completed unattended at ~17:50 on 2026-10-13.

| Time | Step | Result |
|---|---|---|
| 2026-10-13 16:44–17:50 | `--label-only results/topic-models/window-wc2010 --llm Qwen/Qwen3-1.7B` (unattended) | 93/93 topics labelled, 0 errors, 12 flagged `needs_review` |
| 2026-10-13 ~17:00 | Qwen3-4B download completed (3 weight files, ~7.5 GB) | available for relabelling / comparison |
| 2026-10-14 16:52 | Re-ran `--label-only` on the window run to apply the stem-based tone mapping (resumed from `labels.jsonl`, no generation) | `topics.csv`, `report.html`, `topic_merge_table.csv` regenerated |
| 2026-10-14 16:53 | `--label-only results/topic-models/year-2010 --llm Qwen/Qwen3-1.7B` started in the background (94 topics, ~65 min) | resumable via `labels.jsonl` |
| 2026-10-14 | Labels section added to `../session-02_2026-10-13/03_topic-model-test-case.md`; political-content scan of the 2010 window documented below | — |

## Side analysis: political content in the 2010 window (question raised 2026-10-13)

No topic is political as a whole; political discussion runs *inside* sport-centred topics. A dictionary scan
(terms such as politics, government, president, apartheid, Mandela, xenophobia, racism, protest, corruption,
Blatter, human rights, North Korea, Kim Jong, immigration, police, security, economy, bid) flags ~4 % of window
posts, inflated by team names (North Korea, Serbia, Ivory Coast) and by transfer "bids". Concentrations:
topic 24 (North Korea's team as a window on the regime, 57 % of its posts flagged), topic 60 (2018/2022 hosting
bids, FIFA, stadium costs, the Triesman bribery allegation), topics 3 and 56 (FIFA governance via refereeing and
the Jabulani: Blatter, replay technology, "match fixing"), topic 15 (South Africa as host: economy, tourism,
policing, Mandela, Obama's visit), topic 11 (France's implosion → government reaction, racism), plus scattered
posts (Nigeria's president suspending the team; the "Germanness" of Özil/Khedira/Klose; Cameron–Merkel; Biden).
Options proposed for the research design: (1) a curated political-mention flag per post; (2) guided BERTopic with
a seed list for a "politics/governance/FIFA" topic; (3) LLM classification of posts into political sub-types.
Decision pending with the co-authors.
| 2026-10-14 17:30 | Progress report for co-authors written as a Claude Doc (link in `02_progress-report-for-coauthors.md`) with a repository copy | — |
| 2026-10-14 ~18:10 | Root `README.md` written; workspace made a git repository and pushed to GitHub (`alannemi/fifa-reddit-communities`, public) on top of the remote history (3,651 newer collector commits fetched; local work re-based onto them; license notice kept; data excluded via `.gitignore`, explained in `data/README.md`) | pushed |
