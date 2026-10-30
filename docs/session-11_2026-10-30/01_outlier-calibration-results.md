# The outlier calibration: what the coded judgement samples say (2026-10-30)

*Assessment by Claude for Alan Nemirovski, session 11. Decisions taken on it: `00_approvals.md` (K1–K7). Numbers from a
read-only analysis of the 36 coded files on 2026-10-30; `code/15-outlier-calibration.R` reproduces them (tables
`results/descriptives/tab_F*.csv`, figures F1–F2).*

## 0. What the tiers are, and why this check exists

Decision D9 (2026-10-25, design §6.3) sorts every raw HDBSCAN outlier by its cosine similarity to its nearest topic and
cuts at two percentiles of the bin's own distribution: at or above the 80th percentile the post joins its nearest topic
(`reassigned`); between the 35th and the 80th it joins it and carries the second-nearest topic as a second label (`dual`);
below the 35th it stays an `outlier`. The 35th percentile was set "conservative still" and explicitly provisional: the
percentiles are a rank, not a quality level, and the coded judgement samples were to measure the precision of the admitted
bands. Over the whole corpus (collect of 2026-10-27): cluster 712,218 · reassigned 115,526 · dual 260,054 · outlier 202,040
(15.5 %) · not modelled 12,183.

## 1. What was coded

- 9 bins × 4 bands × 100 posts = **3,600 judgements**, `belongs` = 1/0, no notes; every file's rows are identical to the
  blank sample the pipeline wrote (checked). Each band holds posts whose similarity lies just above one cut: the
  35th-percentile band (what the floor T admits last), the 50th and 65th (the rest of the dual tier) and the 80th-and-above
  band (the `reassigned` tier).
- Coders: **BA** bins 1, 3, 5, 7 · **AM** bins 6, 8, 9 · **AN** bins 2, 4. Coder and bin are confounded by construction.
- The instruction (`../session-08_2026-10-25/01_review-guide-bin-1.md` §4) was only: `yes` if the post belongs with that
  topic, judged against the topic's id, agreed name and eight keywords.
- A 100-post band carries a 95 % interval of about ±10 points; a pooled 900-post band about ±3.3 points.

## 2. Results

### 2.1 Share of posts judged to belong, per bin and band

| Bin | Type | Coder | 35th band (T) | 50th | 65th | ≥ 80th (reassigned) |
|---|---|---|---|---|---|---|
| 1 | tournament | BA | 19 % | 19 % | 22 % | 39 % |
| 2 | interim | AN | 52 % | 53 % | 52 % | 57 % |
| 3 | tournament | BA | 41 % | 38 % | 38 % | 51 % |
| 4 | interim | AN | 39 % | 29 % | 67 % | 65 % |
| 5 | tournament | BA | 16 % | 28 % | 32 % | 51 % |
| 6 | interim | AM | 51 % | 67 % | 60 % | 69 % |
| 7 | tournament | BA | 39 % | 35 % | 32 % | 41 % |
| 8 | interim | AM | 57 % | 60 % | 69 % | 60 % |
| 9 | tournament | AM | 49 % | 56 % | 58 % | 48 % |

### 2.2 Pooled over the nine bins, every judgement counting once (Wilson 95 % intervals) — the figures of record (K2)

| Band / tier | n | Precision | 95 % CI |
|---|---|---|---|
| 35th band (what T admits) | 900 | 40.3 % | 37–44 |
| 50th band | 900 | 42.8 % | 40–46 |
| 65th band | 900 | 47.8 % | 45–51 |
| ≥ 80th (`reassigned`) | 900 | 53.4 % | 50–57 |
| **Dual tier (35th–80th)** | 2,700 | **43.6 %** | 42–46 |
| **All admitted (≥ 35th)** | 3,600 | **46.1 %** | 45–48 |

Other weightings, reported for transparency and not used in the codebook: weighted by the corpus posts each bin-and-band
holds, dual 50.2 % / reassigned 60.9 % / all 53.5 % (BA's tournament bins hold 16 % of the corpus's admitted outliers but
supply 44 % of the judgements); equal weight per coder, 45.7 % / 55.2 % / 48.1 %.

Reading: a little under half of the posts the tiers admit were judged to belong to the topic they received. The gradient
with similarity exists but is shallow: 13 points from the lowest band to the top band.

### 2.3 The coder effect is larger than the similarity effect

| Coder | Bins | 35th | 50th | 65th | ≥ 80th | All |
|---|---|---|---|---|---|---|
| BA | 1, 3, 5, 7 | 29 % | 30 % | 31 % | 46 % | 34 % |
| AN | 2, 4 | 46 % | 41 % | 60 % | 61 % | 52 % |
| AM | 6, 8, 9 | 52 % | 61 % | 62 % | 59 % | 59 % |

- Within the tournament bins, the only comparison that holds period type constant, AM's bin 9 scores 48–58 % while BA's
  bins 1, 3, 5 and 7 score 16–51 %.
- In a logistic model of `belongs` on similarity, the BA indicator has a coefficient of −0.8 to −1.0 on the log-odds scale,
  larger than the whole similarity effect across the admitted range.
- Examples at matched similarity (0.62–0.70). BA coded "Cesc Fabregas and Robin van Persie likely to miss opener" against
  *Injury Updates* as **no**, and "AC Milan are out of the Europa League" against *UEFA Disciplinary Actions* as **no**. AM
  coded "Anthony Gordon: I speak Spanish because as a kid I believed I would play for Barça" against *Guardiola's Career* as
  **yes**, and "Club World Cup likely to expand for 2029" against *World Cup Player Participation* as **yes**. The coders
  applied different standards, which the one-line instruction left open.
- Consequence: the differences between bins in §2.1 cannot be separated from differences between coders. No file measures
  inter-coder agreement. **K2: the inter-coder check is deferred to the chapter's editing stage**, and the codebook says so.

### 2.4 Similarity to the nearest topic is a weak predictor of belonging

| Predictor of `belongs` | AUC (0.5 = no information) |
|---|---|
| absolute cosine similarity | 0.61 |
| within-bin percentile rank (the D9 rule) | 0.56 |
| margin to the second-nearest topic (similarity − similarity_2) | 0.58 |
| similarity + margin | 0.64 |
| similarity + margin + coder | 0.67 |

- Inside the top band there is no further rise: posts between the 80th and 90th percentile 53.7 %, at or above the 90th
  53.1 %, at or above the 95th 56 % (n = 199). Raising the single-topic cut would not purify the `reassigned` tier.
- By **absolute** similarity, pooled: ≤ 0.55 → 16 % (n 127) · 0.55–0.60 → 34 % (522) · 0.60–0.65 → 41 % (1,037) ·
  0.65–0.70 → 50 % (955) · 0.70–0.75 → 60 % (541) · 0.75–0.80 → 60 % (243) · > 0.80 → 53 % (175). Precision climbs to
  about 60 % at a similarity of 0.70 and plateaus; even the most similar admitted posts are judged to belong 55–60 % of the
  time.
- The absolute cut points differ by bin under the percentile rule (T from 0.497 in bin 1 to 0.616 in bin 8). Bin 1's dual
  tier admits posts at similarities no other bin admits, and bin 1 has the lowest precision; but bin 1 was coded by the
  strictest coder, so the data cannot apportion this between the rule and the coder.
- Margin to the second topic: the quartile with the smallest margin scores 39 %, the largest 58 %; posts within 0.02 of
  their second topic (35 % of the sample) score 39 % against 50 % for the rest. A reading of why `dual` posts are ambiguous,
  not strong enough for a rule.
- Size of the nearest topic: no relation (37–55 % across size quartiles).

### 2.5 Where the misses are: a few attractor topics, and names too specific for the post

- In the small bins the admitted outliers concentrate: the ten most-attracting topics take 40 % of the admitted posts in
  bin 1 and 35 % in bin 7 (8–12 % in the interim bins, which have 1,000–1,600 topics).
- Topics with many coded posts and **no** post judged to belong: bin 1 *Sevilla vs Barcelona* (24 judged), *Terry &
  Rooney Controversy* (17), *Messi & Maradona Legacy* (16), *World Cup Highlights* (13); bin 7 *Ancelotti & Simeone's
  Management* (19). Topics with many coded posts and near-complete belonging: bin 7 *Loan Deals* (31 across two topics,
  100 %) and *Loan Transfers* (24, 92 %); bin 9 *VAR Disallowed Goals* (11, 91 %) and *Transfer Rumours* (10, 100 %).
- Reading: a **no** mostly means that the topic's *name* is more specific than the post (another Serie A match counted
  under *Inter vs Roma*; another manager under *Ancelotti & Simeone*), not that the post is unrelated in kind. Generic
  topics (transfers, loans, VAR, goals) absorb outliers correctly because almost any such post belongs. At the level of
  topic families or chapter themes the error is therefore smaller than at the level of the individual topic, the unit of
  analysis under D7. This is a reading of the examples, not a measured number.

## 3. What this means for the outlier threshold

1. **The 35th-percentile floor admits a band that is 60 % wrong by the coders' standard**, and the bands above it are not
   much better (43 %, 48 %). The dual tier is 44 % right on its first topic; whether the post belongs to its *second* topic
   was not asked, so the tier's "one of the two" precision is unknown (at least 44 %).
2. **Raising a quantile buys little purity for much coverage.** Floor 35th → 65th: the admitted dual band goes from 40 %
   to 48 %, and 173,000 posts (13 % of modelled posts) return to outlier. Single cut 80th → 90th: no measurable change.
   The shape of the similarity–precision curve, not the position of the cut, is the problem: no cut reaches much above 60 %.
3. **The `reassigned` tier, which the figures already use, is about half right.** The tone figures (C3, C4) and the
   political figures (E1–E7) use cluster + reassigned posts (decisions E3, Q7). Reassigned posts are 115,526 of that
   population of 827,744 (14 %); at 53 % precision about 54,000 posts, 6.5 % of the figure population, carry a topic a
   coder would reject. Cluster members have never been judged (K3: no baseline), so the error of the other 86 % is unmeasured.
4. **The topic table counted dual posts in both topics.** Of the 1,347,852 topic memberships that made up `n_total` across
   the corpus, 520,108 (38.6 %) came from the 260,054 dual posts; `n_total`, `share`, `mean_vader_sentiment` and
   `topics_over_time` of every topic were diluted in a way the figures were not. **This is what K1 changes.**
5. **The data files will carry `topic_uid`/`topic_name`/`topic_tone` on 260,054 dual and 115,526 reassigned posts.** The
   codebook already says the tiers are a rank and that a dual label records ambiguity; it now carries the numbers.

## 4. Options weighed

Effects computed from each bin's `outlier_nearest.csv` (the two nearest topics and similarities of every raw outlier);
shares of the 1,289,838 modelled posts. Differences from the collect's counts are under 100 posts (overrides).

| Option | Tiers after | Precision of admitted posts | Assessment |
|---|---|---|---|
| A. Keep D9; report the precision | as now: cluster 55.2 % · reassigned 9.0 % · dual 20.2 % · outlier 15.7 % | dual 44 %, reassigned 53 % | honest, cheapest; leaves the topic aggregates diluted (§3.4) |
| **B. Keep the per-post tiers; count only cluster + reassigned posts in the topic aggregates** | per post as now | aggregates from posts at ≥ 53 % precision; the hedge stays recorded on each post | **chosen (K1, Alan Nemirovski, 2026-10-30)**: aligns the topic table with the figures' population; reverses D9's counting rule; local, no Colab |
| C. Drop the dual tier (floor = 80th percentile) | reassigned 9.0 % · outlier 35.8 % | 53 % | one topic or none, but a third of the posts without a topic again, which D9 was made to avoid |
| D. Raise the floor to the 50th / 65th percentile | outlier 22.4 % / 29.1 %; dual 13.4 % / 6.7 % | dual band 43 % / 48 % | small purity gain, large coverage loss |
| E. Raise the single cut to the 90th percentile | reassigned 4.5 %, dual 24.6 % | reassigned unchanged | no effect on precision |
| F. One absolute similarity floor for all bins (0.65–0.70) | outliers 21–40 % per bin | 50–60 % | defensible in principle (one embedding space), but the coder confound prevents calibrating it from these samples; high coverage cost |
| G. Confine analyses to cluster posts | — | unknown | cannot be judged without a cluster baseline |
| H. Re-cluster (lower `min_samples`, `leaf`) to cut raw outliers | — | — | refit and relabel all nine bins (≈ 8 h A100), redo the co-author reviews; the QUALTAG tags are per topic id and would be invalidated: off the table |
| I. Topic-specific rule (no reassignment to attractor topics) | — | — | per-topic precision is not estimable from 100 posts per band; observation only |

What each candidate floor would leave as outliers, per bin (share of the bin's modelled posts):

| Bin | Raw outliers | 35th (now) | 50th | 65th | 80th | 90th | 95th |
|---|---|---|---|---|---|---|---|
| 1 | 34.8 % | 12.2 % | 17.4 % | 22.6 % | 27.8 % | 31.3 % | 33.1 % |
| 2 | 48.2 % | 16.8 % | 24.1 % | 31.3 % | 38.5 % | 43.3 % | 45.7 % |
| 3 | 45.8 % | 16.0 % | 22.9 % | 29.7 % | 36.6 % | 41.2 % | 43.5 % |
| 4 | 48.5 % | 16.9 % | 24.2 % | 31.5 % | 38.8 % | 43.6 % | 46.0 % |
| 5 | 43.2 % | 15.1 % | 21.6 % | 28.1 % | 34.6 % | 38.9 % | 41.1 % |
| 6 | 44.5 % | 15.6 % | 22.2 % | 28.9 % | 35.6 % | 40.1 % | 42.3 % |
| 7 | 35.6 % | 12.5 % | 17.8 % | 23.1 % | 28.5 % | 32.0 % | 33.8 % |
| 8 | 40.0 % | 14.0 % | 20.0 % | 26.0 % | 32.0 % | 36.0 % | 38.0 % |
| 9 | 40.1 % | 14.0 % | 20.1 % | 26.1 % | 32.1 % | 36.1 % | 38.1 % |

## 5. Method of the assessment

Python (`pandas`, `scikit-learn` from the project's `.venv`) over the 36 coded files, the nine `calibration_table.csv`,
`outlier_nearest.csv` (second topic, margin), `topics.csv` (`n_cluster`) and the bins' `config.json` (`n_modelled`).
Precision = share of `belongs = 1`; Wilson intervals; logistic regressions without regularisation; AUC on the fitted
probabilities; attractor concentration on the admitted posts (similarity ≥ T) of each bin. `code/15-outlier-calibration.R`
is the script of record and reproduces these numbers.
