# Session 12 (2026-10-30): the reference model on the whole corpus — notes

*Companion to `00_approvals.md` (decisions W1–W4). Design §6.5. The folder: `results/analysis/reference/README.md`. The recipe:
`memory/workflows.md` §11.*

## 1. What the reference model is, and is not

The nine period-bin models give bin-local topic ids. Comparing the World Cup periods, and carrying TP's and BA's QUALTAG coding
(made per bin-local topic) beyond the bins they coded, was to go through a manual crosswalk of 6,245 topics to chapter themes
(`topic_crosswalk.csv → chapter_theme`, design F4/F43) — a blank template to this day. Alan Nemirovski's interim step: fit the
same pipeline once on the whole cleaned corpus, so that a **corpus-wide topic stands beside each post's period topic**. The
period topics then map onto the reference topics by post overlap, and the QUALTAG attributes coded on period topics can be
aggregated onto the reference topics from the coding done so far.

The reference model is **not the unit of analysis** (that remains the period topic, D7), **not a post variable** (W3: its
assignment lives in its own `documents.csv`; `post_variables.csv`, `labels_version`, S6 and the codebook are untouched) and
**not a tenth bin** (`bin_of()` never returns 0; `collect` never reads the folder). It is an interim mapping device.

## 2. Settings (what equals the bins, what differs)

| | The nine bins | The reference run (bin 0) |
|---|---|---|
| Documents | the bin's posts | every post (1,302,021; 1,289,838 modelled), each keeping its true `period_bin` |
| Embeddings | mpnet, cached per calendar year | the same cache — a full hit, nothing encoded |
| UMAP | `UMAP_PARAMS` (15 neighbours, 5 dims, cosine, seed 42), once per run | the same |
| HDBSCAN | `min_cluster_size` searched from the floor 25 by the density guard (D8: 50–75 topics per 10k) | **one fit at the floor** (W1: `min_cluster_size` 25, `min_samples` 10, eom, ε 0); `guard.enabled = false`, `guard.mode = "fixed"`, state `fixed` |
| Gate (§12.2) | raw outliers ≤ 0.50 and the guard converged or floor-bound, asserted by the notebook | not asserted; the share is printed beside 0.50 at the stop point |
| Outlier tiers | D9 from the bin's own quantiles (0.35 / 0.80); K1 counting | the same, from the run's own quantiles |
| Nearest topics of the outliers | one dense similarity matrix | the same values, computed in chunks of 50,000 outliers (`model.outlier_nearest_from`) |
| `topics_over_time.csv` | BERTopic's routine (weekly / monthly; the 2026-10-25/26 files keep the D9 counts, K7) | **model-free monthly counts** of the counted posts, `keywords` empty (`model.topics_over_time_counts`) |
| Report | every topic in the bar charts | the 200 largest topics in the bar charts (above 2,000 topics); the table complete |
| Labelling | Qwen3-1.7B, prompts v1.4, greedy, 40 posts per topic, one debate over all topics | the same, **in chunks of 500 topics** (`--chunk-topics 500`: `labels.jsonl` grows after every chunk) |
| Review pack | `review_pack.md` per bin for the co-authors | none (no co-author read of the reference list is planned) |
| Where | `results/analysis/bin-N/` | `results/analysis/reference/` (`config.run_dir_name(0)`) |
| In git | everything but `model/` | everything but `model/`, `documents.csv`, `labels/`, `umap_5d.npy`, `hdbscan_labels_raw.npy`, `outlier_nearest.csv` (W4) |

## 3. The run on Colab (notebook section 6) and the stop point

1. **6.1** — the cache check (17 years of `embeddings/sentence-transformers_all-mpnet-base-v2/<year>.npz`) and
   `--stage embed --bin 0`, which validates the ids and loads no model.
2. **6.2** (optional) — a mechanics smoke run of 100,000 posts into `/content/smoke` (`--max-docs 100000 --out-root`), the
   timing basis.
3. **6.3** — `--stage topics --bin 0` under `cuml.accel`, then the **stop point**: posts modelled, raw topics at
   `min_cluster_size` 25 (and what the D8 density would have asked for), the raw outlier share beside the bins' 0.50, the
   tiers in force, the elapsed time, and the labelling estimate at 3.5 s per topic. **Alan decides** whether to label.
4. **6.4** — `--stage label --bin 0 --chunk-topics 500` behind the switch `RUN_REFERENCE_LABEL`; resumable after a disconnect.
5. **6.5** — `handback_reference()`: `results/reference/` without `model/` and `labels/`, plus the aggregated tables and the
   log, as `handback/reference_<stamp>.zip`; unzipped at the repository root, `.gitignore` keeps the large files out of git.

Estimates from the bins (not measurements): the topics stage ≈ 1–2 h (bin 8, run alone: 38 min for 217k posts and 967
topics with the model-based topics over time); about 5,700–6,300 raw topics if the floor-bound interim bins' 44.6–48.9 raw
topics per 10,000 posts carry over; labelling 2.8–4.0 s per topic, so ≈ 5–7 h for 6,000 topics. Risks and fallbacks: cuML
UMAP/HDBSCAN on 1.29 M points is untried in this project — if HDBSCAN fails on the GPU, the CPU `hdbscan` package on the saved
`umap_5d.npy` (`--stage topics --bin 0 --force --keep-umap` without `cuml.accel`) is the fallback; RAM: the year matrices
(≈ 4 GB) and their aligned copy (≈ 4 GB), the documents as strings (several GB), the chunked similarities (≈ 1 GB) fit the
83 GB machine; `save_model` writes the whole vocabulary to Drive (hundreds of MB, slow, not fatal).

## 4. How to read the mapping tables (`code/16-reference-map.R`)

- **`map_period_topics.csv`** answers, per period topic, "where did its posts go?": `best_share` close to 1 means the period
  topic sits inside one reference topic; a low `best_share` with a high `n_ref_topics` means the reference model split it.
  `agree` says whether the reference topic nearest by centroid is also the one holding most posts — the two methods should
  agree for well-separated topics. Figure G1 shows `best_share` by bin: a reference model that "replaces the crosswalk"
  should show high shares in every bin, the tournament bins included.
- **`reference_topics.csv`** answers, per reference topic, "what is it made of?": `n_bin1 … n_bin9` (does it recur across
  periods, or is it one period's event?), `top_period_topics` (its names on the period side), and the QUALTAG composition:
  the shares `share_tp_yes` / `share_tp_no` / `share_tp_review` / `share_tp_to_settle` from TP's bins, `share_ba_1` /
  `share_ba_0` / `share_ba_blank` from BA's bins 2 and 4 (BA's blank is undocumented and reported as such), `share_uncoded_bin`
  (bins 6 and 8), `share_unmapped` (posts the period side did not count). A reference topic whose counted posts come mostly
  from `tp_yes` period topics is the candidate for a political tag; the rule (which share, which coder, how the categories
  combine) is **not** decided here (W2).
- **`map_pairs.csv`** is the full cross-tabulation behind both, for any other reading.

## 5. The K1 / K7 note

The reference run's `topics_over_time.csv` carries K1 counts (cluster + reassigned posts per month), because it is computed
without the model; the nine bins' files keep the D9 counts of the 2026-10-25/26 run (K7) until a Colab pass regenerates them.
Nothing reads either file but the reports.

## 6. Not in this step

The QUALTAG transfer rule and any tag on a reference topic or a post (W2, after the tables); S6 in place and codebook v3.0
final (K5); the inter-coder check (K2); TP's 41 topics to settle; BA's bin-2/4 format; any change to the period-bin models,
their tiers or their counts (K1 stands).
