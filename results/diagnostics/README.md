# results/diagnostics/

Diagnostics of the analytical pipeline — not chapter figures — moved here from `results/descriptives/` on 2026-10-31 (session
14, decision V10), when that folder was given over to the chapter's descriptive set. Same conventions: `fig_<id>.png` (300 dpi)
+ `.pdf`, the numbers in `tab_<id>.csv`; every CSV is read as text.

| id | Content | Made by |
|---|---|---|
| D1_topic_network_families, D1b_…_unlabelled, D2_topic_network_periods; tables D1_topic_families / _network_nodes / _network_edges | The topic network by semantic similarity of the topic embeddings (all 6,245 topics; the eight largest families; one panel per period bin) | `code/12-topic-network.R` (~20 s; the embeddings through the `.venv` Python) — session 09 |
| F1_outlier_precision_by_band, F2_outlier_precision_by_similarity; tables F1–F4b | The precision of the outlier tiers from the 36 coded judgement samples (3,600 posts): per bin and band, pooled, by similarity; what each candidate quantile pair would leave as outliers | `code/15-outlier-calibration.R` (~30 s) — session 11, decisions K1–K7 |
| G1_reference_best_share_by_bin | The best-match share of the period topics in the whole-corpus reference model, per bin (written once the reference run is handed back) | `code/16-reference-map.R` (`REF_FIG_DIR`) — session 12 |
| sentiment_validation_agreement / _confusion / _by_class / _by_stratum / _thresholds | VADER against the human coding of the 337-post sample: agreement, κ, the confusion matrix, per class, per stratum, the threshold sensitivity | `code/10-sentiment-validation.R` (`SENT_OUT_DIR`) — session 09, decision E9 |

Records: `docs/session-09_2026-10-27/`, `docs/session-11_2026-10-30/`, `docs/session-12_2026-10-30/`.
