"""The analytical pipeline of the FIFA Communities chapter: period bins → BERTopic per bin → two Qwen3-1.7B analysts
→ VADER → tagging of the r/soccer post corpus (design: docs/session-04_2026-10-22/01_analytical-pipeline-design.md).

Modules (one per stage or concern; every setting in config.py)
--------------------------------------------------------------
config     all settings: paths (PIPELINE_ROOT), documents, period bins, embeddings, BERTopic and the density guard (D8),
           outliers, the analysts and the debate, the tone vocabulary, sentiment; setting(), target_topic_range(), effective_settings()
data       the yearly files, document text (title + body) and its cleaning, token counts
bins       stage S0: period bins (assignment, summary table, loading one bin's documents)
embed      stage S1: sentence embeddings with the per-year cache
model      stage S2: BERTopic with frozen UMAP and cluster labels, the topic-density guard and the HDBSCAN grid diagnostic,
           outlier similarities and the thresholded reassignment (D3), the per-bin tables
prompts    prompt v1.4: the two lenses, S3a labelling, the shared revision and repair turns
agents     stage S3: the language-model wrapper, the D2 debate engine, S3a cluster labelling, the controlled tone
           vocabulary, the topic overrides of the co-author review (the category layer was removed by decision D7)
sentiment  stage S4: VADER per post (title; title + body) with the hand-coding sample
tagging    stage S5: post_variables.csv, topic_labels_all_bins.csv (with mean_vader_sentiment), topic_crosswalk.csv
report     the per-bin report.html / report.md

Entry point: code/05-analysis-pipeline.py (one --stage per call).  Stage S6 (write-back into the data files) is
code/06-tag-corpus.R + code/07-validate-analysis.R.  Google Colab: code/05b-analysis-pipeline-colab.ipynb.
"""
