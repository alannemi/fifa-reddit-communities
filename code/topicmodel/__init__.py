"""The analytical pipeline of the FIFA Communities chapter: period bins → BERTopic per bin → two Qwen3-1.7B analysts
→ VADER → tagging of the r/soccer post corpus (design: docs/session-04_2026-10-22/01_analytical-pipeline-design.md).

Modules (one per stage or concern; every setting in config.py)
--------------------------------------------------------------
config     all settings: paths (PIPELINE_ROOT), documents, period bins, embeddings, BERTopic and the guard, outliers,
           the analysts and the debate, the taxonomy protocol, the tone vocabulary, sentiment; effective_settings()
data       the yearly files, document text (title + body) and its cleaning, token counts
bins       stage S0: period bins (assignment, summary table, loading one bin's documents)
embed      stage S1: sentence embeddings with the per-year cache
model      stage S2: BERTopic with frozen UMAP and cluster labels, the topic-count guard, outlier similarities and the
           thresholded reassignment (D3), the per-bin tables
prompts    prompt v1.2: the two lenses, S3a labelling, S3b definitions and assignment, the shared revision and repair
           turns (the v1.1 single-prompt texts are kept for the partition protocol)
agents     stage S3: the language-model wrapper, the D2 debate engine, S3a cluster labelling, the S3b two-step
           taxonomy (and the superseded partition protocol), the controlled tone vocabulary, the review overrides
sentiment  stage S4: VADER per post (title; title + body) with the hand-coding sample
tagging    stage S5: post_variables.csv and the aggregated label and category tables
report     the per-bin report.html / report.md

Entry point: code/05-analysis-pipeline.py (one --stage per call).  Stage S6 (write-back into the data files) is
code/06-tag-corpus.R + code/07-validate-analysis.R.  Google Colab: code/05b-analysis-pipeline-colab.ipynb.
"""
