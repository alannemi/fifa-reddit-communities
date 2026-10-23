"""BERTopic topic-modelling pipeline for the r/soccer post corpus (FIFA Communities chapter).

Modules
-------
config     all settings (paths, bins, hyper-parameters, model names, seeds, prompts' versions)
data       loading yearly CSVs, building and cleaning document text, window filters
embed      sentence embeddings with a per-year cache (shared by 05 and 06)
model      BERTopic construction and fitting; for 06: frozen UMAP/cluster components, topic-count guard,
           thresholded outlier reassignment, per-bin tables
label      session-02 LLM labelling (used by 05) and the tone_category mapper (used by both)
sentiment  VADER sentiment per document (05) and per corpus year with title/full measures (06, S4)
report     CSV tables and HTML/Markdown report with charts
bins       stage S0 of 06: period bins (assignment, summary, loading a bin's documents)
prompts    prompt v1.1 of the two-agent labelling (06, S3)
agents     the two-agent debate engine, S3a cluster labelling, S3b taxonomy, overrides (06)
tagging    stage S5 of 06: post_variables.csv and the aggregated label tables

Entry points: code/05-topic-model.py (exploratory year/window/era runs, session 02)
              code/06-analysis-pipeline.py (the analytical pipeline, sessions 04–05)
"""
