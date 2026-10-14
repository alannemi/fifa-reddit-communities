"""BERTopic topic-modelling pipeline for the r/soccer post corpus (FIFA Communities chapter).

Modules
-------
config     all settings (paths, windows, hyper-parameters, model names, seeds)
data       loading yearly CSVs, building and cleaning document text, window filters
embed      sentence embeddings with a per-year cache
model      BERTopic construction, fitting, outlier reduction, topics over time, era merging
label      LLM topic labelling (two-analyst debate, JSON output) and manual merge tables
sentiment  VADER sentiment per document and summaries
report     CSV tables and HTML/Markdown report with charts

Entry point: code/05-topic-model.py
"""
