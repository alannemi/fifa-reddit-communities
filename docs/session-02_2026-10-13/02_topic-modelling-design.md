# Plan: BERTopic topic modelling of r/soccer posts (2010–2026)

## Context

With the cleaned corpus finished (v2.1: 1,302,021 posts, 2010-01 → 2026-07), the next step is the substantive analysis: discourse themes, their wording and tone, and sentiment in r/soccer over time, organised around the 2010, 2014, 2018, 2022 and 2026 World Cups. The attached paper (ICWSM submission, AI-generated sexual content on Reddit) is the methodological template: BERTopic clustering → LLM-labelled topics (name, description, emotional tone) in a two-analyst debate → manual merge → quantitative summaries per community/time. We adapt that to a single community (r/soccer) observed over 16 years, with **each post as one document**.

Today: a runnable test case on the 2010 World Cup (local Mac), producing the full output set for co-author review. Next session: review, tune, then scale to all years on Google Colab GPUs.

**Decisions already taken by you (2026-10-13):** topics characterised by their keywords + an LLM tone label (like the paper); document = title + body (body truncated); **both** year-by-year models grouped into World Cup eras **and** focused tournament-window models; local open LLM for labelling with an assessment of alternatives to Qwen3; first sentiment layer (VADER) included now; match/daily-thread posts included with their `post_type` flag; layout `code/` + `results/` + `.venv`; pipeline must also run on Google Colab GPUs.

## Plain-language summary of what will happen

1. **Turn every post into a number vector.** A sentence-embedding model reads each post's title (plus up to 1,000 characters of body) and produces a vector that places posts with similar meaning close together. Headlines about the same transfer saga, the same match, or the same controversy end up neighbours.
2. **Find the dense neighbourhoods.** BERTopic compresses the vectors (UMAP) and finds dense groups (HDBSCAN). Each group is a *topic*; posts that fit nowhere become "outliers" (reported, not hidden).
3. **Describe each topic by its distinctive words.** For every topic, BERTopic computes which words/bigrams are unusually frequent in it compared with the rest (c-TF-IDF). These keywords are the "recurring keyword tags" you asked for. We also keep 5–10 representative posts per topic.
4. **Name and characterise topics with a local AI model.** Exactly as in the template paper, two "analyst" instances of an open LLM each read a topic's keywords and sample posts, propose a name, a description, the communicative purpose, the emotional tone and inclusion cues, then revise towards each other for up to 3 rounds. You and co-authors then merge near-duplicate topics in an editable table.
5. **Measure tone and sentiment.** Each post gets a VADER sentiment score (built for social-media text); we summarise it per topic and per month, next to the LLM's tone label.
6. **Track topics over time.** For each model we plot monthly topic prevalence; year models are aligned within each World Cup era so the same theme can be followed from one year to the next; tournament-window models zoom in on the weeks around each World Cup.
7. **Report.** Every run writes tables (CSV) and an HTML/Markdown report with charts, plus everything needed to reproduce it (config, seeds, versions).

## 1. Environment

* Python 3.14.4 is installed and all required wheels exist for it (checked): `torch 2.14`, `bertopic 0.17.4`, `sentence-transformers 6.1`, `umap-learn 0.5.12`, `hdbscan 0.8.44` (compiles; clang present), `transformers`, `accelerate`, `vaderSentiment`, `plotly`, `pyarrow`, `pandas` (present). GPU on the Mac = Apple MPS (M4, 16 GB).
* Create `.venv/` at the repo root; `code/requirements-topicmodel.txt` with pinned versions so the same environment is rebuilt on Colab and co-authors' machines. `.venv` is excluded from the OneDrive-shared content by a `.gitignore`-style note in `docs/` (it is machine-specific; co-authors recreate it from the requirements file).

## 2. Code layout (new files)

| File | Role |
|---|---|
| `code/topicmodel/config.py` | One place for all settings: paths, text rules, window definitions, model names, hyper-parameters, seeds, device (`auto` → cuda / mps / cpu). Fully commented. |
| `code/topicmodel/data.py` | Load a yearly file (or several) with every column as string; build the document text; apply the window filter; keep `id, created_datetime, year, post_type, lang, author, link_flair_text, num_comments, score, domain`. |
| `code/topicmodel/embed.py` | Sentence-embedding with caching: `results/topic-models/embeddings/<model>/<year>.npy` + ids, so embeddings are computed once per year and reused by year, window and era runs (and survive Colab disconnects). |
| `code/topicmodel/model.py` | BERTopic fit/transform, outlier handling, topics-over-time, era alignment (`BERTopic.merge_models`), saving/loading models. |
| `code/topicmodel/label.py` | LLM labelling (two-analyst debate → JSON), merge-table generation and application. |
| `code/topicmodel/sentiment.py` | VADER per document; per-topic and per-month summaries. |
| `code/topicmodel/report.py` | CSV tables + HTML/Markdown report with Plotly charts (topic sizes, keywords, topics over time, sentiment by topic, intertopic map). |
| `code/05-topic-model.py` | CLI entry point: `--run year --year 2010`, `--run window --worldcup 2010`, `--run era --era 2010`, `--label`, `--device`. Writes everything to `results/topic-models/<run-name>/`. |
| `code/05b-topic-model-colab.ipynb` | Colab notebook: mount Drive, install requirements, run the same CLI on GPU, copy results back to Drive. |

All scripts carry the same documentation standard as the R scripts (header: purpose/why/how to run; section banners; line-level comments).

## 3. Methods and settings (for your review; all in `config.py`)

**Documents.** `title + "\n" + selftext[:1000]`, then light cleaning that does not alter meaning: strip Markdown link syntax and bare URLs, Reddit sprite codes (`[](#sprite…)`), collapse whitespace. No stemming, no lower-casing before embedding (the embedding model handles case). Minimum 3 word tokens, otherwise the post is kept in the tables but marked `too_short` and not modelled.

**Embeddings.** Default `all-MiniLM-L6-v2` for the local test (fast); on Colab GPU switch to `all-mpnet-base-v2` (higher quality) — both from sentence-transformers; the choice is a config value and the cache is keyed by model name, so both can be compared on 2010 if wanted.

**BERTopic.** UMAP (`n_neighbors=15, n_components=5, min_dist=0.0, metric=cosine, random_state=42`); HDBSCAN (`min_cluster_size = max(15, round(0.25% of documents))`, `min_samples=10`, `metric=euclidean`, `prediction_data=True`); `CountVectorizer(ngram_range=(1,2), stop_words=english + a small, documented Reddit/soccer boilerplate list such as "thread", "match thread", "post match", "amp", "http")`; c-TF-IDF with `reduce_frequent_words=True`; representation = `KeyBERTInspired` + `MaximalMarginalRelevance(diversity=0.3)` for readable, non-redundant keywords; `top_n_words=15`; `calculate_probabilities=False` (speed) with `reduce_outliers(strategy="c-tf-idf")` applied and the before/after outlier share reported. Topic count is data-driven (no fixed k); if a run yields > 60 topics we reduce with `nr_topics="auto"` and keep the unreduced version too.

**Topics over time.** `topics_over_time` at monthly resolution for year/era runs, weekly for tournament windows.

**Era alignment.** Year models within an era (2010–13, 2014–17, 2018–21, 2022–25, 2026) are merged with `BERTopic.merge_models(min_similarity=0.85)` so a theme keeps one id across the years; the era report shows each aligned topic's share per year.

**Tournament windows (confirmed: ± 2 months).** From 2 months before the opening match to 2 months after the final: 2010 (11 Jun–11 Jul → **11 Apr–11 Sep 2010**), 2014 (12 Jun–13 Jul → 12 Apr–13 Sep), 2018 (14 Jun–15 Jul → 14 Apr–15 Sep), 2022 (20 Nov–18 Dec → 20 Sep 2022–18 Feb 2023, spanning two yearly files), 2026 (11 Jun–19 Jul → 11 Apr–19 Sep; data ends 31 Jul, so the post-final tail is 12 days — noted in the report). Test case window: 11 Apr–11 Sep 2010 (≈ 9,700 posts).

**LLM labelling — model assessment.** Candidates that run free and locally on a 16 GB Mac and on Colab: Qwen3-1.7B / Qwen3-4B (Apache-2.0), Gemma 3 4B (Gemma licence, restrictions on use), Llama 3.2 3B (Llama community licence), Phi-4-mini 3.8B (MIT). Recommendation: **Qwen3-4B** — same family as the template paper (comparability), permissive Apache-2.0 licence (important for a published academic pipeline), strong instruction-following and JSON output in its size class, and it fits on this Mac in bfloat16 (~8 GB) and on any Colab GPU. Qwen3-1.7B stays available as the fast fallback (`config`). I cannot benchmark label quality here without running them, so the plan includes an **optional bake-off** in the test case: label the same 2010 topics with Qwen3-4B and Phi-4-mini and let co-authors judge which names/tones read better; the pipeline is model-agnostic (any Hugging Face chat model id).
Prompting follows the paper: two analysts (`A`, `B`) each receive topic keywords + up to 40 sample posts and must return JSON `{name, purpose, emotion_tone, description, inclusion_cues}`; up to 3 revision turns "revise to improve agreement with the peer while preserving quality"; if still different, both versions are kept and flagged for manual choice. Thinking mode is disabled for speed; temperature 0.2; outputs validated against a JSON schema and retried once on failure.

**Manual merge.** `results/…/topic_labels.csv` has an editable `merge_into` column; re-running `--apply-merges` collapses topics and recomputes all tables, so co-author decisions are recorded in a file, not in code.

**Sentiment.** VADER compound score per post (title + body text); per topic: mean, share positive (> 0.05) / negative (< −0.05); per month per topic. Clearly flagged in the report as a lexicon baseline to be replaced/validated later (e.g. a transformer model) once the topic approach is agreed.

## 4. Outputs per run (`results/topic-models/<run-name>/`)

* `documents.csv` — id, date, post_type, lang, topic id, topic probability/outlier flag, sentiment score (joinable to the corpus by `id`; text not duplicated).
* `topics.csv` — topic id, size, share, top-15 keywords, 5 representative post ids + titles, LLM name/description/purpose/tone/cues (both analysts + final), merge target.
* `topics_over_time.csv`, `sentiment_by_topic.csv`, `sentiment_by_month.csv`.
* `report.html` (+ `report.md`): run summary (documents, outlier share, parameters, versions), topic table, charts (topic sizes; keywords per topic; topics over time; sentiment by topic; intertopic distance map; for windows, a timeline with kickoff/final markers).
* `model/` (saved BERTopic model, safetensors), `config.json`, `log.txt`.

## 5. Colab integration

* Same code, same CLI; `--device cuda`. Notebook steps: mount Google Drive → `pip install -r requirements` → copy `code/topicmodel/` and the needed yearly CSVs from a Drive folder (you upload `data/yearly datasets/` once, ~2.4 GB) → run per-year jobs → results land in Drive → you copy them back into `results/` in the repo.
* Robustness to disconnects: embeddings and fitted models are checkpointed per year; a job restarts from cache. Free Colab (T4, ~12 h sessions) is enough per year; the full corpus is ~17 year jobs + 5 windows + 5 era merges — Colab Pro (A100/L4) makes this a few hours total; LLM labelling runs on the same GPU.
* The Google Drive connector in this session is not authorised, so uploads/downloads are manual steps in the notebook; nothing in the pipeline depends on it.

## 6. Test case today (≈ 1 h) and next session (2–3 h)

Today:
1. Create `.venv`, install requirements (≈10 min incl. hdbscan compile), download `all-MiniLM-L6-v2` and `Qwen3-4B` weights (≈8 GB).
2. Run `year-2010` (13,638 posts) and `window-wc2010` (≈7,000 posts) locally with MiniLM; produce reports without LLM labels (minutes).
3. Start LLM labelling for both runs (estimate 20–40 topics × 2 analysts × ≤3 turns on MPS: 20–45 min) — runs in the background while docs are written.
4. Write `docs/session-02_2026-10-13/02_topic-modelling-design.md` (this plan in prose, with the plain-language explanation, parameters, and what to look at when reviewing) and update `memory/`.

Next session: review outputs with your co-author feedback; tune (min cluster size, stopwords, merge table, model choice); optional MiniLM-vs-mpnet and Qwen-vs-Phi bake-offs on 2010; build and test the Colab notebook on one year; then schedule the full run.

## 7. Verification

* Pipeline sanity: document count in `documents.csv` equals the number of input posts; every modelled document has a topic or is marked outlier; outlier share reported (target < 30 % after reduction); seeds fixed → re-running `year-2010` reproduces identical topic sizes.
* Label sanity: every topic has valid JSON from both analysts; names unique; tone from a fixed vocabulary (the paper's free-text tones are kept but also mapped to a small controlled set for counting).
* Sentiment sanity: scores in [−1, 1]; spot-check 20 strongly positive/negative posts.
* Human check on the 2010 window: the tournament should surface recognisable themes (e.g. vuvuzelas, Jabulani ball, Lampard's disallowed goal / goal-line technology, Suárez handball, Spain's win); if it does not, that is a signal to revisit settings before co-author review.

## 8. Confirmed with you (2026-10-13)
* Window definition: ± 2 months around each tournament; era grouping as stated.
* LLM choice: Qwen3-4B, with the optional Phi-4-mini comparison on the test case.
* Hyper-parameters and the boilerplate stop-word list as proposed in §3 (explained in plain language during review; all editable in `config.py` and recorded in each run's `config.json`).
