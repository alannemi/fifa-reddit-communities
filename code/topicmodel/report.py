"""
Tables and report for one run.
==============================
Writes CSV tables (joinable to the corpus by post id) and a self-contained
HTML report (Plotly charts; the Plotly library is loaded from a CDN) plus a
Markdown summary.  Nothing here changes the data; it only presents it.
"""
from __future__ import annotations
import datetime as dt
import json
import logging
import pandas as pd
from . import config as C

log = logging.getLogger("topicmodel")


def write_tables(run_dir, docs: pd.DataFrame, topics: pd.DataFrame, tot: pd.DataFrame,
                 sent_topic: pd.DataFrame, sent_period: pd.DataFrame) -> None:
    keep = ["id", "created_datetime", "year", "post_type", "lang", "link_flair_text", "num_comments", "score",
            "too_short", "topic", "sentiment_compound", "sentiment_label"]
    docs[[c for c in keep if c in docs.columns]].to_csv(run_dir / "documents.csv", index=False)
    topics.to_csv(run_dir / "topics.csv", index=False)
    tot.to_csv(run_dir / "topics_over_time.csv", index=False)
    sent_topic.to_csv(run_dir / "sentiment_by_topic.csv", index=False)
    sent_period.to_csv(run_dir / "sentiment_by_period.csv", index=False)


def _fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def write_report(run_dir, meta: dict, docs: pd.DataFrame, topics: pd.DataFrame, tot: pd.DataFrame,
                 sent_topic: pd.DataFrame, bt_model=None, markers: dict | None = None) -> None:
    import plotly.express as px
    import plotly.graph_objects as go
    t = topics[topics["topic"] != -1].copy()
    t["label"] = t.apply(lambda r: f"{r['topic']}: " + (r["label_name"] if "label_name" in t.columns and str(r.get("label_name", "")).strip() else ", ".join(r["keywords"].split(", ")[:4])), axis=1)
    parts = []
    # 1. topic sizes
    fig = px.bar(t.sort_values("size", ascending=True), x="size", y="label", orientation="h",
                 title="Topic sizes (number of posts)", height=max(400, 22 * len(t)))
    parts.append(_fig_html(fig))
    # 2. topics over time (top 12 by size)
    top = t.nlargest(12, "size")["topic"].tolist()
    tt = tot[tot["topic"].isin(top)].merge(t[["topic", "label"]], on="topic")
    fig = px.line(tt, x="period", y="n_docs", color="label", title="Topic prevalence over time (12 largest topics)")
    if markers:
        for name, day in markers.items():
            fig.add_vline(x=pd.Timestamp(day).to_pydatetime(), line_dash="dash", annotation_text=name)
    parts.append(_fig_html(fig))
    # 3. sentiment by topic
    st = sent_topic[sent_topic["topic"] != -1].merge(t[["topic", "label"]], on="topic")
    fig = px.bar(st.sort_values("mean_compound"), x="mean_compound", y="label", orientation="h",
                 title="Mean VADER sentiment by topic (lexicon baseline)", height=max(400, 22 * len(st)))
    parts.append(_fig_html(fig))
    # 4. intertopic distance map
    if bt_model is not None:
        try:
            parts.append(_fig_html(bt_model.visualize_topics()))
        except Exception as e:  # small models can fail the 2-D projection
            log.warning("intertopic map skipped: %s", e)
    # topic table
    cols = ["topic", "size", "share", "keywords"] + [c for c in ("label_name", "label_tone", "label_description", "needs_review") if c in t.columns] + ["representative_titles"]
    table_html = t[cols].to_html(index=False, escape=True, classes="topics")
    summary = {k: v for k, v in meta.items() if k != "config"}
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{meta['run_name']} — topic model report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:Helvetica,Arial,sans-serif;max-width:1200px;margin:20px auto;color:#1f2933}}
table.topics{{border-collapse:collapse;font-size:12px}} table.topics td,table.topics th{{border:1px solid #d9e2ec;padding:4px 6px;vertical-align:top}}
th{{background:#e8eef5}} pre{{background:#f0f4f8;padding:8px;font-size:12px}}</style></head><body>
<h1>Topic model report — {meta['run_name']}</h1>
<p>Generated {dt.datetime.now():%Y-%m-%d %H:%M}. One document = one r/soccer post (title + up to {C.BODY_MAX_CHARS} characters of body).</p>
<h2>Run summary</h2><pre>{json.dumps(summary, indent=1, default=str)}</pre>
<p><b>How to read this.</b> Topics are dense groups of posts with similar meaning found by BERTopic; each is described by its most distinctive words (c-TF-IDF keywords) and, when labelled, by a name, tone and description proposed by two LLM analysts. Posts that fit no topic are topic −1 (outliers). Sentiment is a VADER lexicon score in [−1, 1] and is a first baseline only.</p>
<h2>Charts</h2>{''.join(parts)}
<h2>Topics</h2>{table_html}
</body></html>"""
    (run_dir / "report.html").write_text(html, encoding="utf-8")
    # Markdown summary
    md = [f"# Topic model report — {meta['run_name']}", "", "## Run summary", "```", json.dumps(summary, indent=1, default=str), "```", "",
          "## Topics", "", "| topic | size | share | keywords | " + (" name | tone | " if "label_name" in t.columns else "") + "representative titles |",
          "|---|---|---|---|" + ("---|---|" if "label_name" in t.columns else "") + "---|"]
    for _, r in t.sort_values("size", ascending=False).iterrows():
        extra = f" {r.get('label_name','')} | {r.get('label_tone','')} |" if "label_name" in t.columns else ""
        md.append(f"| {r['topic']} | {r['size']} | {r['share']:.1%} | {r['keywords']} |{extra} {str(r['representative_titles']).replace('|','/')} |")
    (run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("report written: %s", run_dir / "report.html")
