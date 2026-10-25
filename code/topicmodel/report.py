"""
The per-bin report (report.html + report.md) — presentation only; nothing here changes the data.
================================================================================================
Rendered after every stage that changes a bin's tables (topics, reassign, label, taxonomy; `--stage report`
re-renders): topic sizes, prevalence over time with the kick-off / final markers of the tournament bins, the mean
VADER sentiment of the titles by topic, and the topic table with the agreed names, tones, descriptions and the
category of each topic in the bin's taxonomy.  Plotly is loaded from a CDN, so the file is small and self-contained.
"""
from __future__ import annotations
import datetime as dt
import json
import logging
import pandas as pd
from . import config as C

log = logging.getLogger("topicmodel")

LABEL_COLS = ["name", "emotion_tone", "description", "needs_review", "category_id", "category_name"]


def _fig_html(fig) -> str:
    return fig.to_html(full_html=False, include_plotlyjs=False)


def write_report(run_dir, meta: dict, topics: pd.DataFrame, tot: pd.DataFrame, sent_topic: pd.DataFrame,
                 markers: dict | None = None) -> None:
    """`topics` = the columns of topics.csv (topic_id, n_cluster, n_total, share, keywords, representative_titles, …)
    plus, when the stages have run, the label columns (name, emotion_tone, description, needs_review) and the
    category columns (category_id, category_name); `tot` = topics_over_time.csv; `sent_topic` = topic_id, n_docs,
    mean_compound; `markers` = {name: date} vertical lines for the prevalence chart."""
    import plotly.express as px
    t = topics[topics["topic_id"] != -1].copy()
    has_names = "name" in t.columns

    def label(r):
        name = str(r["name"]).strip() if has_names else ""
        return f"{r['topic_id']}: " + (name if name else ", ".join(str(r["keywords"]).split(", ")[:4]))

    t["label"] = t.apply(label, axis=1)
    parts = []
    # 1. topic sizes (posts after the outlier reassignment, if a threshold is set)
    fig = px.bar(t.sort_values("n_total", ascending=True), x="n_total", y="label", orientation="h",
                 title="Topic sizes (number of posts)", height=max(400, 22 * len(t)))
    parts.append(_fig_html(fig))
    # 2. prevalence over time (12 largest topics)
    top = t.nlargest(12, "n_total")["topic_id"].tolist()
    tt = tot[tot["topic_id"].isin(top)].merge(t[["topic_id", "label"]], on="topic_id")
    fig = px.line(tt, x="period", y="n_docs", color="label", title="Topic prevalence over time (12 largest topics)")
    if markers:
        # A shape + an annotation, not fig.add_vline(): with a datetime position and an annotation, add_vline fails on
        # plotly 5.x ("unsupported operand type(s) for +: 'int' and 'datetime.datetime'"; Colab ships 5.24.1, first Colab
        # run 2026-10-25). Date axes accept ISO strings in every plotly version.
        for name, day in markers.items():
            x = pd.Timestamp(day).strftime("%Y-%m-%d")
            fig.add_shape(type="line", x0=x, x1=x, y0=0, y1=1, yref="paper", line={"dash": "dash", "color": "#52606d"})
            fig.add_annotation(x=x, y=1, yref="paper", text=name, showarrow=False, yanchor="bottom", font={"size": 11})
    parts.append(_fig_html(fig))
    # 3. sentiment by topic
    st = sent_topic[sent_topic["topic_id"] != -1].merge(t[["topic_id", "label"]], on="topic_id")
    fig = px.bar(st.sort_values("mean_compound"), x="mean_compound", y="label", orientation="h",
                 title="Mean VADER sentiment of the titles by topic", height=max(400, 22 * len(st)))
    parts.append(_fig_html(fig))
    # the topic table
    cols = ["topic_id", "n_cluster", "n_total", "share", "keywords"] + [c for c in LABEL_COLS if c in t.columns] + ["representative_titles"]
    cols = [c for c in cols if c in t.columns]
    table_html = t[cols].to_html(index=False, escape=True, classes="topics")
    summary = {k: v for k, v in meta.items() if k != "config"}
    html = f"""<!doctype html><html><head><meta charset="utf-8"><title>{meta['run_name']} — period bin report</title>
<script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
<style>body{{font-family:Helvetica,Arial,sans-serif;max-width:1200px;margin:20px auto;color:#1f2933}}
table.topics{{border-collapse:collapse;font-size:12px}} table.topics td,table.topics th{{border:1px solid #d9e2ec;padding:4px 6px;vertical-align:top}}
th{{background:#e8eef5}} pre{{background:#f0f4f8;padding:8px;font-size:12px}}</style></head><body>
<h1>Period bin report — {meta['run_name']}</h1>
<p>Generated {dt.datetime.now():%Y-%m-%d %H:%M}. One document = one r/soccer post (title + up to {C.BODY_MAX_CHARS} characters of body); one BERTopic model per period bin.</p>
<h2>Run summary</h2><pre>{json.dumps(summary, indent=1, default=str)}</pre>
<p><b>How to read this.</b> Topics are dense groups of posts with similar meaning found by BERTopic; each is described by its most distinctive words (c-TF-IDF keywords) and, once the labelling stage has run, by the name, tone and description the two analysts agreed on, and by the category of the bin's discourse taxonomy. Posts that fit no topic are topic −1 (outliers); above the calibrated similarity threshold they are reassigned to their nearest topic (<code>n_cluster</code> = cluster members, <code>n_total</code> = after reassignment). Sentiment is the VADER compound score of the title in [−1, 1], a lexicon measure (design F10).</p>
<h2>Charts</h2>{''.join(parts)}
<h2>Topics</h2>{table_html}
</body></html>"""
    (run_dir / "report.html").write_text(html, encoding="utf-8")
    # Markdown summary
    md = [f"# Period bin report — {meta['run_name']}", "", "## Run summary", "```", json.dumps(summary, indent=1, default=str), "```", "",
          "## Topics", "", "| topic_id | n_total | share | keywords | " + (" name | tone | category | " if has_names else "") + "representative titles |",
          "|---|---|---|---|" + ("---|---|---|" if has_names else "") + "---|"]
    for _, r in t.sort_values("n_total", ascending=False).iterrows():
        extra = f" {r.get('name', '')} | {r.get('emotion_tone', '')} | {r.get('category_id', '')} |" if has_names else ""
        md.append(f"| {r['topic_id']} | {r['n_total']} | {float(r['share']):.1%} | {r['keywords']} |{extra} {str(r['representative_titles']).replace('|', '/')} |")
    (run_dir / "report.md").write_text("\n".join(md), encoding="utf-8")
    log.info("report written: %s", run_dir / "report.html")
