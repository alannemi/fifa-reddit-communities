"""
Loading the cleaned corpus and building documents.
===================================================
* Yearly files are read with every column as a string (same convention as the
  R scripts: no type coercion, empty cell = missing).
* One post = one document.  Document text = title + newline + body (body
  truncated to BODY_MAX_CHARS).  Cleaning removes only presentation noise
  (Markdown link syntax, bare URLs, Reddit sprite codes, HTML entities,
  whitespace); it never changes words.
* Documents with fewer than MIN_TOKENS word tokens are kept in the tables but
  flagged `too_short` and excluded from modelling.
"""
from __future__ import annotations
import datetime as dt
import html
import re
import pandas as pd
from . import config as C

_MD_LINK = re.compile(r"\[([^\]]*)\]\((?:[^)]*)\)")          # [text](url) -> text
_SPRITE = re.compile(r"\[\]\(#[^)]*\)")                         # [](#sprite1-p17) etc.
_URL = re.compile(r"https?://\S+|www\.\S+")
_WS = re.compile(r"\s+")
_TOKEN = re.compile(r"[^\W\d_]{2,}", re.UNICODE)              # word tokens of >= 2 letters


def clean_text(s: str) -> str:
    """Remove presentation noise without altering wording."""
    s = html.unescape(s)
    s = _SPRITE.sub(" ", s)
    s = _MD_LINK.sub(r"\1", s)
    s = _URL.sub(" ", s)
    s = s.replace("&#x200B;", " ").replace("​", " ")
    s = _WS.sub(" ", s).strip()
    return s


def load_year(year: int) -> pd.DataFrame:
    path = C.YEARLY_DIR / f"soccer.data.cleaned-{year}.csv"
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    return df


def load_years(years) -> pd.DataFrame:
    return pd.concat([load_year(y) for y in years], ignore_index=True)


def build_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Add `document`, `n_tokens`, `too_short` and `date` columns; keep metadata columns."""
    title = df["title"].map(clean_text)
    body = df["selftext"].str.slice(0, C.BODY_MAX_CHARS).map(clean_text)
    doc = (title + "\n" + body).str.strip()
    out = df[C.COLUMNS_KEPT].copy()
    out["title"] = df["title"]
    out["document"] = doc
    out["n_tokens"] = doc.map(lambda s: len(_TOKEN.findall(s)))
    out["too_short"] = out["n_tokens"] < C.MIN_TOKENS
    out["date"] = pd.to_datetime(df["created_datetime"], format="%Y-%m-%d %H:%M:%S", utc=True)
    return out


def select_window(docs: pd.DataFrame, start: dt.date, end: dt.date) -> pd.DataFrame:
    """Keep documents with start <= date <= end (inclusive, UTC calendar days)."""
    d = docs["date"].dt.date
    return docs[(d >= start) & (d <= end)].reset_index(drop=True)
