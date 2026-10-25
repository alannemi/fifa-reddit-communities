"""
Loading the cleaned corpus and building documents (design §2).
==============================================================
* The yearly files (config.YEARLY_DIR / config.YEARLY_FILE) are read with every column as a string — the convention
  of the R scripts and the codebook (§5): no type coercion, empty cell = missing.
* One post = one document.  Document text = cleaned title + newline + cleaned body (body truncated to
  BODY_MAX_CHARS).  Cleaning removes only presentation noise (Markdown link syntax, bare URLs, Reddit sprite codes,
  HTML entities, whitespace); it never changes words, case or spelling.
* Documents with fewer than MIN_TOKENS word tokens are kept in every table but flagged `too_short` and not modelled.
"""
from __future__ import annotations
import html
import re
from pathlib import Path
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


def n_tokens(s: str) -> int:
    """Number of word tokens (two or more letters) in a cleaned text — the `too_short` rule counts these."""
    return len(_TOKEN.findall(s))


def yearly_file(year: int) -> Path:
    """Path of one cleaned yearly file."""
    return C.YEARLY_DIR / C.YEARLY_FILE.format(year=year)


def available_years() -> list[int]:
    """The calendar years that have a yearly file under config.YEARLY_DIR, sorted."""
    pattern = C.YEARLY_FILE.format(year="*")
    years = sorted(int(p.stem.rsplit("-", 1)[1]) for p in C.YEARLY_DIR.glob(pattern))
    if not years:
        raise SystemExit(f"no yearly files matching {pattern} under {C.YEARLY_DIR} "
                         "(check PIPELINE_ROOT / PIPELINE_YEARLY_DIR, or copy data/ from the shared folder)")
    return years


def load_year(year: int) -> pd.DataFrame:
    return pd.read_csv(yearly_file(year), dtype=str, keep_default_na=False, encoding="utf-8")


def build_documents(df: pd.DataFrame) -> pd.DataFrame:
    """Add `document`, `n_tokens`, `too_short` and `date` columns; keep the metadata columns of COLUMNS_KEPT."""
    title = df["title"].map(clean_text)
    body = df["selftext"].str.slice(0, C.BODY_MAX_CHARS).map(clean_text)
    doc = (title + "\n" + body).str.strip()
    out = df[C.COLUMNS_KEPT].copy()
    out["title"] = df["title"]
    out["document"] = doc
    out["n_tokens"] = doc.map(n_tokens)
    out["too_short"] = out["n_tokens"] < C.MIN_TOKENS
    out["date"] = pd.to_datetime(df["created_datetime"], format="%Y-%m-%d %H:%M:%S", utc=True)
    return out
