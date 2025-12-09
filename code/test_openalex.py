# openalex_buzzwords.py
"""
Skeleton pipeline for 'Temporal Trends of Tech Buzzwords' using OpenAlex.

Flow (per spec):
1) Query OpenAlex 'works' for each keyword across years (2000–2025).
2) Save raw results per keyword (JSONL) for reproducibility.
3) Clean & normalize terms (map synonyms like "deep learning" ~ "deep neural networks").
4) Aggregate yearly counts to build time series.
5) (Optional) Tag keywords (method/technology/concept) and cluster into themes.

Run:
    python openalex_buzzwords.py

Notes:
- OpenAlex uses cursor-based pagination; max per_page=200.
- Respect polite usage: include your contact via `mailto` param.
- This is a skeleton—fill TODOs where indicated.
"""

from __future__ import annotations
import os
import time
import json
import math
import pathlib
import datetime as dt
from typing import Dict, Iterable, List, Optional

import requests
import pandas as pd


# =========================
# Config
# =========================

MAILTO = os.environ.get("OPENALEX_MAILTO", "your_email@example.com")  # TODO: set or export env var
BASE_URL = "https://api.openalex.org/works"

# Project scope (you can tweak these at runtime)
YEAR_START = 2015
YEAR_END   = 2025

# Your picked buzzwords
KEYWORDS = [
    "big data",
    "quantum computing",
    "artificial general intelligence",
    "blockchain",
    "cryptocurrency",
    "metaverse",
    "edge computing",
    "generative ai",
    "chatbot",
    "augmented reality",
]

# Map synonyms -> canonical buzzword (normalization)
TERM_ALIASES: Dict[str, str] = {
    # Big data
    "big data": "big data",
    "large scale data": "big data",

    # Quantum computing
    "quantum computing": "quantum computing",
    "quantum computer": "quantum computing",

    # Artificial General Intelligence
    "artificial general intelligence": "artificial general intelligence",
    "agi": "artificial general intelligence",

    # Blockchain / crypto
    "blockchain": "blockchain",
    "cryptocurrency": "cryptocurrency",
    "crypto": "cryptocurrency",
    "cryptocurrencies": "cryptocurrency",

    # Metaverse
    "metaverse": "metaverse",

    # Edge computing
    "edge computing": "edge computing",
    "fog computing": "edge computing",

    # Generative AI
    "generative ai": "generative ai",
    "generative models": "generative ai",
    "gpt": "generative ai",
    "stable diffusion": "generative ai",

    # Chatbots
    "chatbot": "chatbot",
    "chat bot": "chatbot",
    "conversational agent": "chatbot",

    # Augmented reality
    "augmented reality": "augmented reality",
    "ar": "augmented reality",
}

# Output dirs
OUTDIR = pathlib.Path("data_openalex")
RAW_DIR = OUTDIR / "raw"
PROCESSED_DIR = OUTDIR / "processed"
TIMESERIES_DIR = OUTDIR / "timeseries"
ANNOTATION_DIR = OUTDIR / "annotation"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)

# =========================
# OpenAlex client
# =========================

class OpenAlexClient:
    """Minimal OpenAlex client for 'works' endpoint with cursor pagination."""

    def __init__(self, base_url: str = BASE_URL, mailto: str = MAILTO, rate_sleep: float = 0.2):
        self.base_url = base_url
        self.mailto = mailto
        self.rate_sleep = rate_sleep  # polite throttle between requests

    def _request(self, params: Dict) -> Dict:
        # Simple GET with retry/backoff
        backoff = 1.0
        for attempt in range(5):
            try:
                resp = requests.get(self.base_url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                # Handle 429/5xx with backoff
                if resp.status_code in (429, 500, 502, 503, 504):
                    print(resp.json())
                    time.sleep(backoff)
                    # backoff *= 2
                else:
                    resp.raise_for_status()
            except requests.RequestException as e:
                # Network hiccup—backoff and retry
                print(resp.json())
                time.sleep(backoff)
                # backoff *= 2
        raise RuntimeError(f"OpenAlex request failed after retries: params={params}")


    def iter_works(
        self,
        query: str,
        from_year: int,
        to_year: int,
        select_fields: Optional[str] = None,
        filters: Optional[Dict[str, str]] = None,
        max_records: int | None = None,
    ) -> Iterable[Dict]:
        cursor = "*"
        yielded = 0

        # Build filter string: date range + any extras
        filt = {
            # option A (publication year range — lighter & simple):
            "publication_year": f"{from_year}-{to_year}",
            # option B (exact dates — uncomment if you prefer day precision):
            # "from_publication_date": f"{from_year}-01-01",
            # "to_publication_date": f"{to_year}-12-31",
        }
        if filters:
            filt.update(filters)

        params = {
            "search": query,                 # or use 'q' if you prefer
            "per_page": 200,                 # 'per-page' also works
            "cursor": cursor,
            "mailto": self.mailto,
            "filter": ",".join(f"{k}:{v}" for k, v in filt.items()),
        }
        if select_fields:
            params["select"] = select_fields

        while True:
            data = self._request(params)
            for item in data.get("results", []):
                yield item
                yielded += 1
                if max_records is not None and yielded >= max_records:
                    return  # stop early
            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor:
                break
            params["cursor"] = next_cursor
            time.sleep(self.rate_sleep)


# =========================
# Normalization helpers
# =========================

def canonical_term(term: str) -> str:
    """Map a term to its canonical buzzword."""
    t = term.strip().lower()
    return TERM_ALIASES.get(t, t)

def contains_term(text: str, term_variants: List[str]) -> bool:
    """Very simple containment check; improve with NLP later if needed."""
    if not text:
        return False
    low = text.lower()
    return any(v in low for v in term_variants)


# =========================
# Data collection
# =========================

def fetch_keyword(
    client: OpenAlexClient,
    keyword: str,
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
    select_fields: Optional[str] = "id,display_name,title,publication_year,publication_date,abstract_inverted_index,primary_location,cited_by_count,related_works",
    extra_filters: Optional[Dict[str, str]] = None,
    save_every_n: int = 1000,
    max_records: int | None = None,
) -> pathlib.Path:
    """
    Fetch all works for a keyword and save JSONL into data_openalex/raw/keyword.jsonl
    """
    canonical = canonical_term(keyword)
    raw_path = RAW_DIR / f"{canonical.replace(' ', '_')}.jsonl"
    count = 0

    # Build a simple variant list (canonical + alias keys pointing to canonical)
    variants = {k for k, v in TERM_ALIASES.items() if v == canonical}
    variants.add(canonical)

    with raw_path.open("w", encoding="utf-8") as f:
        for work in client.iter_works(
            query=canonical,  # start with canonical in OpenAlex 'search'
            from_year=year_start,
            to_year=year_end,
            select_fields=select_fields,
            filters=extra_filters,
            max_records=max_records,
        ):
            # Optional local filter: keep if any variant appears in title or (expanded) abstract
            title = (work.get("title") or "")  # OpenAlex field is 'title'
            abstract_inv = work.get("abstract_inverted_index") or {}
            # Reconstruct abstract text if provided as inverted index
            abstract_text = invert_openalex_abstract(abstract_inv)

            if contains_term(title, list(variants)) or contains_term(abstract_text, list(variants)):
                f.write(json.dumps(work, ensure_ascii=False) + "\n")
                count += 1
                if count % save_every_n == 0:
                    print(f"[{canonical}] saved {count} records...")

    print(f"[{canonical}] DONE. Saved {count} records -> {raw_path}")
    return raw_path


def invert_openalex_abstract(inv_idx: Dict[str, List[int]]) -> str:
    """Convert OpenAlex abstract_inverted_index to plain text."""
    if not inv_idx:
        return ""
    # Find max position
    max_pos = 0
    for positions in inv_idx.values():
        if positions:
            max_pos = max(max_pos, max(positions))
    words = [""] * (max_pos + 1)
    for word, positions in inv_idx.items():
        for p in positions:
            if 0 <= p < len(words):
                words[p] = word
    return " ".join(w for w in words if w)


# =========================
# Processing & aggregation
# =========================

def load_jsonl(path: pathlib.Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def deduplicate_works(records: List[Dict]) -> pd.DataFrame:
    """
    Deduplicate by 'id' and keep minimal fields for aggregation.
    """
    df = pd.DataFrame.from_records(records)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    # Keep only what we need for time series
    keep_cols = [c for c in ["id", "title", "publication_year", "cited_by_count"] if c in df.columns]
    return df[keep_cols]

def sample_titles_for_annotation(df: pd.DataFrame, keyword: str, n: int = 10) -> pd.DataFrame:
    """
    Sample up to n titles for a given keyword for use in Label Studio.

    Returns a small DataFrame with columns:
        id, keyword, title, publication_year
    """
    if "title" not in df.columns:
        return pd.DataFrame()

    # Drop rows without titles
    df = df.dropna(subset=["title"])
    if df.empty:
        return pd.DataFrame()

    sample = df.sample(min(n, len(df)), random_state=42).copy()
    sample["keyword"] = canonical_term(keyword)
    # Keep only the fields we care about for annotation
    keep_cols = [c for c in ["id", "keyword", "title", "publication_year"] if c in sample.columns]
    return sample[keep_cols]


def aggregate_yearly_counts(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    df = df.copy()
    df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce").astype("Int64")
    ts = (
        df.dropna(subset=["publication_year"])
          .groupby("publication_year")
          .size()
          .rename("count")
          .reset_index()
    )
    ts["keyword"] = canonical_term(keyword)
    # Fill missing years with 0 (complete series)
    full = pd.DataFrame({"publication_year": range(YEAR_START, YEAR_END + 1)})
    ts = full.merge(ts, on="publication_year", how="left").fillna({"count": 0})
    ts["count"] = ts["count"].astype(int)
    return ts[["keyword", "publication_year", "count"]]

def normalize_timeseries(ts: pd.DataFrame) -> pd.DataFrame:
    max_count = ts["count"].max()
    if max_count > 0:
        ts["normalized_count"] = ts["count"] / max_count
    else:
        ts["normalized_count"] = 0.0
    return ts

def save_timeseries(ts: pd.DataFrame, keyword: str) -> pathlib.Path:
    out = TIMESERIES_DIR / f"{canonical_term(keyword).replace(' ', '_')}_timeseries.csv"
    ts.to_csv(out, index=False)
    print(f"[{keyword}] time series -> {out}")
    return out


# =========================
# Orchestration
# =========================

def run_pipeline(
    keywords: List[str] = KEYWORDS,
    year_start: int = YEAR_START,
    year_end: int = YEAR_END,
) -> None:
    client = OpenAlexClient()

    all_ts = []
    all_annotation_samples = []
    for kw in keywords:
        raw_path = fetch_keyword(client, kw, year_start, year_end, max_records=10000)

        # Process
        records = load_jsonl(raw_path)
        df = deduplicate_works(records)
        proc_path = PROCESSED_DIR / f"{canonical_term(kw).replace(' ', '_')}.parquet"
        df.to_parquet(proc_path, index=False)
        print(f"[{kw}] processed -> {proc_path}")

        # Sample titles for annotation
        anno_df = sample_titles_for_annotation(df, kw, n=10)
        if not anno_df.empty:
            all_annotation_samples.append(anno_df)

        # Aggregate yearly counts
        ts = aggregate_yearly_counts(df, kw)
        # Normalize counts
        ts = normalize_timeseries(ts)
        save_timeseries(ts, kw)
        all_ts.append(ts)

    # Combined CSV for quick plotting later
    combined = pd.concat(all_ts, ignore_index=True)
    combined_path = TIMESERIES_DIR / "all_keywords_timeseries.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[ALL] combined time series -> {combined_path}")
    
    # Combined annotation file for Label Studio
    if all_annotation_samples:
        anno_all = pd.concat(all_annotation_samples, ignore_index=True)
        # Label Studio likes a "text" field by default; rename title -> text
        anno_all = anno_all.rename(columns={"title": "text"})
        anno_out = ANNOTATION_DIR / "annotation_titles_labelstudio.csv"
        anno_out.parent.mkdir(parents=True, exist_ok=True)
        anno_all.to_csv(anno_out, index=False)
        print(f"[ALL] annotation titles -> {anno_out}")

if __name__ == "__main__":
    # Minimal smoke test: run end-to-end for the default KEYWORDS
    # TODO: add argparse to pass custom keywords or years from CLI.
    run_pipeline()
