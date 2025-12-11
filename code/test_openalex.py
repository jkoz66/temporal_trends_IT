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

MAILTO = os.environ.get("OPENALEX_MAILTO", "your_email@example.com")
BASE_URL = "https://api.openalex.org/works"

YEAR_START = 2015
YEAR_END   = 2025

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

TERM_ALIASES: Dict[str, str] = {
    "big data": "big data",
    "large scale data": "big data",

    "quantum computing": "quantum computing",
    "quantum computer": "quantum computing",

    "artificial general intelligence": "artificial general intelligence",
    "agi": "artificial general intelligence",

    "blockchain": "blockchain",
    "cryptocurrency": "cryptocurrency",
    "crypto": "cryptocurrency",
    "cryptocurrencies": "cryptocurrency",

    "metaverse": "metaverse",

    "edge computing": "edge computing",
    "fog computing": "edge computing",

    "generative ai": "generative ai",
    "generative models": "generative ai",
    "gpt": "generative ai",
    "stable diffusion": "generative ai",

    "chatbot": "chatbot",
    "chat bot": "chatbot",
    "conversational agent": "chatbot",

    "augmented reality": "augmented reality",
    "ar": "augmented reality",
}

BASE_DIR = pathlib.Path(__file__).resolve().parent.parent
OUTDIR = BASE_DIR / "data" / "data_openalex"

RAW_DIR = OUTDIR / "raw"
PROCESSED_DIR = OUTDIR / "processed"
TIMESERIES_DIR = OUTDIR / "timeseries"
ANNOTATION_DIR = OUTDIR / "annotation"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)
ANNOTATION_DIR.mkdir(parents=True, exist_ok=True)


class OpenAlexClient:
    def __init__(self, base_url: str = BASE_URL, mailto: str = MAILTO, rate_sleep: float = 0.2):
        self.base_url = base_url
        self.mailto = mailto
        self.rate_sleep = rate_sleep

    def _request(self, params: Dict) -> Dict:
        backoff = 1.0
        for attempt in range(5):
            try:
                resp = requests.get(self.base_url, params=params, timeout=30)
                if resp.status_code == 200:
                    return resp.json()
                if resp.status_code in (429, 500, 502, 503, 504):
                    print(resp.json())
                    time.sleep(backoff)
                else:
                    resp.raise_for_status()
            except requests.RequestException as e:
                print(resp.json())
                time.sleep(backoff)
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

        filt = {
            "from_publication_date": f"{from_year}-01-01",
            "to_publication_date": f"{to_year}-12-31",
        }
        if filters:
            filt.update(filters)

        params = {
            "search": query,
            "per_page": 200,
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
                    return

            next_cursor = data.get("meta", {}).get("next_cursor")
            if not next_cursor:
                break

            params["cursor"] = next_cursor
            time.sleep(self.rate_sleep)


# Normalization helpers
def canonical_term(term: str) -> str:
    t = term.strip().lower()
    return TERM_ALIASES.get(t, t)

def contains_term(text: str, term_variants: List[str]) -> bool:
    if not text:
        return False
    low = text.lower()
    return any(v in low for v in term_variants)


# Data collection
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
    canonical = canonical_term(keyword)
    raw_path = RAW_DIR / f"{canonical.replace(' ', '_')}.jsonl"
    count = 0

    variants = {k for k, v in TERM_ALIASES.items() if v == canonical}
    variants.add(canonical)

    with raw_path.open("w", encoding="utf-8") as f:
        for work in client.iter_works(
            query=canonical,
            from_year=year_start,
            to_year=year_end,
            select_fields=select_fields,
            filters=extra_filters,
            max_records=max_records,
        ):
           
            title = (work.get("title") or "")
            abstract_inv = work.get("abstract_inverted_index") or {}
            
            abstract_text = invert_openalex_abstract(abstract_inv)

            if contains_term(title, list(variants)) or contains_term(abstract_text, list(variants)):
                f.write(json.dumps(work, ensure_ascii=False) + "\n")
                count += 1
                if count % save_every_n == 0:
                    print(f"[{canonical}] saved {count} records...")

    print(f"[{canonical}] DONE. Saved {count} records -> {raw_path}")
    return raw_path


def invert_openalex_abstract(inv_idx: Dict[str, List[int]]) -> str:
    if not inv_idx:
        return ""
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


# Processing & aggregation
def load_jsonl(path: pathlib.Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f]

def deduplicate_works(records: List[Dict]) -> pd.DataFrame:
    df = pd.DataFrame.from_records(records)
    if "id" in df.columns:
        df = df.drop_duplicates(subset=["id"])
    keep_cols = [
        c for c in [
            "id",
            "title",
            "publication_year",
            "publication_date",
            "cited_by_count",
        ]
        if c in df.columns
    ]
    return df[keep_cols]


def sample_titles_for_annotation(df: pd.DataFrame, keyword: str, n: int = 10) -> pd.DataFrame:
    if "title" not in df.columns:
        return pd.DataFrame()

    df = df.dropna(subset=["title"])
    if df.empty:
        return pd.DataFrame()

    sample = df.sample(min(n, len(df)), random_state=42).copy()
    sample["keyword"] = canonical_term(keyword)
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
    full = pd.DataFrame({"publication_year": range(YEAR_START, YEAR_END + 1)})
    ts = full.merge(ts, on="publication_year", how="left").fillna({"count": 0})
    ts["count"] = ts["count"].astype(int)
    return ts[["keyword", "publication_year", "count"]]

def aggregate_monthly_counts(df: pd.DataFrame, keyword: str) -> pd.DataFrame:
    df = df.copy()

    if "publication_date" in df.columns and df["publication_date"].notna().any():
        df["pub_date"] = pd.to_datetime(df["publication_date"], errors="coerce")
    else:
        df["pub_date"] = pd.to_datetime(
            df["publication_year"].astype(str) + "-06-01",
            errors="coerce",
        )

    df = df.dropna(subset=["pub_date"])

    df["month"] = df["pub_date"].dt.to_period("M").dt.to_timestamp("M")

    ts = (
        df.groupby("month")
          .size()
          .rename("count")
          .reset_index()
    )
    ts["keyword"] = canonical_term(keyword)

    full_months = pd.date_range(
        start=f"{YEAR_START}-01-01",
        end=f"{YEAR_END}-12-31",
        freq="M",
    )
    full = pd.DataFrame({"month": full_months})
    ts = full.merge(ts, on="month", how="left").fillna({"count": 0})
    ts["count"] = ts["count"].astype(int)

    return ts[["keyword", "month", "count"]]


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

def save_monthly_timeseries(ts: pd.DataFrame, keyword: str) -> pathlib.Path:
    out = TIMESERIES_DIR / f"{canonical_term(keyword).replace(' ', '_')}_timeseries_monthly.csv"
    ts.to_csv(out, index=False)
    print(f"[{keyword}] monthly time series -> {out}")
    return out


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

        records = load_jsonl(raw_path)
        df = deduplicate_works(records)
        proc_path = PROCESSED_DIR / f"{canonical_term(kw).replace(' ', '_')}.parquet"
        df.to_parquet(proc_path, index=False)
        print(f"[{kw}] processed -> {proc_path}")

        anno_df = sample_titles_for_annotation(df, kw, n=10)
        if not anno_df.empty:
            all_annotation_samples.append(anno_df)

        ts = aggregate_yearly_counts(df, kw)
        ts = normalize_timeseries(ts)
        save_timeseries(ts, kw)
        all_ts.append(ts)

        ts_month = aggregate_monthly_counts(df, kw)
        save_monthly_timeseries(ts_month, kw)

    combined = pd.concat(all_ts, ignore_index=True)
    combined_path = TIMESERIES_DIR / "all_keywords_timeseries.csv"
    combined.to_csv(combined_path, index=False)
    print(f"[ALL] combined time series -> {combined_path}")
    
    if all_annotation_samples:
        anno_all = pd.concat(all_annotation_samples, ignore_index=True)
        anno_all = anno_all.rename(columns={"title": "text"})
        anno_out = ANNOTATION_DIR / "annotation_titles_labelstudio.csv"
        anno_out.parent.mkdir(parents=True, exist_ok=True)
        anno_all.to_csv(anno_out, index=False)
        print(f"[ALL] annotation titles -> {anno_out}")

if __name__ == "__main__":
    run_pipeline()