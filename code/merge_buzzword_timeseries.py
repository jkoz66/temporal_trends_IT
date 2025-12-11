#!/usr/bin/env python
"""
Merge academic (OpenAlex), Wikipedia, and Google Trends time series
into a single master file per buzzword-year.

Inputs (your current layout):

    data_openalex/timeseries/all_keywords_timeseries.csv
        - from OpenAlex pipeline
        - columns: keyword, year, count, normalized_count

    data_wiki/wiki_pageviews_yearly.csv
        - yearly Wikipedia pageviews per buzzword
        - we normalize this here to get wiki_norm (0–1 per buzzword)

    data_trends/data_trends_yearly_long.csv
        - yearly Google Trends interest per buzzword
        - columns: buzzword, year, interest (0–100)
        - we normalize this here to get gtrends_norm (0–1 per buzzword)

Output:

    data_openalex/timeseries/buzzword_timeseries_master.csv
        columns (at minimum):
            keyword
            year
            academic_count
            academic_norm
            wiki_norm
            gtrends_norm
"""

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd



# ---------------------------------------------------------------------
# Paths – adapted to your directory structure
# ---------------------------------------------------------------------

OPENALEX_TS = Path("data/data_openalex") / "timeseries" / "all_keywords_timeseries.csv"
WIKI_TS     = Path("data/data_wiki")     / "wiki_pageviews_yearly.csv"
GTRENDS_TS  = Path("data/data_trends")   / "data_trends_yearly_long.csv"

MASTER_OUT  = Path("data") / "buzzword_timeseries_master.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def _infer_value_column(df: pd.DataFrame, preferred_names: list[str]) -> str:
    """Pick the first existing column from preferred_names, or a numeric fallback."""
    for name in preferred_names:
        if name in df.columns:
            return name

    candidates = [
        c for c in df.columns
        if c not in {"keyword", "buzzword", "year", "source"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not candidates:
        raise ValueError(f"Could not infer value column from columns: {list(df.columns)}")
    return candidates[0]


def load_openalex_ts(path: Path) -> pd.DataFrame:
    """Load OpenAlex time series, standardize column names."""
    if not path.exists():
        raise SystemExit(f"OpenAlex timeseries file not found: {path}")

    df = pd.read_csv(path)

    # Normalize names: expect keyword, year, count, normalized_count
    rename_map = {}
    if "keyword" not in df.columns:
        for alt in ["term", "buzzword"]:
            if alt in df.columns:
                rename_map[alt] = "keyword"
                break
    if "year" not in df.columns:
        for alt in ["publication_year", "year_int"]:
            if alt in df.columns:
                rename_map[alt] = "year"
                break

    df = df.rename(columns=rename_map)

    # 🔹 Canonicalize here as well
    df["keyword"] = df["keyword"].str.strip().str.lower()

    required = {"keyword", "year"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"OpenAlex timeseries must have columns keyword & year (got {list(df.columns)})"
        )

    # academic count
    if "count" in df.columns:
        df = df.rename(columns={"count": "academic_count"})
    elif "academic_count" not in df.columns:
        df["academic_count"] = pd.NA

    # academic normalized
    if "normalized_count" in df.columns:
        df = df.rename(columns={"normalized_count": "academic_norm"})
    elif "academic_norm" not in df.columns:
        numeric_cols = [c for c in df.columns if c not in {"keyword", "year"}]
        if numeric_cols:
            df = df.rename(columns={numeric_cols[0]: "academic_norm"})
        else:
            raise ValueError("Could not find academic normalized column in OpenAlex TS.")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["academic_norm"] = pd.to_numeric(df["academic_norm"], errors="coerce")

    return df[["keyword", "year", "academic_count", "academic_norm"]]


def load_wiki_ts(path: Path) -> Optional[pd.DataFrame]:
    """Load Wikipedia pageviews, keep wiki_raw and compute wiki_norm per keyword."""
    if not path.exists():
        print(f"[WARN] Wikipedia timeseries file not found, skipping: {path}")
        return None

    df = pd.read_csv(path)

    # 1) Standardize name of keyword column
    if "keyword" not in df.columns and "buzzword" in df.columns:
        df = df.rename(columns={"buzzword": "keyword"})

    # 🔹 Make wiki keywords canonical: strip + lowercase
    df["keyword"] = df["keyword"].str.strip().str.lower()

    # 2) Make sure we have a clean integer year column
    #    If your file already has 'year' as 2015, 2016, ... this is enough.
    #    If it has a date column instead (like '2015-01-01'), change this to slice:
    #    df["year"] = df["date"].astype(str).str[:4]
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Drop rows where year couldn't be parsed
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype("Int64")

    # 3) Make sure we have raw pageviews
    #    Adjust 'pageviews' to the actual column name in wiki_pageviews_yearly.csv
    df["wiki_raw"] = pd.to_numeric(df["views"], errors="coerce")

    # 4) Normalize per keyword -> wiki_norm in [0, 1]
    def _norm(x: pd.Series) -> pd.Series:
        max_val = x.max()
        if pd.isna(max_val) or max_val == 0:
            return 0.0
        return x / max_val

    df["wiki_norm"] = df.groupby("keyword")["wiki_raw"].transform(_norm)

    return df[["keyword", "year", "wiki_raw", "wiki_norm"]]


def load_gtrends_ts(path: Path) -> Optional[pd.DataFrame]:
    """Load Google Trends yearly interest, normalize to gtrends_norm 0–1 per keyword."""
    if not path.exists():
        print(f"[WARN] Google Trends timeseries file not found, skipping: {path}")
        return None

    df = pd.read_csv(path)

    # Your file: buzzword, year, interest (0–100)
    rename_map = {}
    if "keyword" not in df.columns and "buzzword" in df.columns:
        rename_map["buzzword"] = "keyword"

    df = df.rename(columns=rename_map)

    if not {"keyword", "year", "interest"}.issubset(df.columns):
        raise ValueError(
            f"Expected columns keyword/buzzword, year, interest in Trends TS (got {list(df.columns)})"
        )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["interest"] = pd.to_numeric(df["interest"], errors="coerce")

    # Normalize 0–1 per keyword (in case it's not exactly 0–100 or to be safe)
    def _norm(x: pd.Series) -> pd.Series:
        max_val = x.max()
        if max_val in (0, None) or pd.isna(max_val):
            return 0.0
        return x / max_val

    df["gtrends_norm"] = df.groupby("keyword")["interest"].transform(_norm)

    return df[["keyword", "year", "gtrends_norm"]]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] OpenAlex TS: {OPENALEX_TS}")
    df_ac = load_openalex_ts(OPENALEX_TS)
    print(f"[INFO] OpenAlex rows: {len(df_ac)}")

    print(f"[LOAD] Wikipedia TS: {WIKI_TS}")
    df_wiki = load_wiki_ts(WIKI_TS)

    print(f"[LOAD] Google Trends TS: {GTRENDS_TS}")
    df_gt = load_gtrends_ts(GTRENDS_TS)

    # Start from academic TS as backbone
    df = df_ac.copy()

    if df_wiki is not None:
        df = df.merge(df_wiki, on=["keyword", "year"], how="left")
        print(f"[MERGE] + Wikipedia -> rows: {len(df)}")

    if df_gt is not None:
        df = df.merge(df_gt, on=["keyword", "year"], how="left")
        print(f"[MERGE] + Google Trends -> rows: {len(df)}")

    # ----- Combined public interest index -----
    df["public_norm"] = (df["wiki_norm"].fillna(0) + df["gtrends_norm"].fillna(0)) / 2

    df = df.sort_values(["keyword", "year"]).reset_index(drop=True)

    df.to_csv(MASTER_OUT, index=False)
    print(f"[SAVE] master buzzword timeseries -> {MASTER_OUT}")

    print("\n[HEAD] master timeseries:")
    print(df.head())


if __name__ == "__main__":
    main()
