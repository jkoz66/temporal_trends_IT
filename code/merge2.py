#!/usr/bin/env python
"""
Merge academic (OpenAlex) and Wikipedia time series
for 2015–2025 into a single master file per buzzword-year.

OpenAlex:
    data_openalex/timeseries/all_keywords_timeseries.csv
    expected columns include:
        keyword, publication_year, count, normalized_count

Wikipedia:
    data_wiki/wiki_pageviews_yearly.csv
    expected columns include:
        keyword (or term), year (or year_int), and some views column

We:
    - restrict both to 2015–2025
    - min–max normalize per keyword within 2015–2025:
        academic_raw -> academic_norm
        wiki_raw -> wiki_norm
"""

from pathlib import Path
from typing import List, Optional

import pandas as pd

YEAR_START = 2015
YEAR_END = 2025

OUTDIR = Path("data/data_openalex")
TIMESERIES_DIR = OUTDIR / "timeseries"

OPENALEX_TS = TIMESERIES_DIR / "all_keywords_timeseries.csv"
WIKI_TS = Path("data/data_wiki/wiki_pageviews_yearly.csv")

MASTER_OUT = TIMESERIES_DIR / "buzzword_timeseries_master.csv"


# ---------- helpers ----------

def normalize_by_keyword(df: pd.DataFrame, value_col: str,
                         group_cols: List[str] = ["keyword"]) -> pd.DataFrame:
    """
    Min–max normalize each keyword's time series on value_col.

    Adds a new column f"{value_col}_norm" with values in [0,1] per group.
    """
    new_col = value_col + "_norm"

    def _norm(s: pd.Series) -> pd.Series:
        mn, mx = s.min(), s.max()
        if pd.isna(mn) or pd.isna(mx) or mx == mn:
            return pd.Series(0.0, index=s.index)
        return (s - mn) / (mx - mn)

    df[new_col] = df.groupby(group_cols)[value_col].transform(_norm)
    return df


def _infer_value_column(df: pd.DataFrame, preferred_names: List[str]) -> str:
    """Pick the first existing column from preferred_names, or a numeric fallback."""
    for name in preferred_names:
        if name in df.columns:
            return name

    candidates = [
        c for c in df.columns
        if c not in {"keyword", "year", "publication_year", "source"}
        and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not candidates:
        raise ValueError(f"Could not infer value column from columns: {list(df.columns)}")
    return candidates[0]


# ---------- loaders ----------

def load_openalex_ts(path: Path) -> pd.DataFrame:
    """Load OpenAlex TS, restrict to 2015–2025, normalize within that window."""
    if not path.exists():
        raise SystemExit(f"OpenAlex timeseries file not found: {path}")

    df = pd.read_csv(path)

    # Rename columns to standard names
    df = df.rename(columns={
        "publication_year": "year",
        "count": "academic_raw",
    })

    if not {"keyword", "year", "academic_raw"}.issubset(df.columns):
        raise ValueError(f"OpenAlex TS missing required columns in {path}")

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["academic_raw"] = pd.to_numeric(df["academic_raw"], errors="coerce")

    # Restrict to 2015–2025
    df = df[(df["year"] >= YEAR_START) & (df["year"] <= YEAR_END)].copy()

    # Re-normalize within 2015–2025
    df = normalize_by_keyword(df, value_col="academic_raw")
    df = df.rename(columns={"academic_raw_norm": "academic_norm"})

    return df[["keyword", "year", "academic_raw", "academic_norm"]]


def load_wiki_ts(path: Path) -> Optional[pd.DataFrame]:
    """Load Wiki yearly TS, restrict to 2015–2025, normalize within that window."""
    if not path.exists():
        print(f"[WARN] Wikipedia timeseries file not found, skipping: {path}")
        return None

    df = pd.read_csv(path)

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

    if not {"keyword", "year"}.issubset(df.columns):
        raise ValueError(f"Wikipedia TS must have keyword & year (got {list(df.columns)})")

    value_col = _infer_value_column(
        df,
        preferred_names=["views", "pageviews", "count", "value"],
    )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    # Restrict to 2015–2025
    df = df[(df["year"] >= YEAR_START) & (df["year"] <= YEAR_END)].copy()

    # Normalize within 2015–2025 per keyword
    df = normalize_by_keyword(df, value_col=value_col)
    df = df.rename(columns={
        value_col: "wiki_raw",
        value_col + "_norm": "wiki_norm",
    })

    return df[["keyword", "year", "wiki_raw", "wiki_norm"]]


# ---------- main ----------

def main() -> None:
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] OpenAlex TS: {OPENALEX_TS}")
    df_ac = load_openalex_ts(OPENALEX_TS)
    print(f"[INFO] OpenAlex rows (2015–2025): {len(df_ac)}")

    print(f"[LOAD] Wikipedia TS: {WIKI_TS}")
    df_wiki = load_wiki_ts(WIKI_TS)

    df = df_ac.copy()

    if df_wiki is not None:
        df = df.merge(df_wiki, on=["keyword", "year"], how="outer")
        print(f"[MERGE] + Wikipedia -> rows: {len(df)}")

    df = df.sort_values(["keyword", "year"]).reset_index(drop=True)

    MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_OUT, index=False)
    print(f"[SAVE] master buzzword timeseries -> {MASTER_OUT}")

    print("\n[HEAD] master timeseries:")
    print(df.head())


if __name__ == "__main__":
    main()
