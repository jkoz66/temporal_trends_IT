#!/usr/bin/env python
"""
Merge academic (OpenAlex), Wikipedia, and Google Trends time series
into a single master file per buzzword-year.

Input (default paths, adjust as needed):

    data_openalex/timeseries/all_keywords_timeseries.csv
        - from openalex_buzzwords.py run_pipeline()
        - expected columns: keyword, year, count, normalized_count

    data_openalex/timeseries/wiki_timeseries_all_keywords.csv
        - your preprocessed Wikipedia time series
        - must have at least: keyword, year, (some value column)

    data_openalex/timeseries/google_trends_timeseries_all_keywords.csv
        - your preprocessed Google Trends time series
        - must have at least: keyword, year, (some value column)

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

import pandas as pd


# ---------------------------------------------------------------------
# Paths – adjust these two for your actual Wiki / Trends filenames
# ---------------------------------------------------------------------

OUTDIR = Path("data_openalex")
TIMESERIES_DIR = OUTDIR / "timeseries"

OPENALEX_TS = TIMESERIES_DIR / "all_keywords_timeseries.csv"

# 🔧 TODO: change these to your actual filenames
WIKI_TS = TIMESERIES_DIR / "wiki_timeseries_all_keywords.csv"
GTRENDS_TS = TIMESERIES_DIR / "google_trends_timeseries_all_keywords.csv"

MASTER_OUT = TIMESERIES_DIR / "buzzword_timeseries_master.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def load_openalex_ts(path: Path) -> pd.DataFrame:
    """Load OpenAlex time series, standardize column names."""

    if not path.exists():
        raise SystemExit(f"OpenAlex timeseries file not found: {path}")

    df = pd.read_csv(path)

    # Expect something like: keyword, year, count, normalized_count
    # Normalize column names
    # Try some common variants in case the CSV is slightly different.
    rename_map = {}

    # keyword
    if "keyword" not in df.columns:
        # try some alternatives (can extend if needed)
        for alt in ["term", "buzzword"]:
            if alt in df.columns:
                rename_map[alt] = "keyword"
                break

    # year
    if "year" not in df.columns:
        for alt in ["publication_year", "year_int"]:
            if alt in df.columns:
                rename_map[alt] = "year"
                break

    df = df.rename(columns=rename_map)

    required = {"keyword", "year"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"OpenAlex timeseries must have columns keyword & year (got {list(df.columns)})"
        )

    # academic count
    if "count" in df.columns:
        df = df.rename(columns={"count": "academic_count"})
    else:
        # if no 'count', we just set a dummy; the normalized column is more important.
        if "academic_count" not in df.columns:
            df["academic_count"] = pd.NA

    # academic normalized
    if "normalized_count" in df.columns:
        df = df.rename(columns={"normalized_count": "academic_norm"})
    elif "academic_norm" not in df.columns:
        # fall back to any number-like column if needed
        numeric_cols = [c for c in df.columns if c not in {"keyword", "year"}]
        if numeric_cols:
            df = df.rename(columns={numeric_cols[0]: "academic_norm"})
        else:
            raise ValueError("Could not find academic normalized column in OpenAlex TS.")

    # Ensure dtypes
    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df["academic_norm"] = pd.to_numeric(df["academic_norm"], errors="coerce")

    return df[["keyword", "year", "academic_count", "academic_norm"]]


def _infer_value_column(df: pd.DataFrame, preferred_names: list[str]) -> str:
    """Pick the first existing column from preferred_names, or a numeric fallback."""
    for name in preferred_names:
        if name in df.columns:
            return name

    # fallback: any numeric column that isn't keyword/year/source
    candidates = [
        c for c in df.columns
        if c not in {"keyword", "year", "source"} and pd.api.types.is_numeric_dtype(df[c])
    ]
    if not candidates:
        raise ValueError(f"Could not infer value column from columns: {list(df.columns)}")
    return candidates[0]


def load_wiki_ts(path: Path) -> Optional[pd.DataFrame]:
    """Load Wikipedia time series, return None if file missing."""
    if not path.exists():
        print(f"[WARN] Wikipedia timeseries file not found, skipping: {path}")
        return None

    df = pd.read_csv(path)

    # Standardize
    rename_map = {}
    if "keyword" not in df.columns:
        for alt in ["term", "buzzword"]:
            if alt in df.columns:
                rename_map[alt] = "keyword"
                break
    if "year" not in df.columns:
        for alt in ["year_int"]:
            if alt in df.columns:
                rename_map[alt] = "year"
                break

    df = df.rename(columns=rename_map)
    if not {"keyword", "year"}.issubset(df.columns):
        raise ValueError(f"Wikipedia TS must have keyword & year (got {list(df.columns)})")

    value_col = _infer_value_column(
        df,
        preferred_names=["normalized_count", "normalized_views", "wiki_norm", "value"],
    )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.rename(columns={value_col: "wiki_norm"})
    return df[["keyword", "year", "wiki_norm"]]


def load_gtrends_ts(path: Path) -> Optional[pd.DataFrame]:
    """Load Google Trends time series, return None if file missing."""
    if not path.exists():
        print(f"[WARN] Google Trends timeseries file not found, skipping: {path}")
        return None

    df = pd.read_csv(path)

    rename_map = {}
    if "keyword" not in df.columns:
        for alt in ["term", "buzzword"]:
            if alt in df.columns:
                rename_map[alt] = "keyword"
                break
    if "year" not in df.columns:
        for alt in ["year_int"]:
            if alt in df.columns:
                rename_map[alt] = "year"
                break

    df = df.rename(columns=rename_map)
    if not {"keyword", "year"}.issubset(df.columns):
        raise ValueError(f"Google Trends TS must have keyword & year (got {list(df.columns)})")

    value_col = _infer_value_column(
        df,
        preferred_names=["normalized_count", "normalized_interest", "gtrends_norm", "value"],
    )

    df["year"] = pd.to_numeric(df["year"], errors="coerce").astype("Int64")
    df[value_col] = pd.to_numeric(df[value_col], errors="coerce")

    df = df.rename(columns={value_col: "gtrends_norm"})
    return df[["keyword", "year", "gtrends_norm"]]


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    TIMESERIES_DIR.mkdir(parents=True, exist_ok=True)

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

    # Optional: sort for neatness
    df = df.sort_values(["keyword", "year"]).reset_index(drop=True)

    MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(MASTER_OUT, index=False)
    print(f"[SAVE] master buzzword timeseries -> {MASTER_OUT}")

    # Small sanity print
    print("\n[HEAD] master timeseries:")
    print(df.head())


if __name__ == "__main__":
    main()
