from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

OPENALEX_TS = Path("data/data_openalex") / "timeseries" / "all_keywords_timeseries.csv"
WIKI_TS     = Path("data/data_wiki")     / "wiki_pageviews_yearly.csv"
GTRENDS_TS  = Path("data/data_trends")   / "data_trends_yearly_long.csv"

MASTER_OUT  = Path("data") / "buzzword_timeseries_master.csv"

def _infer_value_column(df: pd.DataFrame, preferred_names: list[str]) -> str:
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
    if not path.exists():
        raise SystemExit(f"OpenAlex timeseries file not found: {path}")

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

    df["keyword"] = df["keyword"].str.strip().str.lower()

    required = {"keyword", "year"}
    if not required.issubset(df.columns):
        raise ValueError(
            f"OpenAlex timeseries must have columns keyword & year (got {list(df.columns)})"
        )

    if "count" in df.columns:
        df = df.rename(columns={"count": "academic_count"})
    elif "academic_count" not in df.columns:
        df["academic_count"] = pd.NA

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
    if not path.exists():
        print(f"[WARN] Wikipedia timeseries file not found, skipping: {path}")
        return None

    df = pd.read_csv(path)

    if "keyword" not in df.columns and "buzzword" in df.columns:
        df = df.rename(columns={"buzzword": "keyword"})

    df["keyword"] = df["keyword"].str.strip().str.lower()

    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype("Int64")

    df["wiki_raw"] = pd.to_numeric(df["views"], errors="coerce")

    def _norm(x: pd.Series) -> pd.Series:
        max_val = x.max()
        if pd.isna(max_val) or max_val == 0:
            return 0.0
        return x / max_val

    df["wiki_norm"] = df.groupby("keyword")["wiki_raw"].transform(_norm)

    return df[["keyword", "year", "wiki_raw", "wiki_norm"]]


def load_gtrends_ts(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        print(f"[WARN] Google Trends timeseries file not found, skipping: {path}")
        return None

    df = pd.read_csv(path)

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

    def _norm(x: pd.Series) -> pd.Series:
        max_val = x.max()
        if max_val in (0, None) or pd.isna(max_val):
            return 0.0
        return x / max_val

    df["gtrends_norm"] = df.groupby("keyword")["interest"].transform(_norm)

    return df[["keyword", "year", "gtrends_norm"]]


def main() -> None:
    MASTER_OUT.parent.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] OpenAlex TS: {OPENALEX_TS}")
    df_ac = load_openalex_ts(OPENALEX_TS)
    print(f"[INFO] OpenAlex rows: {len(df_ac)}")

    print(f"[LOAD] Wikipedia TS: {WIKI_TS}")
    df_wiki = load_wiki_ts(WIKI_TS)

    print(f"[LOAD] Google Trends TS: {GTRENDS_TS}")
    df_gt = load_gtrends_ts(GTRENDS_TS)

    df = df_ac.copy()

    if df_wiki is not None:
        df = df.merge(df_wiki, on=["keyword", "year"], how="left")
        print(f"[MERGE] + Wikipedia -> rows: {len(df)}")

    if df_gt is not None:
        df = df.merge(df_gt, on=["keyword", "year"], how="left")
        print(f"[MERGE] + Google Trends -> rows: {len(df)}")

    df["public_norm"] = (df["wiki_norm"].fillna(0) + df["gtrends_norm"].fillna(0)) / 2

    df = df.sort_values(["keyword", "year"]).reset_index(drop=True)

    df.to_csv(MASTER_OUT, index=False)
    print(f"[SAVE] master buzzword timeseries -> {MASTER_OUT}")

    print("\n[HEAD] master timeseries:")
    print(df.head())


if __name__ == "__main__":
    main()
