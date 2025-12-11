#Analyze Label Studio hype annotations and merge with OpenAlex metadata.
from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

OUTDIR = Path("data/data_openalex")
ANNOTATION_DIR = OUTDIR / "annotation"
PROCESSED_DIR = OUTDIR / "processed"

LABELSTUDIO_EXPORT = (
    ANNOTATION_DIR
    / "Annotation_results.csv")


def load_and_clean_annotations(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    print("[DEBUG] Columns in Label Studio export:", list(df.columns))

    if "id" in df.columns:
        df = df.rename(columns={"id": "task_id"})

    cols_to_drop: List[str] = [c for c in df.columns if c.startswith("Unnamed")]
    if "1-Piotr 2-Hanka 3-Jakub 4-Ondrej" in df.columns:
        cols_to_drop.append("1-Piotr 2-Hanka 3-Jakub 4-Ondrej")
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    rater_cols = [c for c in ["Piotr", "Ondrej", "Jakub", "Hanka"] if c in df.columns]
    if not rater_cols:
        raise ValueError("No rater columns found in Label Studio export.")

    df[rater_cols] = df[rater_cols].apply(pd.to_numeric, errors="coerce")

    df["hype_mean"] = df[rater_cols].mean(axis=1)
    df["hype_std"] = df[rater_cols].std(axis=1)
    df["hype_median"] = df[rater_cols].median(axis=1)
    df["n_ratings"] = df[rater_cols].notna().sum(axis=1)

    out_clean = ANNOTATION_DIR / "hype_scores_clean.csv"
    df.to_csv(out_clean, index=False)
    print(f"[SAVE] cleaned hype scores -> {out_clean}")

    return df

def merge_with_openalex_meta(df_anno: pd.DataFrame) -> pd.DataFrame:
    meta_path = ANNOTATION_DIR / "annotation_titles_labelstudio.csv"
    df_meta = pd.read_csv(meta_path)

    df_meta = df_meta.rename(columns={"id": "openalex_id"})

    if "text" not in df_meta.columns:
        raise ValueError(f"'text' column not found in {meta_path}")
    if "text" not in df_anno.columns:
        raise ValueError("'text' column not found in annotation DataFrame")

    merged = df_meta.merge(
        df_anno,
        on="text",
        how="inner",
        suffixes=("_meta", "_anno"),
    )

    print(f"[MERGE] meta x annotations -> {len(merged)} rows")

    citation_frames = []
    for p in sorted(PROCESSED_DIR.glob("*.parquet")):
        try:
            df_p = pd.read_parquet(p)
        except Exception as e:
            print(f"[WARN] failed to read {p}: {e}")
            continue

        keep_cols = [c for c in ["id", "cited_by_count"] if c in df_p.columns]
        if not keep_cols:
            continue

        citation_frames.append(df_p[keep_cols])

    if citation_frames:
        df_cit = (
            pd.concat(citation_frames, ignore_index=True)
            .drop_duplicates(subset=["id"])
            .rename(columns={"id": "openalex_id"})
        )
        merged = merged.merge(df_cit, on="openalex_id", how="left")
        print(f"[MERGE] added citations for {merged['cited_by_count'].notna().sum()} papers")
    else:
        print("[WARN] no citation data found in processed parquet files")

    out_path = ANNOTATION_DIR / "hype_with_openalex_meta.csv"
    merged.to_csv(out_path, index=False)
    print(f"[SAVE] hype + OpenAlex meta -> {out_path}")

    return merged

def main() -> None:
    if not LABELSTUDIO_EXPORT.exists():
        raise SystemExit(f"Label Studio export not found: {LABELSTUDIO_EXPORT}")

    print(f"[LOAD] Label Studio export: {LABELSTUDIO_EXPORT}")
    df_anno = load_and_clean_annotations(LABELSTUDIO_EXPORT)
    merged = merge_with_openalex_meta(df_anno)

    print("\n[HEAD] merged hype + meta:")
    cols = [
        "openalex_id",
        "keyword",
        "publication_year",
        "text",
        "hype_mean",
    ]
    if "cited_by_count" in merged.columns:
        cols.append("cited_by_count")
    print(merged[cols].head())

if __name__ == "__main__":
    main()
