#!/usr/bin/env python
"""
Analyze Label Studio hype annotations and merge with OpenAlex metadata.

Steps:
- load Label Studio CSV export
- compute Krippendorff's alpha (ordinal, 1–5 scale)
- compute per-title hype_mean etc.
- merge with original annotation_titles_labelstudio.csv (OpenAlex id, keyword, year)
- merge citations from processed OpenAlex parquet files
- write out:
    data_openalex/annotation/hype_scores_clean.csv
    data_openalex/annotation/hype_with_openalex_meta.csv
"""

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd

# ---------------------------------------------------------------------
# Paths (adjust LABELSTUDIO_EXPORT to your actual file name if needed)
# ---------------------------------------------------------------------

OUTDIR = Path("data/data_openalex")
ANNOTATION_DIR = OUTDIR / "annotation"
PROCESSED_DIR = OUTDIR / "processed"

# TODO: change this to match your Label Studio export filename if different
LABELSTUDIO_EXPORT = (
    ANNOTATION_DIR
    / "Annotation_results.csv"
)

# ---------------------------------------------------------------------
# Krippendorff's alpha (ordinal)
# ---------------------------------------------------------------------

def krippendorff_alpha(data: np.ndarray, level_of_measurement: str = "ordinal") -> float:
    """
    Compute Krippendorff's alpha for inter-annotator agreement.

    Parameters
    ----------
    data : 2D array-like (items x raters)
        Use np.nan for missing ratings.
    level_of_measurement : {"nominal", "ordinal"}
        For hype ratings 1–5 we use "ordinal".

    Returns
    -------
    alpha : float
        Krippendorff's alpha, or np.nan if it cannot be computed.
    """
    data = np.asarray(data, dtype=float)

    # Remove items with no ratings at all
    mask_nonempty = ~np.all(np.isnan(data), axis=1)
    data = data[mask_nonempty]
    if data.size == 0:
        return np.nan

    # Distinct rating values
    values = np.unique(data[~np.isnan(data)])
    if len(values) < 2:
        return np.nan

    # Map rating values to indices
    val_to_idx = {v: i for i, v in enumerate(values)}
    k = len(values)

    # Coincidence matrix
    coincidence = np.zeros((k, k), dtype=float)
    for row in data:
        row_vals = row[~np.isnan(row)]
        n = len(row_vals)
        if n < 2:
            continue
        for i in range(n):
            for j in range(n):
                if i == j:
                    continue
                vi, vj = row_vals[i], row_vals[j]
                coincidence[val_to_idx[vi], val_to_idx[vj]] += 1

    # Distance matrix
    if level_of_measurement == "nominal":
        delta = np.ones((k, k)) - np.eye(k)
    elif level_of_measurement == "ordinal":
        delta = np.zeros((k, k), dtype=float)
        for i, vi in enumerate(values):
            for j, vj in enumerate(values):
                delta[i, j] = (vi - vj) ** 2
    else:
        raise ValueError(f"Unsupported level_of_measurement: {level_of_measurement}")

    # Observed disagreement
    Do = float((coincidence * delta).sum())

    # Expected disagreement
    marginals = coincidence.sum(axis=0)
    N = float(marginals.sum())
    if N <= 1:
        return np.nan

    expected = np.outer(marginals, marginals) / (N - 1)
    De = float((expected * delta).sum())
    if De == 0:
        return np.nan

    return 1.0 - Do / De

# ---------------------------------------------------------------------
# Main pieces
# ---------------------------------------------------------------------

def load_and_clean_annotations(path: Path) -> pd.DataFrame:
    """Load Label Studio CSV export, compute hype stats & Krippendorff's alpha."""
    df = pd.read_csv(path)
    print("[DEBUG] Columns in Label Studio export:", list(df.columns))

    # Rename id to avoid confusion with OpenAlex id later
    if "id" in df.columns:
        df = df.rename(columns={"id": "task_id"})

    # Drop junk columns if present
    cols_to_drop: List[str] = [c for c in df.columns if c.startswith("Unnamed")]
    if "1-Piotr 2-Hanka 3-Jakub 4-Ondrej" in df.columns:
        cols_to_drop.append("1-Piotr 2-Hanka 3-Jakub 4-Ondrej")
    if cols_to_drop:
        df = df.drop(columns=cols_to_drop)

    # Ensure rater columns exist and are numeric
    rater_cols = [c for c in ["Piotr", "Ondrej", "Jakub", "Hanka"] if c in df.columns]
    if not rater_cols:
        raise ValueError("No rater columns found in Label Studio export.")

    df[rater_cols] = df[rater_cols].apply(pd.to_numeric, errors="coerce")

    # Compute Krippendorff's alpha (ordinal)
    alpha_ord = krippendorff_alpha(df[rater_cols].values, "ordinal")
    print(f"Krippendorff's alpha (ordinal): {alpha_ord:.3f}")

    # Per-title summary stats
    df["hype_mean"] = df[rater_cols].mean(axis=1)
    df["hype_std"] = df[rater_cols].std(axis=1)
    df["hype_median"] = df[rater_cols].median(axis=1)
    df["n_ratings"] = df[rater_cols].notna().sum(axis=1)

    # Save a clean version with hype scores
    out_clean = ANNOTATION_DIR / "hype_scores_clean.csv"
    df.to_csv(out_clean, index=False)
    print(f"[SAVE] cleaned hype scores -> {out_clean}")

    return df

def merge_with_openalex_meta(df_anno: pd.DataFrame) -> pd.DataFrame:
    """
    Merge cleaned annotation DataFrame with OpenAlex metadata and citations.

    Expects:
        - annotation_titles_labelstudio.csv in ANNOTATION_DIR
        - parquet files in PROCESSED_DIR with columns id, cited_by_count
    """
    meta_path = ANNOTATION_DIR / "annotation_titles_labelstudio.csv"
    df_meta = pd.read_csv(meta_path)

    # Original meta file has OpenAlex id in 'id' and title in 'text'
    df_meta = df_meta.rename(columns={"id": "openalex_id"})

    if "text" not in df_meta.columns:
        raise ValueError(f"'text' column not found in {meta_path}")
    if "text" not in df_anno.columns:
        raise ValueError("'text' column not found in annotation DataFrame")

    # Merge on title text (Label Studio kept the 'text' column)
    merged = df_meta.merge(
        df_anno,
        on="text",
        how="inner",
        suffixes=("_meta", "_anno"),
    )

    print(f"[MERGE] meta x annotations -> {len(merged)} rows")

    # -----------------------------------------------------------------
    # Attach citations from processed parquet files
    # -----------------------------------------------------------------
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

    # Quick sanity check printout
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
