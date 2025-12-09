#!/usr/bin/env python
"""
Plot hype annotations vs citations and related patterns.

Input:
    data_openalex/annotation/hype_with_openalex_meta.csv

Outputs (all PNGs):
    figures/hype_vs_citations.png
    figures/hype_vs_citations_logy.png
    figures/hype_vs_citations_by_keyword_logy.png
    figures/hype_vs_citations_reg_logy.png
    figures/hype_score_distribution.png
    figures/citations_vs_year_colored_by_hype.png
"""

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


# ---------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------

OUTDIR = Path("data_openalex")
ANNOTATION_DIR = OUTDIR / "annotation"
FIG_DIR = Path("figures")
FIG_DIR.mkdir(exist_ok=True)

DATA_FILE = ANNOTATION_DIR / "hype_with_openalex_meta.csv"


# ---------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------

def clean_data(path: Path) -> pd.DataFrame:
    """Load merged file, clean basic stuff, return DataFrame."""
    if not path.exists():
        raise SystemExit(f"Data file not found: {path}")

    df = pd.read_csv(path)

    # Make sure expected columns exist
    required_cols = ["hype_mean", "cited_by_count"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {path}")

    # Numeric conversions
    df["hype_mean"] = pd.to_numeric(df["hype_mean"], errors="coerce")
    df["cited_by_count"] = pd.to_numeric(df["cited_by_count"], errors="coerce")

    # Drop rows without hype or citations
    df = df.dropna(subset=["hype_mean", "cited_by_count"]).copy()

    # Optional: clip negative citations (shouldn’t exist, but just in case)
    df = df[df["cited_by_count"] >= 0]

    # Optional: ensure publication_year numeric if present
    if "publication_year" in df.columns:
        df["publication_year"] = pd.to_numeric(df["publication_year"], errors="coerce")

    return df


def print_correlations(df: pd.DataFrame) -> None:
    """Print Pearson and Spearman correlations between hype and citations."""
    x = df["hype_mean"]
    y = df["cited_by_count"]

    pearson = x.corr(y, method="pearson")
    spearman = x.corr(y, method="spearman")

    print("=== Correlations: hype_mean vs cited_by_count ===")
    print(f"Pearson : {pearson:.3f}")
    print(f"Spearman: {spearman:.3f}")
    print("===============================================\n")


# ---------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------

def plot_hype_vs_citations(df: pd.DataFrame) -> None:
    """Scatter: hype_mean vs cited_by_count (linear scale)."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="hype_mean",
        y="cited_by_count",
        alpha=0.7,
    )
    plt.xlabel("Hype score (mean of annotators)")
    plt.ylabel("Citations")
    plt.title("Hype Score vs Citations")
    plt.tight_layout()

    out = FIG_DIR / "hype_vs_citations.png"
    plt.savefig(out, dpi=300)
    print(f"[SAVE] {out}")
    plt.close()


def plot_hype_vs_citations_logy(df: pd.DataFrame) -> None:
    """Scatter: hype_mean vs cited_by_count (log y-axis)."""
    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="hype_mean",
        y="cited_by_count",
        alpha=0.7,
    )
    plt.yscale("log")
    plt.xlabel("Hype score (mean of annotators)")
    plt.ylabel("Citations (log scale)")
    plt.title("Hype Score vs Citations (log scale)")
    plt.tight_layout()

    out = FIG_DIR / "hype_vs_citations_logy.png"
    plt.savefig(out, dpi=300)
    print(f"[SAVE] {out}")
    plt.close()


def plot_hype_vs_citations_by_keyword(df: pd.DataFrame) -> None:
    """Scatter: hype_mean vs cited_by_count, colored by keyword (log y-axis)."""
    if "keyword" not in df.columns:
        print("[SKIP] plot_hype_vs_citations_by_keyword: no 'keyword' column found.")
        return

    plt.figure(figsize=(9, 7))
    sns.scatterplot(
        data=df,
        x="hype_mean",
        y="cited_by_count",
        hue="keyword",
        alpha=0.8,
    )
    plt.yscale("log")
    plt.xlabel("Hype score (mean of annotators)")
    plt.ylabel("Citations (log scale)")
    plt.title("Hype vs Citations by Keyword")
    plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", title="Keyword")
    plt.tight_layout()

    out = FIG_DIR / "hype_vs_citations_by_keyword_logy.png"
    plt.savefig(out, dpi=300)
    print(f"[SAVE] {out}")
    plt.close()


def plot_hype_vs_citations_reg_logy(df: pd.DataFrame) -> None:
    """
    Regression: hype_mean vs log(1 + citations).

    We regress on log(1 + citations) to handle zeros and heavy tail.
    The plot shows log(1 + citations) on y-axis.
    """
    # Create a transformed column for regression
    df = df.copy()
    df["log1p_citations"] = np.log1p(df["cited_by_count"])

    plt.figure(figsize=(8, 6))
    sns.regplot(
        data=df,
        x="hype_mean",
        y="log1p_citations",
        scatter_kws={"alpha": 0.5},
        line_kws={},
    )
    plt.xlabel("Hype score (mean of annotators)")
    plt.ylabel("log(1 + Citations)")
    plt.title("Hype Score vs log(1 + Citations) with Linear Trend")
    plt.tight_layout()

    out = FIG_DIR / "hype_vs_citations_reg_logy.png"
    plt.savefig(out, dpi=300)
    print(f"[SAVE] {out}")
    plt.close()


def plot_hype_score_distribution(df: pd.DataFrame) -> None:
    """Distribution of hype_mean across all annotated titles."""
    plt.figure(figsize=(8, 6))
    sns.histplot(
        df["hype_mean"],
        bins=np.arange(0.5, 5.6, 0.5),
        kde=False,
    )
    plt.xlabel("Hype score (mean of annotators)")
    plt.ylabel("Number of papers")
    plt.title("Distribution of Hype Scores")
    plt.tight_layout()

    out = FIG_DIR / "hype_score_distribution.png"
    plt.savefig(out, dpi=300)
    print(f"[SAVE] {out}")
    plt.close()


def plot_citations_vs_year_colored_by_hype(df: pd.DataFrame) -> None:
    """
    Scatter: publication_year vs citations, colored by hype_mean.

    This helps see if hype is associated with more/less citedness
    across years and whether certain years are more hypey.
    """
    if "publication_year" not in df.columns:
        print("[SKIP] plot_citations_vs_year_colored_by_hype: no 'publication_year' column found.")
        return

    sub = df.dropna(subset=["publication_year"]).copy()
    if sub.empty:
        print("[SKIP] plot_citations_vs_year_colored_by_hype: no non-null publication_year.")
        return

    plt.figure(figsize=(9, 6))
    scatter = plt.scatter(
        sub["publication_year"],
        sub["cited_by_count"],
        c=sub["hype_mean"],
        alpha=0.7,
    )
    plt.yscale("log")
    plt.xlabel("Publication year")
    plt.ylabel("Citations (log scale)")
    plt.title("Citations over Time, Colored by Hype Score")
    cbar = plt.colorbar(scatter)
    cbar.set_label("Hype score (mean)")
    plt.tight_layout()

    out = FIG_DIR / "citations_vs_year_colored_by_hype.png"
    plt.savefig(out, dpi=300)
    print(f"[SAVE] {out}")
    plt.close()


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------

def main() -> None:
    sns.set_theme(style="whitegrid")

    print(f"[LOAD] {DATA_FILE}")
    df = clean_data(DATA_FILE)
    print(f"[INFO] rows after cleaning: {len(df)}")

    print_correlations(df)

    plot_hype_vs_citations(df)
    plot_hype_vs_citations_logy(df)
    plot_hype_vs_citations_by_keyword(df)
    plot_hype_vs_citations_reg_logy(df)
    plot_hype_score_distribution(df)
    plot_citations_vs_year_colored_by_hype(df)


if __name__ == "__main__":
    main()
