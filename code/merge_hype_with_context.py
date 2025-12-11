#!/usr/bin/env python
"""
Merge paper-level hype annotations with yearly buzzword context
(academic & public interest), and run basic RQ3 analyses.

Inputs:
    data/data_openalex/annotation/hype_with_openalex_meta.csv
        - from analyze_hype_annotations.py
        - has: hype_mean, cited_by_count, keyword, publication_year, ...

    data/buzzword_timeseries_master.csv
        - from merge_buzzword_timeseries.py
        - has: keyword, year, academic_norm, wiki_norm, gtrends_norm, public_norm

Output:
    data/data_openalex/annotation/hype_with_context.csv
        - paper-level file with added context columns:
            academic_norm_in_year
            wiki_norm_in_year
            gtrends_norm_in_year
            external_interest_in_publication_year

Figures:
    figures/hype_vs_public_interest.png
    figures/hype_vs_public_interest_by_keyword.png
    figures/hype_vs_academic_in_year.png
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

# ------------------ paths ------------------

ANNOT_PATH = Path("data/data_openalex/annotation/hype_with_openalex_meta.csv")
TS_PATH    = Path("data/buzzword_timeseries_master.csv")
OUT_PATH   = Path("data/data_openalex/annotation/hype_with_context.csv")
FIG_DIR    = Path("figures")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    # ---------- 1) Load data ----------

    print(f"[LOAD] hype annotations + meta: {ANNOT_PATH}")
    df = pd.read_csv(ANNOT_PATH)

    print(f"[LOAD] master timeseries: {TS_PATH}")
    ts = pd.read_csv(TS_PATH)

    # Canonicalise keyword strings on both sides
    for d in (df, ts):
        d["keyword"] = d["keyword"].astype(str).str.strip().str.lower()

    # ---------- 2) Select relevant yearly context ----------

    ts_ctx = ts[[
        "keyword",
        "year",
        "academic_norm",
        "wiki_norm",
        "gtrends_norm",
        "public_norm",
    ]].copy()

    # ---------- 3) Merge paper-level with yearly context ----------

    merged = df.merge(
        ts_ctx,
        left_on=["keyword", "publication_year"],
        right_on=["keyword", "year"],
        how="left",
    )
    merged = merged.drop(columns=["year"])

    merged = merged.rename(columns={
        "academic_norm": "academic_norm_in_year",
        "wiki_norm": "wiki_norm_in_year",
        "gtrends_norm": "gtrends_norm_in_year",
        "public_norm": "external_interest_in_publication_year",
    })

    print(f"[MERGE] hype + context -> {len(merged)} rows")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(OUT_PATH, index=False)
    print(f"[SAVE] {OUT_PATH}")

    # ---------- 4) Basic correlations (RQ3) ----------

    # Filter to rows where we actually have external interest
    dfc = merged.dropna(subset=["external_interest_in_publication_year"])

    if dfc.empty:
        print("[WARN] No rows with external interest; skipping correlation + plots.")
        return

    print("\n=== Correlations: hype_mean vs external_interest_in_publication_year ===")
    print("Pearson :",
          dfc["hype_mean"].corr(dfc["external_interest_in_publication_year"], method="pearson"))
    print("Spearman:",
          dfc["hype_mean"].corr(dfc["external_interest_in_publication_year"], method="spearman"))

    print("\n=== Correlations: hype_mean vs academic_norm_in_year ===")
    print("Pearson :",
          dfc["hype_mean"].corr(dfc["academic_norm_in_year"], method="pearson"))
    print("Spearman:",
          dfc["hype_mean"].corr(dfc["academic_norm_in_year"], method="spearman"))

        # ---------- 5) Plots (RQ3 figures) ----------

    # (a) Global: Hype vs public interest at publication year
    plt.figure(figsize=(6, 4))
    plt.scatter(
        dfc["hype_mean"],
        dfc["external_interest_in_publication_year"],
        alpha=0.8,
    )
    plt.xlabel("Title hype score (mean of annotators)")
    plt.ylabel("Public interest (public_norm) in year")
    plt.title("Hype Score vs Public Interest in Publication Year")
    plt.tight_layout()
    out_main_public = FIG_DIR / "hype_vs_public_interest.png"
    plt.savefig(out_main_public, dpi=300)
    plt.close()
    print(f"[SAVE] {out_main_public}")

    # (b) Per-keyword panels: Hype vs public interest
    for kw, sub in dfc.groupby("keyword"):
        plt.figure(figsize=(5, 4))
        plt.scatter(
            sub["hype_mean"],
            sub["external_interest_in_publication_year"],
            alpha=0.8,
        )
        plt.xlabel("Title hype score (mean of annotators)")
        plt.ylabel("Public interest (public_norm) in year")
        plt.title(f"Hype vs Public Interest — {kw}")
        plt.tight_layout()
        fname = FIG_DIR / f"hype_vs_public_interest_{kw.replace(' ', '_')}.png"
        plt.savefig(fname, dpi=300)
        plt.close()
        print(f"[SAVE] {fname}")

    # (c) Global: Hype vs academic_norm_in_year
    plt.figure(figsize=(6, 4))
    plt.scatter(
        dfc["hype_mean"],
        dfc["academic_norm_in_year"],
        alpha=0.8,
    )
    plt.xlabel("Title hype score (mean of annotators)")
    plt.ylabel("Normalized academic activity in year")
    plt.title("Hype Score vs Academic Interest in Publication Year")
    plt.tight_layout()
    out_main_academic = FIG_DIR / "hype_vs_academic_in_year.png"
    plt.savefig(out_main_academic, dpi=300)
    plt.close()
    print(f"[SAVE] {out_main_academic}")



        # ---------- 6) Per-buzzword correlations ----------
    print("\n=== Per-buzzword correlations (hype_mean vs public_norm) ===")

    rows = []
    for kw, sub in merged.groupby("keyword"):
        sub = sub.dropna(subset=["external_interest_in_publication_year"])
        if len(sub) < 3:
            continue  # skip tiny groups; correlation meaningless

        pear = sub["hype_mean"].corr(sub["external_interest_in_publication_year"], method="pearson")
        spear = sub["hype_mean"].corr(sub["external_interest_in_publication_year"], method="spearman")

        rows.append({
            "keyword": kw,
            "n": len(sub),
            "pearson": pear,
            "spearman": spear,
        })

    corr_df = pd.DataFrame(rows)
    print(corr_df)

    # Optionally save
    corr_df.to_csv(FIG_DIR / "per_buzzword_correlations_public_norm.csv", index=False)
    print(f"[SAVE] {FIG_DIR / 'per_buzzword_correlations_public_norm.csv'}")


if __name__ == "__main__":
    main()
