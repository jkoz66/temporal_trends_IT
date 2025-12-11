#!/usr/bin/env python

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats

ANNOT_PATH = Path("../data/data_openalex/annotation/hype_with_openalex_meta.csv")
TS_PATH = Path("../data/buzzword_timeseries_master.csv")
OUT_PATH = Path("../data/data_openalex/annotation/hype_with_context.csv")
FIG_DIR = Path("../figures")
DATA_DIR = Path("../data")

def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    
    print(f"[LOAD] hype annotations + meta: {ANNOT_PATH}")
    df = pd.read_csv(ANNOT_PATH)
    
    print(f"[LOAD] master timeseries: {TS_PATH}")
    ts = pd.read_csv(TS_PATH)
    
    for d in (df, ts):
        d["keyword"] = d["keyword"].astype(str).str.strip().str.lower()
    
    ts_ctx = ts[["keyword", "year", "academic_norm", "public_norm"]].copy()
    merged = df.merge(
        ts_ctx,
        left_on=["keyword", "publication_year"],
        right_on=["keyword", "year"],
        how="left",
    )
    merged = merged.drop(columns=["year"])
    merged = merged.rename(columns={
        "academic_norm": "academic_norm_in_year",
        "public_norm": "external_interest_in_publication_year",
    })
    
    dfc = merged.dropna(subset=["hype_mean", "academic_norm_in_year", "external_interest_in_publication_year"])
    
    if dfc.empty:
        print("[WARN] No valid data; skipping.")
        return
    
    print(f"[DATA] {len(dfc)} rows with complete data")
    
    correlations = []
    
    for kw, sub in dfc.groupby("keyword"):
        if len(sub) < 3:             continue
        
        spear_academic = sub["hype_mean"].corr(sub["academic_norm_in_year"], method="spearman")
        spear_public = sub["hype_mean"].corr(sub["external_interest_in_publication_year"], method="spearman")
        n_papers = len(sub)
        
        correlations.append({
            "keyword": kw,
            "n_papers": n_papers,
            "spearman_academic": spear_academic,
            "spearman_public": spear_public
        })
    
    corr_df = pd.DataFrame(correlations)
    
    if corr_df.empty:
        print("[WARN] No buzzwords with enough data; skipping plots.")
        return
    
    top10 = corr_df.nlargest(10, "n_papers")
    
    print("\n=== Top 10 buzzwords by Spearman correlation ===")
    print(top10[["keyword", "n_papers", "spearman_academic", "spearman_public"]].round(3))
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(top10)), top10["spearman_academic"], 
                   color='skyblue', edgecolor='navy', alpha=0.8)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{top10["spearman_academic"].iloc[i]:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Buzzwords (sorted by # papers)")
    plt.ylabel("Spearman Correlation (hype_mean vs academic_norm)")
    plt.title("Top 10 Buzzwords: Hype vs Academic Interest Correlation")
    plt.xticks(range(len(top10)), top10["keyword"], rotation=45, ha='right')
    plt.ylim(-1, 1)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    academic_plot = FIG_DIR / "top10_spearman_hype_vs_academic.png"
    plt.savefig(academic_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] Academic bar chart: {academic_plot}")
    
    plt.figure(figsize=(12, 6))
    
    bars = plt.bar(range(len(top10)), top10["spearman_public"], 
                   color='lightcoral', edgecolor='darkred', alpha=0.8)
    
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{top10["spearman_public"].iloc[i]:.3f}',
                ha='center', va='bottom', fontsize=10)
    
    plt.xlabel("Buzzwords (sorted by # papers)")
    plt.ylabel("Spearman Correlation (hype_mean vs public_norm)")
    plt.title("Top 10 Buzzwords: Hype vs Public Interest Correlation")
    plt.xticks(range(len(top10)), top10["keyword"], rotation=45, ha='right')
    plt.ylim(-1, 1)
    plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
    plt.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    
    public_plot = FIG_DIR / "top10_spearman_hype_vs_public.png"
    plt.savefig(public_plot, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"[SAVE] Public bar chart: {public_plot}")
    
    corr_df.to_csv(DATA_DIR / "buzzword_spearman_correlations.csv", index=False)
    print(f"[SAVE] Full correlations: {DATA_DIR / 'buzzword_spearman_correlations.csv'}")

if __name__ == "__main__":
    main()

