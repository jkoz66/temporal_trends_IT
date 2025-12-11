from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ANNOT_PATH = Path("data/data_openalex/annotation/hype_with_openalex_meta.csv")
TS_PATH    = Path("data/buzzword_timeseries_master.csv")
OUT_PATH   = Path("data/data_openalex/annotation/hype_with_context.csv")
FIG_DIR    = Path("figures")


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[LOAD] hype annotations + meta: {ANNOT_PATH}")
    df = pd.read_csv(ANNOT_PATH)

    print(f"[LOAD] master timeseries: {TS_PATH}")
    ts = pd.read_csv(TS_PATH)

    for d in (df, ts):
        d["keyword"] = d["keyword"].astype(str).str.strip().str.lower()

    ts_ctx = ts[[
        "keyword",
        "year",
        "academic_norm",
        "wiki_norm",
        "gtrends_norm",
        "public_norm",
    ]].copy()

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


if __name__ == "__main__":
    main()