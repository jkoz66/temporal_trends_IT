import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


def load_and_prepare_data(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path).copy()

    # --- unify YEAR column ---
    if "year" in df.columns:
        df["year"] = pd.to_numeric(df["year"], errors="coerce")
    elif "publication_year" in df.columns:
        df["year"] = pd.to_numeric(df["publication_year"], errors="coerce")
    else:
        raise ValueError(
            "Could not find a 'year' or 'publication_year' column in the CSV. "
            f"Columns present: {df.columns.tolist()}"
        )

    # --- unify / create NORMALIZED academic series for plotting ---
    if "academic_norm" in df.columns:
        df["academic_norm_plot"] = df["academic_norm"]
    elif "normalized_count" in df.columns:
        df["academic_norm_plot"] = df["normalized_count"]
    elif "count" in df.columns:
        # normalize per keyword: value / max(value) for that keyword
        df["academic_norm_plot"] = (
            df.groupby("keyword")["count"].transform(
                lambda x: x / x.max() if x.max() > 0 else 0
            )
        )
    else:
        raise ValueError(
            "Could not find 'academic_norm', 'normalized_count', or 'count' columns "
            f"to plot. Columns present: {df.columns.tolist()}"
        )

    # Drop rows with missing year or keyword
    df = df.dropna(subset=["year", "keyword"])
    return df


def plot_rq1_multiline(df: pd.DataFrame, out_path: Path) -> None:
    df_sorted = df.sort_values(["keyword", "year"])

    plt.figure(figsize=(10, 6))

    for keyword, sub in df_sorted.groupby("keyword"):
        plt.plot(
            sub["year"],
            sub["academic_norm_plot"],
            marker="o",
            linestyle="-",
            label=keyword,
        )

    plt.xlabel("Year")
    plt.ylabel("Normalized academic attention")
    plt.title("Academic attention to tech buzzwords over time")
    plt.legend(title="Keyword", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.tight_layout()

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved multi-line RQ1 plot to: {out_path}")


def plot_rq1_small_multiples(df: pd.DataFrame, out_path: Path) -> None:
    df_sorted = df.sort_values(["keyword", "year"])
    keywords = sorted(df_sorted["keyword"].unique())
    n_keywords = len(keywords)

    # Match RQ2 layout: 2 columns, many rows
    n_cols = 2
    n_rows = (n_keywords + n_cols - 1) // n_cols

    # Match RQ2 figsize style: wide, readable in ACM PDF
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(10, 12))
    axes = axes.flatten()

    for ax, keyword in zip(axes, keywords):
        sub = df_sorted[df_sorted["keyword"] == keyword]

        ax.plot(
            sub["year"],
            sub["academic_norm_plot"],
            marker="o",
            linestyle="-",
            linewidth=1.2
        )
        ax.set_title(keyword, fontsize=10)
        ax.grid(True, linestyle=":", alpha=0.5)

        ax.set_xlabel("Year", fontsize=9)
        ax.set_ylabel("Normalized academic\nattention", fontsize=9)

        ax.set_ylim(-0.05, 1.05)

    # Hide unused axes if count is odd
    for ax in axes[n_keywords:]:
        ax.set_visible(False)

    fig.suptitle("Academic hype cycles by keyword", fontsize=12, y=0.98)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.25)

    sep_line = Line2D(
        [0.5, 0.5],     #x-coordinates
        [0.02, 0.945],  #y-coordinates
        transform=fig.transFigure,
        linewidth=0.8,
        color="grey",
        alpha=0.7,
    )
    fig.add_artist(sep_line)

    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved small-multiples RQ1 plot to: {out_path}")


def main():
    # === CHOOSE WHICH FILE TO USE HERE ===
    # Option A: master (OpenAlex + external signals)
    csv_path = Path("data/buzzword_timeseries_master.csv")

    # Option B: OpenAlex-only timeseries from the pipeline
    # csv_path = Path("all_keywords_timeseries.csv")

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV file not found: {csv_path}")

    print(f"Loading data from: {csv_path}")
    df = load_and_prepare_data(csv_path)

    # === RQ1 plots ===
    plot_rq1_multiline(df, Path("figures/RQ1_hypecycles_multiline.png"))
    plot_rq1_small_multiples(df, Path("figures/RQ1_hypecycles_smallmultiples.png"))


if __name__ == "__main__":
    main()
