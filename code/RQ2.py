import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path


# =========================
# CONFIG
# =========================

# Main input file
CSV_PATH = Path("data/buzzword_timeseries_master.csv")

# Whether to also make separate plots for wiki_norm and gtrends_norm
MAKE_WIKI_PLOTS = True
MAKE_GTRENDS_PLOTS = True


# =========================
# Data loading & preparation
# =========================

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

    # --- make sure the key RQ2 columns exist ---
    required_cols = ["keyword", "year", "academic_norm", "public_norm"]
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        raise ValueError(
            f"Missing required columns for RQ2: {missing}. "
            f"Available columns: {df.columns.tolist()}"
        )

    # coerce to numeric
    df["academic_norm"] = pd.to_numeric(df["academic_norm"], errors="coerce")
    df["public_norm"] = pd.to_numeric(df["public_norm"], errors="coerce")

    # optional components
    if "wiki_norm" in df.columns:
        df["wiki_norm"] = pd.to_numeric(df["wiki_norm"], errors="coerce")
    if "gtrends_norm" in df.columns:
        df["gtrends_norm"] = pd.to_numeric(df["gtrends_norm"], errors="coerce")

    # Drop rows with missing key values
    df = df.dropna(subset=["year", "keyword", "academic_norm", "public_norm"])

    return df


# =========================
# Generic helpers
# =========================

def scatter_grid(df: pd.DataFrame, x_col: str, y_col: str,
                 x_label: str, y_label: str,
                 title: str, out_path: Path) -> None:
    df_sorted = df.sort_values(["keyword", "year"])
    keywords = sorted(df_sorted["keyword"].unique())
    n_keywords = len(keywords)

    # Use 2 columns so each subplot is wider and easier to read in the PDF
    n_cols = 2
    n_rows = (n_keywords + n_cols - 1) // n_cols

    # Larger figure size so details survive ACM column scaling
    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(6 * n_cols, 3 * n_rows),
        sharex=False,
        sharey=False
    )
    axes = axes.flatten()

    for ax, keyword in zip(axes, keywords):
        sub = df_sorted[df_sorted["keyword"] == keyword]

        ax.scatter(
            sub[x_col],
            sub[y_col],
            alpha=0.7,
            edgecolor="none",
            s=25,  # bigger points so they are visible after shrinking in LaTeX
        )
        ax.set_title(keyword, fontsize=10)
        ax.set_xlabel(x_label, fontsize=9)
        ax.set_ylabel(y_label, fontsize=9)
        ax.grid(True, linestyle=":", alpha=0.5)

        ax.set_ylim(-0.05, 1.05)
        ax.set_xlim(-0.05, 1.05)

    # Hide unused axes (for example if n_keywords is odd)
    for ax in axes[n_keywords:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.16)

    sep_line = Line2D(
        [0.5, 0.5],     #x-coordinates
        [0.02, 0.945],  #y-coordinates
        transform=fig.transFigure,
        linewidth=0.8,
        color="grey",
        alpha=0.7,
    )
    fig.add_artist(sep_line)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved scatter grid to: {out_path}")


def dual_axis_grid(df: pd.DataFrame, left_col: str, right_col: str,
                   left_label: str, right_label: str,
                   title: str, out_path: Path) -> None:
    """
    Create a grid of dual-axis time-series plots (one per keyword),
    laid out similarly to the scatter_grid (2 columns, many rows)
    so that each panel is wide enough to be readable in the PDF.
    """
    df_sorted = df.sort_values(["keyword", "year"])
    keywords = sorted(df_sorted["keyword"].unique())
    n_keywords = len(keywords)

    # Match layout style to scatter_grid: 2 columns, more rows
    n_cols = 2
    n_rows = (n_keywords + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(10, 12),
    )  # Fixed size
    axes = axes.flatten()

    for ax, keyword in zip(axes, keywords):
        sub = df_sorted[df_sorted["keyword"] == keyword]

        # Left axis: academic_norm
        ax.plot(
            sub["year"],
            sub[left_col],
            marker="o",
            linestyle="-",
            label=left_label,
        )
        ax.set_title(keyword, fontsize=9)
        ax.set_xlabel("Year", fontsize=8)
        ax.set_ylabel(left_label, fontsize=8)
        ax.grid(True, linestyle=":", alpha=0.5)

        ax.set_ylim(-0.05, 1.05)

        # Right axis: public_norm
        ax2 = ax.twinx()
        ax2.plot(
            sub["year"],
            sub[right_col],
            marker="s",
            linestyle="--",
            label=right_label,
        )
        ax2.set_ylabel(right_label, fontsize=8)

        ax2.set_ylim(-0.05, 1.05)

    # Hide unused axes if n_keywords is odd
    for ax in axes[n_keywords:]:
        ax.set_visible(False)

    fig.suptitle(title, fontsize=12, y=0.98)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.subplots_adjust(wspace=0.4)

    sep_line = Line2D(
        [0.5, 0.5],     #x-coordinates
        [0.02, 0.945],  #y-coordinates
        transform=fig.transFigure,
        linewidth=0.8,
        color="grey",
        alpha=0.7,
    )
    fig.add_artist(sep_line)

    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved dual-axis grid to: {out_path}")


def correlation_bar(df: pd.DataFrame, y_col: str,
                    title: str, out_path: Path) -> None:
    keywords = sorted(df["keyword"].unique())
    pearsons = []
    spearmans = []

    for kw in keywords:
        sub = df[df["keyword"] == kw]
        if len(sub) < 2:
            pearsons.append(float("nan"))
            spearmans.append(float("nan"))
            continue

        pearson = sub[["academic_norm", y_col]].corr(method="pearson").iloc[0, 1]
        spearman = sub[["academic_norm", y_col]].corr(method="spearman").iloc[0, 1]
        pearsons.append(pearson)
        spearmans.append(spearman)

    corr_df = pd.DataFrame({
        "keyword": keywords,
        "pearson": pearsons,
        "spearman": spearmans,
    })

    plt.figure(figsize=(10, 5))
    plt.bar(corr_df["keyword"], corr_df["pearson"])
    plt.axhline(0, color="black", linewidth=0.8)
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(f"Pearson correlation\n(academic_norm vs {y_col})")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved correlation bar chart to: {out_path}")

    print(f"\n=== Correlations: academic_norm vs {y_col} ===")
    print(corr_df.to_string(index=False, float_format=lambda x: f"{x:0.3f}"))


# =========================
# Main
# =========================

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV file not found: {CSV_PATH}")

    print(f"Loading data from: {CSV_PATH}")
    df = load_and_prepare_data(CSV_PATH)

    # ---------- MAIN RQ2: academic_norm vs public_norm ----------
    scatter_grid(
        df,
        x_col="academic_norm",
        y_col="public_norm",
        x_label="Academic attention (normalized)",
        y_label="Public interest (normalized)",
        title="RQ2: Academic vs combined public interest (per keyword)",
        out_path=Path("figures/RQ2_scatter_academic_vs_public.png"),
    )

    dual_axis_grid(
        df,
        left_col="academic_norm",
        right_col="public_norm",
        left_label="Academic (norm)",
        right_label="Public (norm)",
        title="RQ2: Academic vs combined public interest over time",
        out_path=Path("figures/RQ2_timeseries_academic_vs_public.png"),
    )

    correlation_bar(
        df,
        y_col="public_norm",
        title="RQ2: Correlation between academic and combined public interest by keyword",
        out_path=Path("figures/RQ2_corr_academic_vs_public.png"),
    )

    # ---------- OPTIONAL: academic vs wiki_norm ----------
    if MAKE_WIKI_PLOTS and "wiki_norm" in df.columns:
        scatter_grid(
            df,
            x_col="academic_norm",
            y_col="wiki_norm",
            x_label="Academic attention (normalized)",
            y_label="Wikipedia attention (normalized)",
            title="RQ2 (component): Academic vs Wikipedia interest (per keyword)",
            out_path=Path("figures/RQ2_scatter_academic_vs_wiki.png"),
        )

        dual_axis_grid(
            df,
            left_col="academic_norm",
            right_col="wiki_norm",
            left_label="Academic (norm)",
            right_label="Wikipedia (norm)",
            title="RQ2 (component): Academic vs Wikipedia interest over time",
            out_path=Path("figures/RQ2_timeseries_academic_vs_wiki.png"),
        )

        correlation_bar(
            df,
            y_col="wiki_norm",
            title="RQ2 (component): Correlation between academic and Wikipedia interest",
            out_path=Path("figures/RQ2_corr_academic_vs_wiki.png"),
        )

    # ---------- OPTIONAL: academic vs gtrends_norm ----------
    if MAKE_GTRENDS_PLOTS and "gtrends_norm" in df.columns:
        scatter_grid(
            df,
            x_col="academic_norm",
            y_col="gtrends_norm",
            x_label="Academic attention (normalized)",
            y_label="Google Trends interest (normalized)",
            title="RQ2 (component): Academic vs Google Trends interest (per keyword)",
            out_path=Path("figures/RQ2_scatter_academic_vs_gtrends.png"),
        )

        dual_axis_grid(
            df,
            left_col="academic_norm",
            right_col="gtrends_norm",
            left_label="Academic (norm)",
            right_label="Google Trends (norm)",
            title="RQ2 (component): Academic vs Google Trends interest over time",
            out_path=Path("figures/RQ2_timeseries_academic_vs_gtrends.png"),
        )

        correlation_bar(
            df,
            y_col="gtrends_norm",
            title="RQ2 (component): Correlation between academic and Google Trends interest",
            out_path=Path("figures/RQ2_corr_academic_vs_gtrends.png"),
        )


if __name__ == "__main__":
    main()
