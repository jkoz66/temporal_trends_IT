import pandas as pd
import pathlib
import random

# -----------------------------
# CONFIG
# -----------------------------
PROCESSED_DIR = pathlib.Path("data_openalex/processed")
OUTPUT_FILE   = "annotation_titles_2015_2025.csv"

# List of canonical buzzwords (same order as in your pipeline)
KEYWORDS = [
    "big data",
    "quantum computing",
    "artificial general intelligence",
    "blockchain",
    "cryptocurrency",
    "metaverse",
    "edge computing",
    "generative ai",
    "chatbot",
    "augmented reality",
]

N_PER_KEYWORD = 10
YEAR_MIN, YEAR_MAX = 2015, 2025
RANDOM_STATE = 42

# -----------------------------
# SAMPLING LOGIC
# -----------------------------
random.seed(RANDOM_STATE)

samples = []

for kw in KEYWORDS:
    fname = PROCESSED_DIR / f"{kw.replace(' ', '_')}.parquet"
    print(f"Loading {fname} ...")

    df = pd.read_parquet(fname)

    # Filter year range
    df = df[df["publication_year"].between(YEAR_MIN, YEAR_MAX)]

    # Drop rows with missing titles (rare but safe)
    df = df.dropna(subset=["title"])

    # Sample n=10 (or fewer if dataset is small)
    n = min(N_PER_KEYWORD, len(df))
    sample_df = df.sample(n=n, random_state=RANDOM_STATE).copy()

    # Attach keyword column explicitly (some pipelines keep it, some don't)
    sample_df["keyword"] = kw

    # Keep only fields needed for annotation
    sample_df = sample_df[["id", "keyword", "title", "publication_year"]]

    samples.append(sample_df)

# Combine all keywords
annotation_df = pd.concat(samples, ignore_index=True)

# Label Studio likes a "text" column
annotation_df = annotation_df.rename(columns={"title": "text"})

# Save
annotation_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved annotation sample → {OUTPUT_FILE}")
print(f"Total rows: {len(annotation_df)}")
