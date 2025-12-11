import pandas as pd
import pathlib
import random

PROCESSED_DIR = pathlib.Path("data_openalex/processed")
OUTPUT_FILE   = "annotation_titles_2015_2025.csv"

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
RANDOM_STATE = 16

random.seed(RANDOM_STATE)

samples = []

for kw in KEYWORDS:
    fname = PROCESSED_DIR / f"{kw.replace(' ', '_')}.parquet"
    print(f"Loading {fname} ...")

    df = pd.read_parquet(fname)

    df = df[df["publication_year"].between(YEAR_MIN, YEAR_MAX)]

    df = df.dropna(subset=["title"])

    n = min(N_PER_KEYWORD, len(df))
    sample_df = df.sample(n=n, random_state=RANDOM_STATE).copy()

    sample_df["keyword"] = kw

    sample_df = sample_df[["id", "keyword", "title", "publication_year"]]

    samples.append(sample_df)

annotation_df = pd.concat(samples, ignore_index=True)

annotation_df = annotation_df.rename(columns={"title": "text"})

annotation_df.to_csv(OUTPUT_FILE, index=False)
print(f"\nSaved annotation sample → {OUTPUT_FILE}")
print(f"Total rows: {len(annotation_df)}")
