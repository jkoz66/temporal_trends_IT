import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- Load data ---
df = pd.read_csv("../data/Annotation_results.csv")

# Columns: Piotr, Hanka, Jakub, Ondrej, text
annotator_cols = ["Piotr", "Hanka", "Jakub", "Ondrej"]

# --- Compute mean annotation per title ---
df["mean_score"] = df[annotator_cols].mean(axis=1)

# --- Long format for per-annotator plots ---
long_df = df.melt(
    id_vars=["text"],
    value_vars=annotator_cols,
    var_name="annotator",
    value_name="score"
)

plt.figure(figsize=(8, 5))
sns.histplot(
    data=df,
    x="mean_score",
    bins=9,  
    kde=True
)
plt.xlabel("Mean hype score across annotators")
plt.ylabel("Number of titles")
plt.title("Distribution of mean hype score per title")
plt.tight_layout()
plt.show()
