import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path

# Load blockchain academic timeseries
df = pd.read_csv("data/data_openalex/timeseries/blockchain_timeseries.csv")

# Ensure correct columns
if "publication_year" not in df.columns:
    raise ValueError("Expected publication_year in blockchain_timeseries.csv")

# Convert year to datetime (Jan 1st each year)
df["Date"] = pd.to_datetime(df["publication_year"].astype(str) + "-01-01")

# Determine which academic metric to plot
if "normalized_count" in df.columns:
    y = df["normalized_count"]
    y_label = "Academic attention (normalized)"
elif "count" in df.columns:
    y = df["count"]
    y_label = "Publication count"
else:
    raise ValueError("Expected count or normalized_count in the CSV.")

# ---------- PLOT ----------
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Date"], y, linewidth=2)

ax.set_xlabel("Date")
ax.set_ylabel(y_label)
ax.set_title("Academic publications over time – Blockchain")

# Match the grid style from your examples
ax.grid(True, linestyle="--", alpha=0.6)

# Match x-axis formatting
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("figures/blockchain_academic_timeseries.png", dpi=300, bbox_inches="tight")
plt.show()
