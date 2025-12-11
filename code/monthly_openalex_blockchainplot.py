import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

# Load the monthly OpenAlex data for blockchain
df = pd.read_csv("data/data_openalex/timeseries/blockchain_timeseries_monthly.csv")

# Convert the "month" column to datetime (it will be month-end timestamps)
df["Date"] = pd.to_datetime(df["month"])

# Determine which metric to plot
if "normalized_count" in df.columns:
    y = df["normalized_count"]
    y_label = "Academic attention (monthly, normalized)"
elif "count" in df.columns:
    y = df["count"]
    y_label = "Academic publications (monthly)"
else:
    raise ValueError("Expected 'count' or 'normalized_count' in the monthly CSV.")

# ---- Create a plot that matches your GTrends + Wikipedia style ----
fig, ax = plt.subplots(figsize=(10, 6))

ax.plot(df["Date"], y, linewidth=2)

# Labels to match your existing figures
ax.set_xlabel("Date")
ax.set_ylabel(y_label)
ax.set_title("Academic publications over time – Blockchain")

# Grid style matching your other plots
ax.grid(True, linestyle="--", alpha=0.6)

# Set monthly ticks / yearly formatting (same as your GTrends plot)
ax.xaxis.set_major_locator(mdates.YearLocator())
ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
plt.xticks(rotation=45)

plt.tight_layout()
plt.savefig("figures/blockchain_academic_timeseries_monthly.png", dpi=300, bbox_inches="tight")
plt.show()
