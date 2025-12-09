# trends_two_terms.py
from pytrends.request import TrendReq
import pandas as pd

# --- configure here ---
KW1 = "blockchain"          # plain search term
KW2 = "metaverse"           # plain search term
TIMEFRAME = "2010-01-01 2025-12-31"  # 2004→today; you can also use "today 5-y"
GEO = ""                    # "" = worldwide (e.g., "US", "DK" for country)
GPROP = ""                  # "" web search; "images", "news", "youtube", "froogle" (shopping)

# Tip: exact phrase search -> put quotes in the string, e.g. KW1 = '"deep learning"'

def main():
    py = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=3, backoff_factor=0.5)

    # Request both terms in a single payload (max compare ≈ 5)
    py.build_payload([KW1, KW2], timeframe=TIMEFRAME, geo=GEO, gprop=GPROP)

    # Interest over time (weekly or monthly depending on span)
    df = py.interest_over_time().reset_index()

    # Save & print a quick peek
    out_csv = "trends_two_terms.csv"
    df.to_csv(out_csv, index=False)
    print(f"Saved: {out_csv}")
    print(df.head())

    # Optional: yearly aggregation for easy join with your OpenAlex series
    yearly = (
        df.assign(year=pd.to_datetime(df["date"]).dt.year)
          .groupby("year")[[KW1, KW2]].mean().round(2).reset_index()
    )
    yearly_out = "trends_two_terms_yearly.csv"
    yearly.to_csv(yearly_out, index=False)
    print(f"Saved: {yearly_out}")
    print(yearly.head())

if __name__ == "__main__":
    main()
