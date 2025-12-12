from pytrends.request import TrendReq
import pandas as pd
import time


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
    "augmented reality"
]
# static anchor
ANCHOR = "blockchain"
TIMEFRAME = "2015-01-01 2025-12-09"
GEO = ""
GPROP = ""


def fetch_trends_with_anchor(keywords, anchor, timeframe, geo="", gprop="", batch_size=5):
    py = TrendReq(hl="en-US", tz=0, timeout=(10, 30), retries=3, backoff_factor=0.5)
    all_batches = []

    keywords_to_fetch = [kw for kw in keywords if kw.lower() != anchor.lower()]
    for i in range(0, len(keywords_to_fetch), batch_size - 1):
        batch = keywords_to_fetch[i:i + batch_size - 1]
        payload = batch + [anchor]  # include anchor in every batch
        print(f"Fetching batch {i//(batch_size-1) + 1}: {payload}")
        try:
            py.build_payload(payload, timeframe=timeframe, geo=geo, gprop=gprop)
            df = py.interest_over_time().reset_index()
            df["__batch_id__"] = i // (batch_size - 1) + 1
            all_batches.append(df)
            time.sleep(1)
        except Exception as e:
            print(f"Error fetching batch: {e}")
            time.sleep(5)

    return all_batches


def _mean_positive(s):
    s = pd.to_numeric(s, errors="coerce")
    pos = s[s > 0]
    return pos.mean() if not pos.empty else None


def normalize_and_merge_batches(batches, anchor):
    """
    Scale all batches to the first batch using the anchor column.
    For batch k>1: scale = mean(anchor_base) / mean(anchor_k),
    then multiply all non-anchor columns in batch k by 'scale'.
    """
    if not batches:
        return None

    base = batches[0].copy()
    
    if anchor not in base.columns:
        raise ValueError("Anchor not found in first batch; cannot normalize.")

    base_anchor_med = _mean_positive(base[anchor])
    if base_anchor_med in (None, 0):
        raise ValueError("Anchor series in base batch has no positive values; pick a stronger anchor.")

    merged = base.copy()

    for b in batches[1:]:
        if anchor not in b.columns:
            # skip scaling
            scaled = b.copy()
        else:
            batch_anchor_med = _mean_positive(b[anchor])
            if batch_anchor_med in (None, 0):
                # skip scaling for this batch
                scaled = b.copy()
            else:
                scale = float(base_anchor_med) / float(batch_anchor_med)
                scaled = b.copy()
                for c in scaled.columns:
                    if c not in ("date", "isPartial", "__batch_id__", anchor):
                        scaled[c] = pd.to_numeric(scaled[c], errors="coerce").astype(float) * scale

        # drop duplicate anchor before merging
        scaled = scaled.drop(columns=[anchor], errors="ignore")
        merged = merged.merge(scaled, on=["date"], how="outer")

    # cleanup
    merged = merged.drop(columns=["isPartial"], errors="ignore")
    return merged.sort_values("date")


def main():
    batches = fetch_trends_with_anchor(KEYWORDS, ANCHOR, timeframe=TIMEFRAME, geo=GEO, gprop=GPROP)
    if not batches:
        print("No data.")
        return

    result = normalize_and_merge_batches(batches, anchor=ANCHOR)

    out_csv = "../data/data_trends/data_trends.csv"
    result.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")
    print(result.head())

    # yearly aggregation
    result["year"] = pd.to_datetime(result["date"]).dt.year

    # keep only tracked keywords
    cols = [c for c in KEYWORDS if c in result.columns]
    yearly = result.groupby("year")[cols].mean().round(2).reset_index()
    yearly_out = "../data/data_trends/data_trends_yearly.csv"
    yearly.to_csv(yearly_out, index=False)
    print(f"Saved: {yearly_out}")
    print(yearly.head())


if __name__ == "__main__":
    main()
