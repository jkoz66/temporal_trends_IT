import pandas as pd

# ---------- MONTHLY ----------

# this function reshapes the google trends data to uniformly process data from other sources
def reshape_monthly(
    in_path="./data/data_trends/data_trends.csv",
    out_path="./data/data_trends/data_trends_monthly_long.csv",
):
    df = pd.read_csv(in_path)

    # ensure date is datetime and extract
    df["date"] = pd.to_datetime(df["date"])
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month

    # columns that are not keywords
    helper_cols = {
        "date",
        "isPartial_x",
        "isPartial_y",
        "__batch_id___x",
        "__batch_id___y",
        "__batch_id__",
        "year",
        "month",
    }

    keyword_cols = [c for c in df.columns if c not in helper_cols]

    # wide to long
    long_df = df.melt(
        id_vars=["year", "month"],
        value_vars=keyword_cols,
        var_name="buzzword",
        value_name="interest",
    )

    # lowercase all buzzwords
    long_df["buzzword"] = long_df["buzzword"].str.lower()

    long_df = long_df[["buzzword", "year", "month", "interest"]]

    long_df.to_csv(out_path, index=False)
    print(f"Saved monthly long data to: {out_path}")
    print(long_df.head())


# ---------- YEARLY ----------

def reshape_yearly(
    in_path="./data/data_trends/data_trends_yearly.csv",
    out_path="./data/data_trends/data_trends_yearly_long.csv",
):
    df = pd.read_csv(in_path)

    keyword_cols = [c for c in df.columns if c != "year"]

    long_df = df.melt(
        id_vars=["year"],
        value_vars=keyword_cols,
        var_name="buzzword",
        value_name="interest",
    )

    # lowercase all buzzwords
    long_df["buzzword"] = long_df["buzzword"].str.lower()

    long_df = long_df[["buzzword", "year", "interest"]]

    long_df.to_csv(out_path, index=False)
    print(f"Saved yearly long data to: {out_path}")
    print(long_df.head())


if __name__ == "__main__":
    reshape_monthly()
    reshape_yearly()
