import requests
import pandas as pd
from urllib.parse import quote

CONTACT_EMAIL = "hanka.dubovska@gmail.com"

# Time window
START_DATE = "20150701"
END_DATE = "20251130"


BUZZWORD_TO_ARTICLE = {
    "Artificial intelligence": "Artificial_intelligence",
    "Cryptocurrency": "Cryptocurrency",
    "Quantum computing": "Quantum_computing",
    "Blockchain": "Blockchain",
    "Edge computing": "Edge_computing",
    "Metaverse": "Metaverse",
    "Chatbot": "Chatbot",
    "Augmented reality": "Augmented_reality",
    "Big data": "Big_data",
    "Generative artificial intelligence": "Generative_artificial_intelligence",
}


def get_wiki_pageviews(article_title: str,
                       start: str = START_DATE,
                       end: str = END_DATE) -> pd.DataFrame:

    encoded_title = quote(article_title.replace(" ", "_"))

    url = (
        "https://wikimedia.org/api/rest_v1/metrics/pageviews/per-article/"
        f"en.wikipedia.org/all-access/all-agents/{encoded_title}/monthly/"
        f"{start}/{end}"
    )

    headers = {
        "User-Agent": f"ITU-Student-Project/1.0 ({CONTACT_EMAIL})"
    }

    response = requests.get(url, headers=headers)
    response.raise_for_status()
    data = response.json()

    records = []
    for item in data.get("items", []):
        ts = item["timestamp"]  # e.g. '2015010100'
        year = int(ts[:4])
        month = int(ts[4:6])

        records.append(
            {
                "article": article_title,
                "year": year,
                "month": month,
                "views": item["views"],
            }
        )

    return pd.DataFrame(records)


def fetch_all_buzzwords() -> pd.DataFrame:
    all_monthly = []

    for buzzword, article_title in BUZZWORD_TO_ARTICLE.items():
        print(f"Fetching data for '{buzzword}' ({article_title})...")
        df = get_wiki_pageviews(article_title)
        df["buzzword"] = buzzword
        all_monthly.append(df)

    if not all_monthly:
        raise RuntimeError("No data fetched for any buzzword.")

    monthly_df = pd.concat(all_monthly, ignore_index=True)
    return monthly_df


def aggregate_yearly(monthly_df: pd.DataFrame) -> pd.DataFrame:
    yearly_df = (
        monthly_df
        .groupby(["buzzword", "year"], as_index=False)["views"]
        .sum()
        .sort_values(["buzzword", "year"])
    )
    return yearly_df



if __name__ == "__main__":
    wiki_monthly = fetch_all_buzzwords()
    print("\nMonthly data (head):")
    print(wiki_monthly.head())

    wiki_yearly = aggregate_yearly(wiki_monthly)
    print("\nYearly data (head):")
    print(wiki_yearly.head())

    wiki_monthly.to_csv("data/data_wiki/wiki_pageviews_monthly.csv", index=False)
    wiki_yearly.to_csv("data/data_wiki/wiki_pageviews_yearly.csv", index=False)

    print("\nSaved:")
    print(" - wiki_pageviews_monthly.csv")
    print(" - wiki_pageviews_yearly.csv")


import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter, YearLocator

# Use the monthly file instead of the yearly one
wiki_monthly = pd.read_csv("wiki_pageviews_monthly.csv")

# Build a proper datetime column from year + month
wiki_monthly["date"] = pd.to_datetime(
    wiki_monthly["year"].astype(int).astype(str) + "-" +
    wiki_monthly["month"].astype(int).astype(str) + "-01"
)

# Sort by buzzword + date
wiki_monthly = wiki_monthly.sort_values(["buzzword", "date"])

for buzz in wiki_monthly["buzzword"].unique():
    sub = wiki_monthly[wiki_monthly["buzzword"] == buzz]

    plt.figure()
    plt.plot(sub["date"], sub["views"])  # no marker for smoother line
    plt.title(f"Wikipedia pageviews over time – {buzz}")
    plt.xlabel("Date")
    plt.ylabel("Pageviews (monthly)")
    plt.grid(True)

    # Make the x-axis show years like the left chart
    ax = plt.gca()
    ax.xaxis.set_major_locator(YearLocator())
    ax.xaxis.set_major_formatter(DateFormatter("%b %Y"))
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig(f"figures/wiki_trend_monthly_{buzz.replace(' ', '_')}.png", dpi=150)


plt.show()
