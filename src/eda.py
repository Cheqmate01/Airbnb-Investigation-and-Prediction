"""Exploratory analysis figures and summary tables for the Data Science track.

Every price chart respects the currency constraint: raw local-currency prices
are shown per city in separate panels, and cross-city comparisons use a
dimensionless *price index* (listing price ÷ that city's median price).
"""

from __future__ import annotations

import matplotlib
import matplotlib.ticker

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from src.config import ANALYSIS_DATASET, CITY_ORDER, FIGURES, REVIEWS_MONTHLY, TABLES

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120
plt.rcParams["axes.titlesize"] = 10


def load() -> pd.DataFrame:
    df = pd.read_parquet(ANALYSIS_DATASET)
    df["city"] = pd.Categorical(df["city"].astype(str), categories=CITY_ORDER, ordered=True)
    df["price_index"] = df["price_usd"] / df.groupby("city", observed=True)["price_usd"].transform("median")
    return df


def save(fig, name: str) -> str:
    path = FIGURES / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def run(df: pd.DataFrame) -> dict[str, str]:
    out: dict[str, str] = {}
    priced = df[df["price_analysable"]].copy()

    # 1. Listings by city ---------------------------------------------------
    counts = df["city"].value_counts().reindex(CITY_ORDER)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    sns.barplot(x=counts.index.astype(str), y=counts.values, ax=ax, color="#4C72B0")
    for i, v in enumerate(counts.values):
        ax.text(i, v + counts.max() * 0.01, f"{v:,}", ha="center", va="bottom", fontsize=8)
    ax.set_ylabel("Listings")
    ax.set_xlabel("")
    ax.set_title("Airbnb listings by city (n = 279,712)")
    ax.tick_params(axis="x", rotation=30)
    out["listings_by_city"] = save(fig, "ds_listings_by_city.png")

    # 2. Price distribution per city, local currency, separate panels -------
    fig, axes = plt.subplots(2, 5, figsize=(16, 6.5), sharey=False)
    for ax, city in zip(axes.ravel(), CITY_ORDER):
        g = priced[priced["city"] == city]
        currency = g["currency"].iloc[0] if len(g) else "?"
        values = g["price"][g["price"] > 0]
        ax.hist(np.log10(values), bins=40, color="#4C72B0", alpha=0.85)
        ax.set_title(f"{city} ({currency})\nmedian {values.median():,.0f}")
        ax.set_xlabel("log10 price (local currency)")
        ax.tick_params(labelsize=7)
    fig.suptitle("Advertised price distributions by city — each panel uses its own local currency", y=1.02)
    out["price_distribution"] = save(fig, "ds_price_distribution_by_city.png")

    # 3. Price index by room type and city (unitless, comparable) -----------
    room_pivot = (
        priced.groupby(["city", "room_type"], observed=True)["price_usd"].median().unstack()
        .reindex(CITY_ORDER)
    )
    city_med = priced.groupby("city", observed=True)["price_usd"].median().reindex(CITY_ORDER)
    index = room_pivot.div(city_med, axis=0)
    fig, ax = plt.subplots(figsize=(7.5, 5))
    sns.heatmap(index, annot=True, fmt=".2f", cmap="RdBu_r", center=1.0, ax=ax,
                cbar_kws={"label": "median price ÷ city median"})
    ax.set_title("Relative price by room type within each city\n(1.00 = city median; unitless, so currencies are comparable)")
    ax.set_xlabel("")
    ax.set_ylabel("")
    out["price_index_heatmap"] = save(fig, "ds_price_roomtype_heatmap.png")

    # 4. Room-type mix by city ---------------------------------------------
    mix = pd.crosstab(df["city"], df["room_type"], normalize="index").reindex(CITY_ORDER) * 100
    fig, ax = plt.subplots(figsize=(9, 4.5))
    mix.plot(kind="bar", stacked=True, ax=ax, colormap="tab20c", width=0.8)
    ax.set_ylabel("% of listings")
    ax.set_xlabel("")
    ax.set_title("Room-type composition by city")
    ax.legend(title="", fontsize=7, ncol=2, loc="upper right")
    ax.tick_params(axis="x", rotation=30)
    out["roomtype_mix"] = save(fig, "ds_roomtype_mix_by_city.png")

    # 5. Property families --------------------------------------------------
    fam = df["property_family"].value_counts().head(12)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    sns.barplot(x=fam.values, y=fam.index, ax=ax, color="#55A868")
    ax.set_xlabel("Listings")
    ax.set_ylabel("")
    ax.set_title("Most common property families")
    out["property_family"] = save(fig, "ds_property_families.png")

    # 6. Review activity over time -----------------------------------------
    monthly = pd.read_parquet(REVIEWS_MONTHLY)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for city, g in monthly.groupby("city"):
        ax.plot(g["month"], g["len"], lw=1, label=city)
    ax.set_ylabel("Reviews per month")
    ax.set_xlabel("")
    ax.set_title("Historical review activity by city (5.37m review records)")
    ax.legend(fontsize=6, ncol=2)
    out["review_activity"] = save(fig, "ds_review_activity_over_time.png")

    # 7. Reviews per listing ------------------------------------------------
    fig, ax = plt.subplots(figsize=(7, 4))
    values = df.loc[df["has_reviews"], "review_count"]
    ax.hist(np.log10(values.clip(lower=1)), bins=45, color="#C44E52", alpha=0.9)
    ax.set_xlabel("log10 reviews per listing (listings with ≥ 1 review)")
    ax.set_ylabel("Listings")
    ax.set_title(f"Review volume is highly skewed: median {values.median():.0f}, max {values.max():,}")
    out["reviews_per_listing"] = save(fig, "ds_reviews_per_listing.png")

    # 8. Ratings by room type ----------------------------------------------
    rated = df.dropna(subset=["review_scores_rating"])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    sns.boxplot(data=rated, x="room_type", y="review_scores_rating", ax=ax, color="#8172B2", showfliers=False)
    ax.set_xlabel("")
    ax.set_ylabel("Listing review score (0–100 scale)")
    ax.set_title(f"Listing-level ratings by room type (rated listings only, n = {len(rated):,} of {len(df):,})")
    out["ratings_roomtype"] = save(fig, "ds_ratings_by_roomtype.png")

    # 9. Host response rate by city ----------------------------------------
    host = df.dropna(subset=["host_response_rate"])
    fig, ax = plt.subplots(figsize=(9, 4.2))
    sns.boxplot(data=host, x="city", y="host_response_rate", ax=ax, color="#64B5CD", showfliers=False)
    ax.set_ylabel("Host response rate")
    ax.set_xlabel("")
    ax.set_title(f"Host response rate by city (available for {len(host):,} of {len(df):,} listings)")
    ax.tick_params(axis="x", rotation=30)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    out["host_response"] = save(fig, "ds_host_response_by_city.png")

    # 10. Capacity and relative price --------------------------------------
    cap = priced.groupby(["city", "accommodates"], observed=True)["price_index"].median().reset_index()
    cap = cap[(cap["accommodates"] >= 1) & (cap["accommodates"] <= 8)]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    sns.lineplot(data=cap, x="accommodates", y="price_index", hue="city", marker="o",
                 markersize=3, lw=1, ax=ax)
    ax.axhline(1.0, color="grey", ls="--", lw=1)
    ax.set_xlabel("Accommodates (guests)")
    ax.set_ylabel("Median price ÷ city median")
    ax.set_title("Relative price rises with capacity in every market (1–8 guests)")
    ax.legend(fontsize=6, ncol=2)
    out["capacity_price"] = save(fig, "ds_capacity_vs_price_index.png")

    # ------------------------------------------------------------------ tables
    summary = (
        priced.groupby("city", observed=True)
        .agg(
            currency=("currency", "first"),
            listings=("listing_id", "size"),
            median_price_local=("price", "median"),
            mean_price_local=("price", "mean"),
            median_price_usd=("price_usd", "median"),
            mean_price_usd=("price_usd", "mean"),
            p90_price_usd=("price_usd", lambda s: s.quantile(0.90)),
            median_accommodates=("accommodates", "median"),
            pct_entire_place=("room_type", lambda s: 100 * (s == "Entire place").mean()),
            median_rating=("review_scores_rating", "median"),
            rated_share=("review_scores_rating", lambda s: 100 * s.notna().mean()),
            median_reviews=("review_count", "median"),
            mean_reviews=("review_count", "mean"),
            pct_with_reviews=("has_reviews", lambda s: 100 * s.mean()),
            median_response_rate=("host_response_rate", "median"),
            mean_response_rate=("host_response_rate", "mean"),
            median_acceptance_rate=("host_acceptance_rate", "median"),
            superhost_share=("host_is_superhost", lambda s: 100 * s.eq(True).mean()),
        )
        .reset_index()
    )
    summary.to_csv(TABLES / "eda_city_summary.csv", index=False)

    room_summary = (
        priced.groupby(["city", "room_type"], observed=True)
        .agg(n=("listing_id", "size"), median_price_local=("price", "median"),
             median_price_usd=("price_usd", "median"), median_rating=("review_scores_rating", "median"))
        .reset_index()
    )
    room_summary.to_csv(TABLES / "eda_roomtype_by_city.csv", index=False)

    review_summary = (
        df.groupby("city", observed=True)
        .agg(
            listings=("listing_id", "size"),
            listings_with_reviews=("has_reviews", "sum"),
            total_reviews=("review_count", "sum"),
            median_reviews=("review_count", "median"),
            mean_reviews=("review_count", "mean"),
            p90_reviews=("review_count", lambda s: s.quantile(0.90)),
        )
        .reset_index()
    )
    review_summary["reviews_per_listing"] = review_summary["total_reviews"] / review_summary["listings"]
    review_summary.to_csv(TABLES / "eda_review_summary.csv", index=False)

    return out


if __name__ == "__main__":
    paths = run(load())
    for k, v in paths.items():
        print(k, "->", v)
