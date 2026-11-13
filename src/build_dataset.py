"""Build the cleaned listings table and the listing-level analysis dataset.

Run order: ``python -m src.build_dataset``

Outputs
-------
data/processed/listings_clean.parquet      every supplied listing, cleaned
data/processed/reviews_by_listing.parquet  review-history features per listing
data/processed/reviews_monthly_city.parquet
data/processed/analysis_dataset.parquet    modelling/analysis ready table
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src.config import (
    ANALYSIS_DATASET,
    FX_SNAPSHOT_DATE,
    INTERIM,
    LISTINGS_CLEAN,
    REVIEWS_BY_LISTING,
    REVIEWS_MONTHLY,
    TABLES,
)
from src.loading import load_listings
from src.reviews import build_review_aggregates

# Modelling convention: prices above the 99th percentile of their own city are
# flagged rather than deleted. The flag lets EDA describe the full market while
# the price model trains on a defensible, comparable sample.
UPPER_QUANTILE = 0.99


def attach_reviews(listings: pd.DataFrame, reviews: pd.DataFrame, snapshot: str) -> pd.DataFrame:
    df = listings.merge(reviews, on="listing_id", how="left", validate="one_to_one")

    df["review_count"] = df["review_count"].fillna(0).astype("int32")
    df["reviewer_count"] = df["reviewer_count"].fillna(0).astype("int32")
    df["reviews_last_365d"] = df["reviews_last_365d"].fillna(0).astype("int32")
    df["has_reviews"] = df["review_count"] > 0

    snap = pd.Timestamp(snapshot)
    df["days_since_last_review"] = (snap - pd.to_datetime(df["last_review_date"])).dt.days
    df["listing_age_years"] = (snap - pd.to_datetime(df["first_review_date"])).dt.days / 365.25
    df["review_activity_per_year"] = np.where(
        df["listing_age_years"] > 0,
        df["review_count"] / df["listing_age_years"],
        np.nan,
    )
    df["reviews_per_month"] = df["reviews_per_month"].fillna(0.0)
    return df


def add_price_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add per-city price-distribution flags used by EDA and modelling."""
    grouped = df.groupby("city", observed=True)["price_usd"]
    df["city_price_p01"] = grouped.transform(lambda s: s.quantile(0.01))
    df["city_price_p50"] = grouped.transform("median")
    df["city_price_p99"] = grouped.transform(lambda s: s.quantile(UPPER_QUANTILE))

    df["price_outlier_high"] = df["price_usd"] > df["city_price_p99"]
    df["price_outlier_low"] = df["price_usd"] < df["city_price_p01"]
    df["price_analysable"] = df["price_valid"] & ~df["price_outlier_high"]
    df["price_above_city_median"] = df["price_usd"] > df["city_price_p50"]
    return df


def main() -> None:
    listings = load_listings()
    reviews, monthly, meta = build_review_aggregates(listings)
    snapshot = meta["snapshot_date"]

    listings["host_tenure_years"] = (
        pd.Timestamp(snapshot) - listings["host_since"]
    ).dt.days / 365.25

    df = attach_reviews(listings, reviews, snapshot)
    df = add_price_flags(df)

    listings.to_parquet(INTERIM / "listings_clean_raw.parquet", index=False)
    reviews.to_parquet(REVIEWS_BY_LISTING, index=False)
    monthly.to_parquet(REVIEWS_MONTHLY, index=False)
    df.to_parquet(LISTINGS_CLEAN, index=False)

    # The analysis dataset keeps every listing that has a usable positive price
    # so descriptive analysis is not silently censored; the `price_analysable`
    # flag controls the modelling sample.
    analysis = df[df["price_valid"]].copy()
    analysis.to_parquet(ANALYSIS_DATASET, index=False)

    summary = {
        "snapshot_date": snapshot,
        "fx_snapshot_date": FX_SNAPSHOT_DATE,
        "listings_rows": int(len(df)),
        "analysis_rows": int(len(analysis)),
        "price_analysable_rows": int(df["price_analysable"].sum()),
        "price_nonpositive_rows": int(df["price_nonpositive"].sum()),
        "price_outlier_high_rows": int(df["price_outlier_high"].sum()),
        "listings_with_reviews": int(df["has_reviews"].sum()),
        "review_rows_aggregated": int(reviews["review_count"].sum()),
    }
    (TABLES / "build_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
