"""Review-file validation and aggregation.

The reviews file holds 5.37 million rows and is processed with Polars' lazy
scanner. Three outputs are produced:

1. ``reviews_by_listing``  - one row per listing with review-history features.
2. ``reviews_monthly_city`` - monthly review counts per city.
3. ``reviews_duplicate_ids`` - evidence about the 160 repeated review IDs.

Important: ``review_id`` is **not** unique in the supplied file, so no
aggregation relies on it. ``(listing_id, review_id)`` is the unique key.
"""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import pandas as pd
import polars as pl

from src.config import REVIEWS_RAW


def _scan(path: Path | str) -> pl.LazyFrame:
    return pl.scan_csv(
        path,
        schema_overrides={
            "listing_id": pl.Int64,
            "review_id": pl.Int64,
            "reviewer_id": pl.Int64,
        },
    ).with_columns(pl.col("date").str.to_date("%Y-%m-%d").alias("date"))


def validate_reviews(path: Path | str = REVIEWS_RAW) -> dict[str, object]:
    """Run structural checks and return a dictionary of findings."""
    lf = _scan(path)

    basic = lf.select(
        pl.len().alias("rows"),
        pl.col("listing_id").n_unique().alias("unique_listings"),
        pl.col("review_id").n_unique().alias("unique_review_ids"),
        pl.col("reviewer_id").n_unique().alias("unique_reviewers"),
        pl.col("date").min().alias("min_date"),
        pl.col("date").max().alias("max_date"),
        pl.col("listing_id").null_count().alias("null_listing_id"),
        pl.col("review_id").null_count().alias("null_review_id"),
        pl.col("date").null_count().alias("null_date"),
        pl.col("reviewer_id").null_count().alias("null_reviewer_id"),
    ).collect().to_dicts()[0]

    key_dupes = (
        lf.group_by(["listing_id", "review_id"]).len().filter(pl.col("len") > 1).collect()
    )
    repeated_ids = (
        lf.group_by("review_id").len().filter(pl.col("len") > 1).collect()
    )
    exact_dupes = (
        lf.group_by(["listing_id", "review_id", "date", "reviewer_id"])
        .len()
        .filter(pl.col("len") > 1)
        .collect()
    )

    basic.update(
        {
            "duplicated_review_id_values": repeated_ids.height,
            "extra_rows_from_repeated_ids": int(repeated_ids["len"].sum() - repeated_ids.height) if repeated_ids.height else 0,
            "duplicated_listing_review_keys": key_dupes.height,
            "exact_duplicate_rows": int(exact_dupes["len"].sum() - exact_dupes.height) if exact_dupes.height else 0,
        }
    )
    return basic


def duplicate_review_evidence(path: Path | str = REVIEWS_RAW) -> pd.DataFrame:
    """Rows involved in a repeated review_id, for the audit trail."""
    lf = _scan(path)
    repeated = lf.group_by("review_id").len().filter(pl.col("len") > 1).select("review_id")
    detail = (
        lf.join(repeated, on="review_id", how="inner")
        .sort(["review_id", "listing_id"])
        .collect()
    )
    return detail.to_pandas()


def build_review_aggregates(
    listings: pd.DataFrame,
    path: Path | str = REVIEWS_RAW,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object]]:
    """Create listing-level and city-month review aggregates."""
    lf = _scan(path)
    snapshot = lf.select(pl.col("date").max()).collect().item()
    cutoff = snapshot - timedelta(days=365)

    by_listing = (
        lf.group_by("listing_id")
        .agg(
            pl.len().alias("review_count"),
            pl.col("reviewer_id").n_unique().alias("reviewer_count"),
            pl.col("date").min().alias("first_review_date"),
            pl.col("date").max().alias("last_review_date"),
            (pl.col("date") >= pl.lit(cutoff)).sum().alias("reviews_last_365d"),
        )
        .with_columns(
            ((pl.col("last_review_date") - pl.col("first_review_date")).dt.total_days() / 30.44)
            .alias("review_span_months"),
        )
        .with_columns(
            pl.when(pl.col("review_span_months") > 0)
            .then(pl.col("review_count") / pl.col("review_span_months"))
            .otherwise(pl.col("review_count").cast(pl.Float64))
            .alias("reviews_per_month")
        )
        .collect()
        .to_pandas()
    )

    city_map = pl.from_pandas(listings[["listing_id", "city"]].assign(city=listings["city"].astype(str)))
    monthly = (
        lf.select("listing_id", "date")
        .join(city_map.lazy(), on="listing_id", how="inner")
        .with_columns(pl.col("date").dt.truncate("1mo").alias("month"))
        .group_by(["city", "month"])
        .len()
        .sort(["city", "month"])
        .collect()
        .to_pandas()
    )

    meta = {"snapshot_date": snapshot.isoformat(), "cutoff_365d": cutoff.isoformat()}
    return by_listing, monthly, meta
