"""Data-contract tests: the processed dataset must satisfy documented invariants."""

from pathlib import Path

import pandas as pd
import pytest

from src.config import ANALYSIS_DATASET, CITY_ORDER, LISTINGS_RAW, REVIEWS_RAW

pytestmark = pytest.mark.skipif(
    not ANALYSIS_DATASET.exists(), reason="run `python -m src.build_dataset` first"
)


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    return pd.read_parquet(ANALYSIS_DATASET)


def test_row_counts(df):
    assert len(df) > 270_000
    assert df["listing_id"].is_unique


def test_cities(df):
    assert set(df["city"].astype(str)) == set(CITY_ORDER)


def test_prices_positive_in_analysis_set(df):
    assert (df["price"] > 0).all()
    assert (df["price_usd"] > 0).all()


def test_currency_constant_within_city(df):
    per_city = df.groupby("city", observed=True)["currency"].nunique()
    assert (per_city == 1).all()


def test_review_count_non_negative(df):
    assert (df["review_count"] >= 0).all()
    assert (df["reviewer_count"] <= df["review_count"]).all()


def test_no_review_id_duplicates_assumption(df):
    """The aggregate must be one row per listing, not per review."""
    assert df["listing_id"].is_unique


def test_ratings_within_documented_scale(df):
    rated = df["review_scores_rating"].dropna()
    assert rated.between(0, 100).all()
    sub = df["review_scores_value"].dropna()
    assert sub.between(0, 10).all()


def test_source_files_present():
    assert LISTINGS_RAW.exists()
    assert REVIEWS_RAW.exists()
