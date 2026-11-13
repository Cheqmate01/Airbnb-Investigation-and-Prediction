"""
Project-wide configuration: paths, city-currency mapping and FX assumptions.

All paths are resolved relative to the project root so that every script and notebook behaves identically regardless of the working directory it is launched from.
"""

from __future__ import annotations

from pathlib import Path

# --------------------------------------------------------------------------
# Paths
# --------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "Datasets"
DATA = ROOT / "data"
INTERIM = DATA / "interim"
PROCESSED = DATA / "processed"
OUTPUTS = ROOT / "outputs"
FIGURES = OUTPUTS / "figures"
TABLES = OUTPUTS / "tables"
MODELS = OUTPUTS / "models"
REPORTS = ROOT / "reports"
DOCS = ROOT / "docs"

for _p in (INTERIM, PROCESSED, FIGURES, TABLES, MODELS, REPORTS):
    _p.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Reproducibility
# --------------------------------------------------------------------------
RANDOM_STATE = 42
TEST_SIZE = 0.20

# --------------------------------------------------------------------------
# Currency
# --------------------------------------------------------------------------
# Listings.csv carries no currency column: prices are expressed in the local currency of the market the listing belongs to. This mapping is derived from the city of each listing.
CITY_CURRENCY: dict[str, str] = {
    "Paris": "EUR",
    "Rome": "EUR",
    "New York": "USD",
    "Sydney": "AUD",
    "Rio de Janeiro": "BRL",
    "Istanbul": "TRY",
    "Mexico City": "MXN",
    "Bangkok": "THB",
    "Cape Town": "ZAR",
    "Hong Kong": "HKD",
}

# European Central Bank euro foreign-exchange reference rates. Rates are quoted as units of foreign currency per 1 EUR.
ECB_RATES_PER_EUR: dict[str, float] = {
    "USD": 1.1127,
    "TRY": 55.8897,
    "AUD": 1.6222,
    "BRL": 6.1827,
    "HKD": 8.7298,
    "MXN": 20.7623,
    "THB": 37.4870,
    "ZAR": 18.4307,
    "EUR": 1.0,
}

FX_SNAPSHOT_DATE = "2026-11-11"
FX_SOURCE = (
    "European Central Bank euro foreign-exchange reference rates, "
    f"snapshot {FX_SNAPSHOT_DATE} (docs/ecb_reference_rates_2026-11-11.xml)"
)


def usd_per_unit(currency: str) -> float:
    """USD value of one unit of `currency` on the FX snapshot date."""
    if currency not in ECB_RATES_PER_EUR:
        raise KeyError(f"No ECB reference rate recorded for {currency!r}")
    return ECB_RATES_PER_EUR["USD"] / ECB_RATES_PER_EUR[currency]


# Convenience table used in documentation and the audit.
FX_TO_USD: dict[str, float] = {
    c: round(usd_per_unit(c), 6) for c in set(CITY_CURRENCY.values())
}

# --------------------------------------------------------------------------
# Analysis conventions
# --------------------------------------------------------------------------
# Review scores are documented out of 100 for the overall rating and out of 10 for the sub-scores.
SCORE_FIELDS = [
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
]

CITY_ORDER = [
    "Paris",
    "New York",
    "Sydney",
    "Rome",
    "Rio de Janeiro",
    "Istanbul",
    "Mexico City",
    "Bangkok",
    "Cape Town",
    "Hong Kong",
]

# File names
LISTINGS_RAW = RAW / "Listings.csv"
REVIEWS_RAW = RAW / "Reviews.csv"
LISTINGS_DICT = RAW / "Listings_data_dictionary.csv"
REVIEWS_DICT = RAW / "Reviews_data_dictionary.csv"

LISTINGS_CLEAN = PROCESSED / "listings_clean.parquet"
REVIEWS_BY_LISTING = PROCESSED / "reviews_by_listing.parquet"
REVIEWS_MONTHLY = PROCESSED / "reviews_monthly_city.parquet"
ANALYSIS_DATASET = PROCESSED / "analysis_dataset.parquet"
