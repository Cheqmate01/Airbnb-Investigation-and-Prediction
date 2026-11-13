"""
Reading and cleaning the supplied Airbnb listings data.

Design notes
------------
* The supplied `Listings.csv` is **not** valid UTF-8 throughout: it mixes UTF-8 content with a small number of stray bytes (984 invalid sequences). Reading with ``encoding="latin-1"`` never fails, and a bounded mojibake repair recovers the intended characters for the text fields we use.
* Percentages arrive as strings (``"100%"``) and booleans as ``"t"``/``"f"``,  so both are normalised explicitly rather than left to type inference.
* No rows are dropped here. Invalid or implausible prices are flagged so that the downstream modelling code can make an explicit, documented choice.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from src.config import CITY_CURRENCY, CITY_ORDER, LISTINGS_RAW, usd_per_unit

RAW_ENCODING = "latin-1"


# --------------------------------------------------------------------------
# Text repair
# --------------------------------------------------------------------------
def repair_mojibake(value: object) -> object:
    """Undo up to two rounds of UTF-8 -> Latin-1 mojibake on a string."""
    if not isinstance(value, str):
        return value
    current = value
    for _ in range(2):
        try:
            candidate = current.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            break
        if candidate == current:
            break
        current = candidate
    return current


# --------------------------------------------------------------------------
# Value normalisation
# --------------------------------------------------------------------------
def parse_percentage(series: pd.Series) -> pd.Series:
    """Convert strings such as ``"97%"`` into proportions in [0, 1]."""
    cleaned = (
        series.astype("string")
        .str.strip()
        .str.replace("%", "", regex=False)
        .str.replace(",", "", regex=False)
    )
    numeric = pd.to_numeric(cleaned, errors="coerce").astype("float64")
    # Defensive: a value already expressed as a proportion stays as-is.
    return (numeric / 100.0).where(numeric > 1.0, numeric)


def parse_boolean(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip().str.lower().map({"t": True, "f": False})


def _extract_amenity_list(value: object) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    try:
        parsed = json.loads(value)
        if isinstance(parsed, list):
            return [str(item).strip() for item in parsed]
    except json.JSONDecodeError:
        # Fall back to a tolerant split on commas inside quotes / brackets.
        inner = value.strip().strip("[]")
        return [part.strip().strip('"') for part in inner.split(",") if part.strip()]
    return []


AMENITY_FLAGS = {
    "has_wifi": ("wifi", "internet"),
    "has_kitchen": ("kitchen",),
    "has_air_conditioning": ("air conditioning", "air-conditioning"),
    "has_washer": ("washer", "washing machine"),
    "has_heating": ("heating",),
    "has_parking": ("parking",),
    "has_pool": ("pool",),
    "has_gym": ("gym",),
    "has_elevator": ("elevator", "lift"),
    "has_tv": ("tv", "television"),
    "has_long_term_stay": ("long term stays allowed", "long-term stays allowed"),
}


_PROPERTY_SYNONYMS = {
    "serviced apartment": "apartment",
    "apartment": "apartment",
    "condominium": "condominium",
    "condo": "condominium",
    "house": "house",
    "townhouse": "house",
    "loft": "loft",
    "villa": "villa",
    "cottage": "cottage",
    "bungalow": "bungalow",
    "cabin": "cabin",
    "chalet": "chalet",
    "guesthouse": "guesthouse",
    "guest suite": "guesthouse",
    "bed and breakfast": "bed and breakfast",
    "hostel": "hostel",
    "hotel": "hotel",
    "boutique hotel": "hotel",
    "resort": "resort",
    "home": "house",
    "place": "apartment",
    "earth house": "house",
    "dome house": "house",
    "tiny house": "house",
    "boat": "boat",
    "camper": "camper",
    "tent": "tent",
    "castle": "castle",
    "treehouse": "treehouse",
    "barn": "barn",
    "farm stay": "farm stay",
    "island": "island",
    "lighthouse": "lighthouse",
    "cave": "cave",
    "yurt": "yurt",
    "train": "train",
    "plane": "plane",
    "parking space": "parking space",
    "entire floor": "apartment",
    "private room": "room",
}


def property_family(value: object) -> str:
    """Reduce 144 raw property types to a small, interpretable family."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    text = value.lower().strip()
    text = re.sub(r"^(entire|private room in|shared room in|room in)\s+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    for key in sorted(_PROPERTY_SYNONYMS, key=len, reverse=True):
        if text == key or text.endswith(" " + key) or text.startswith(key + " "):
            return _PROPERTY_SYNONYMS[key]
    return "other"


RESPONSE_TIME_ORDER = {
    "within an hour": 1,
    "within a few hours": 2,
    "within a day": 3,
    "a few days or more": 4,
}


# --------------------------------------------------------------------------
# Loader
# --------------------------------------------------------------------------
LISTING_COLUMNS: list[str] = [
    "listing_id", "name", "host_id", "host_since", "host_location",
    "host_response_time", "host_response_rate", "host_acceptance_rate",
    "host_is_superhost", "host_total_listings_count", "host_has_profile_pic",
    "host_identity_verified", "neighbourhood", "district", "city",
    "latitude", "longitude", "property_type", "room_type", "accommodates",
    "bedrooms", "amenities", "price", "minimum_nights", "maximum_nights",
    "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
    "review_scores_checkin", "review_scores_communication",
    "review_scores_location", "review_scores_value", "instant_bookable",
]


def load_listings(path: Path | str = LISTINGS_RAW) -> pd.DataFrame:
    """Read and clean the listings file without dropping any rows."""
    df = pd.read_csv(
        path,
        encoding=RAW_ENCODING,
        dtype={"listing_id": "int64", "host_id": "int64"},
        low_memory=False,
    )
    assert list(df.columns) == LISTING_COLUMNS, "Unexpected listings schema"

    # --- text fields -------------------------------------------------------
    for col in ["name", "host_location", "neighbourhood", "district", "city", "property_type", "amenities"]:
        df[col] = df[col].map(repair_mojibake).astype("string")

    # --- dates & booleans --------------------------------------------------
    df["host_since"] = pd.to_datetime(df["host_since"], errors="coerce")
    for col in ["host_is_superhost", "host_has_profile_pic", "host_identity_verified", "instant_bookable"]:
        df[col] = parse_boolean(df[col])

    # --- percentages -------------------------------------------------------
    df["host_response_rate"] = parse_percentage(df["host_response_rate"])
    df["host_acceptance_rate"] = parse_percentage(df["host_acceptance_rate"])

    # --- host response time as an ordinal --------------------------------
    df["host_response_time"] = df["host_response_time"].astype("string").str.strip().str.lower()
    df["host_response_time_rank"] = df["host_response_time"].map(RESPONSE_TIME_ORDER)

    # --- numeric coercion --------------------------------------------------
    for col in [
        "latitude", "longitude", "accommodates", "bedrooms", "price",
        "minimum_nights", "maximum_nights", "host_total_listings_count",
        "review_scores_rating", "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication",
        "review_scores_location", "review_scores_value",
    ]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # --- currency & USD standardisation -----------------------------------
    unknown = set(df["city"].unique()) - set(CITY_CURRENCY)
    if unknown:
        raise ValueError(f"Cities without a currency mapping: {sorted(unknown)}")
    df["currency"] = df["city"].map(CITY_CURRENCY)
    df["fx_usd_per_unit"] = df["currency"].map(usd_per_unit).astype(float)
    df["price_usd"] = df["price"] * df["fx_usd_per_unit"]

    # --- price validity flags ---------------------------------------------
    df["price_missing"] = df["price"].isna()
    df["price_nonpositive"] = df["price"].notna() & (df["price"] <= 0)
    df["price_valid"] = df["price"].notna() & (df["price"] > 0)

    # --- amenities ---------------------------------------------------------
    amenity_lists = df["amenities"].map(_extract_amenity_list)
    df["amenity_count"] = amenity_lists.map(len).astype("int16")
    lowered = amenity_lists.map(lambda items: " | ".join(items).lower())
    for flag, needles in AMENITY_FLAGS.items():
        pattern = "|".join(re.escape(n) for n in needles)
        df[flag] = lowered.str.contains(pattern, regex=True, na=False).astype(bool)

    # --- property / room characteristics ----------------------------------
    df["property_family"] = df["property_type"].map(property_family)
    df["bedrooms_missing"] = df["bedrooms"].isna()

    # Host tenure is filled once the dataset snapshot date is known; see
    # build_dataset.py. We still provide it here relative to the listing data.
    snapshot = pd.Timestamp("2026-11-11")
    df["host_tenure_years"] = (snapshot - df["host_since"]).dt.days / 365.25

    df["city"] = pd.Categorical(df["city"], categories=CITY_ORDER, ordered=True)
    return df


def load_reviews(path: Path | str) -> pd.DataFrame:
    """Load the reviews file as pandas (used only for small checks)."""
    return pd.read_csv(
        path,
        encoding=RAW_ENCODING,
        dtype={"listing_id": "int64", "review_id": "int64", "reviewer_id": "int64"},
        parse_dates=["date"],
    )


def currency_reference_table() -> pd.DataFrame:
    from src.config import ECB_RATES_PER_EUR

    rows = []
    for city in CITY_ORDER:
        currency = CITY_CURRENCY[city]
        rows.append(
            {
                "city": city,
                "currency": currency,
                "units_per_eur": ECB_RATES_PER_EUR[currency],
                "usd_per_unit": usd_per_unit(currency),
            }
        )
    return pd.DataFrame(rows)
