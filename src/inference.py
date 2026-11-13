"""Load the packaged price model and score new listings.

Example
-------
>>> from src.inference import predict_price
>>> predict_price({
...     "city": "Rome", "room_type": "Entire place", "property_family": "apartment",
...     "accommodates": 4, "bedrooms": 2, "minimum_nights": 2, "maximum_nights": 365,
...     "amenity_count": 20, "latitude": 41.9028, "longitude": 12.4964,
...     "host_total_listings_count": 3, "host_tenure_years": 6,
...     "host_response_rate": 0.98, "host_acceptance_rate": 0.9,
...     "host_response_time_rank": 1, "host_is_superhost": 1,
...     "host_has_profile_pic": 1, "host_identity_verified": 1, "instant_bookable": 1,
... })

The function validates required fields, returns USD plus the local-currency
equivalent at the documented FX snapshot, and never raises on missing optional
values (they are imputed by the pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from src.config import MODELS, usd_per_unit
from src.features import CORE_CATEGORICAL, CORE_NUMERIC

DEFAULT_MODEL_PATH = MODELS / "price_model_core.joblib"
METADATA_PATH = MODELS / "model_metadata.json"

REQUIRED_FIELDS = {"city", "room_type"}
VALID_CITIES = {
    "Paris", "Rome", "New York", "Sydney", "Rio de Janeiro", "Istanbul",
    "Mexico City", "Bangkok", "Cape Town", "Hong Kong",
}
CURRENCY_BY_CITY = {
    "Paris": "EUR", "Rome": "EUR", "New York": "USD", "Sydney": "AUD",
    "Rio de Janeiro": "BRL", "Istanbul": "TRY", "Mexico City": "MXN",
    "Bangkok": "THB", "Cape Town": "ZAR", "Hong Kong": "HKD",
}


def load_model(path: Path | str = DEFAULT_MODEL_PATH):
    return joblib.load(path)


def validate_input(record: dict) -> dict:
    missing = REQUIRED_FIELDS - set(record)
    if missing:
        raise ValueError(f"Missing required field(s): {sorted(missing)}")
    city = record["city"]
    if city not in VALID_CITIES:
        raise ValueError(f"Unsupported city {city!r}. Expected one of {sorted(VALID_CITIES)}")
    room_type = record["room_type"]
    if not isinstance(room_type, str) or not room_type.strip():
        raise ValueError("room_type must be a non-empty string")
    for key in ("accommodates", "bedrooms", "minimum_nights"):
        if key in record and record[key] is not None and record[key] < 0:
            raise ValueError(f"{key} must be non-negative")
    return record


def predict_price(record: dict, model=None, model_path: Path | str = DEFAULT_MODEL_PATH) -> dict:
    """Predict the advertised nightly price for one listing."""
    record = validate_input(dict(record))
    model = model or load_model(model_path)

    row = {col: record.get(col, np.nan) for col in CORE_NUMERIC}
    for col in CORE_CATEGORICAL:
        row[col] = record.get(col, "unknown")
    frame = pd.DataFrame([row])

    log_pred = float(model.predict(frame)[0])
    price_usd = float(np.expm1(log_pred))
    price_usd = max(price_usd, 0.0)
    currency = CURRENCY_BY_CITY[record["city"]]
    local = price_usd / usd_per_unit(currency)

    metadata = {}
    if Path(METADATA_PATH).exists():
        metadata = json.loads(Path(METADATA_PATH).read_text())
        metadata = {
            "model_family": metadata.get("model_family"),
            "currency_assumption": metadata.get("currency_assumption"),
        }

    return {
        "predicted_price_usd": round(price_usd, 2),
        "predicted_price_local": round(local, 2),
        "local_currency": currency,
        "basis": "advertised nightly price (not revenue, occupancy or profit)",
        **metadata,
    }
