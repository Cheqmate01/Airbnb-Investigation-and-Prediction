"""Inference tests: validation, determinism and currency conversion."""

import pytest

from src.config import MODELS, usd_per_unit
from src.inference import predict_price

pytestmark = pytest.mark.skipif(
    not (MODELS / "price_model_core.joblib").exists(),
    reason="run `python -m src.modelling` first",
)

BASE = {
    "city": "Rome",
    "room_type": "Entire place",
    "property_family": "apartment",
    "accommodates": 4,
    "bedrooms": 2,
    "minimum_nights": 2,
    "maximum_nights": 365,
    "amenity_count": 20,
    "latitude": 41.9028,
    "longitude": 12.4964,
    "host_total_listings_count": 3,
    "host_tenure_years": 6,
    "host_response_rate": 0.98,
    "host_acceptance_rate": 0.9,
    "host_response_time_rank": 1,
    "host_is_superhost": 1,
    "host_has_profile_pic": 1,
    "host_identity_verified": 1,
    "instant_bookable": 1,
}


def test_prediction_is_positive():
    result = predict_price(BASE)
    assert result["predicted_price_usd"] > 0
    assert result["predicted_price_local"] > 0
    assert result["local_currency"] == "EUR"


def test_local_currency_conversion_is_consistent():
    result = predict_price(BASE)
    expected = result["predicted_price_usd"] / usd_per_unit("EUR")
    assert result["predicted_price_local"] == pytest.approx(expected, rel=1e-3)


def test_deterministic():
    assert predict_price(BASE) == predict_price(BASE)


def test_city_changes_prediction():
    rome = predict_price(BASE)["predicted_price_usd"]
    bangkok = predict_price({**BASE, "city": "Bangkok"})["predicted_price_usd"]
    assert rome != bangkok


def test_rejects_unknown_city():
    with pytest.raises(ValueError, match="Unsupported city"):
        predict_price({**BASE, "city": "Atlantis"})


def test_rejects_missing_required_field():
    with pytest.raises(ValueError, match="Missing required field"):
        predict_price({"city": "Rome"})


def test_rejects_negative_capacity():
    with pytest.raises(ValueError, match="non-negative"):
        predict_price({**BASE, "accommodates": -2})
