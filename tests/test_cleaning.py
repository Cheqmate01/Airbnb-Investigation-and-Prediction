"""Unit tests for cleaning helpers and the currency configuration."""

import math

import pandas as pd
import pytest

from src.config import CITY_CURRENCY, usd_per_unit
from src.loading import parse_boolean, parse_percentage, property_family, repair_mojibake


def test_repair_mojibake():
    assert repair_mojibake("cafÃ©") == "café"
    assert repair_mojibake("plain text") == "plain text"
    assert repair_mojibake(None) is None


def test_parse_percentage():
    s = pd.Series(["97%", "100%", None, "0.5", "50"])
    out = parse_percentage(s)
    assert out.iloc[0] == pytest.approx(0.97)
    assert out.iloc[1] == pytest.approx(1.0)
    assert math.isnan(out.iloc[2])
    assert out.iloc[3] == pytest.approx(0.5)
    assert out.iloc[4] == pytest.approx(0.5)


def test_parse_boolean():
    out = parse_boolean(pd.Series(["t", "f", None, "t"]))
    assert out.tolist()[:2] == [True, False]
    assert pd.isna(out.iloc[2])
    assert bool(out.iloc[3]) is True


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("Entire apartment", "apartment"),
        ("Private room in apartment", "apartment"),
        ("Private room in house", "house"),
        ("Entire serviced apartment", "apartment"),
        ("Room in boutique hotel", "hotel"),
        ("Entire condominium", "condominium"),
        ("Entire loft", "loft"),
        ("Entire villa", "villa"),
        (None, "unknown"),
        ("", "unknown"),
        ("Entire castle", "castle"),
    ],
)
def test_property_family(raw, expected):
    assert property_family(raw) == expected


def test_currency_mapping_covers_ten_cities():
    assert len(CITY_CURRENCY) == 10
    assert CITY_CURRENCY["Paris"] == "EUR"
    assert CITY_CURRENCY["New York"] == "USD"


def test_usd_per_unit():
    assert usd_per_unit("USD") == pytest.approx(1.0)
    assert usd_per_unit("EUR") == pytest.approx(1.1127)
    assert usd_per_unit("TRY") < 0.05
    assert usd_per_unit("AUD") == pytest.approx(1.1127 / 1.6222)
