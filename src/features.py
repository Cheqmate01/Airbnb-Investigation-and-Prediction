"""Feature definitions and preprocessing for the price model.

Two feature sets are defined deliberately:

``CORE``
    Attributes a host can supply when a listing is created: location, property
    type, capacity, policy, amenities and host profile. This is the deployable
    model and contains no information that only accrues after the listing has
    been live.

``EXTENDED``
    CORE plus review-history and rating fields. These exist for most mature
    listings but cannot be known for a brand-new listing, so they are tested as
    a sensitivity analysis rather than presented as the production feature set.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

AMENITY_FEATURES = [
    "has_wifi", "has_kitchen", "has_air_conditioning", "has_washer", "has_heating",
    "has_parking", "has_pool", "has_gym", "has_elevator", "has_tv",
    "has_long_term_stay",
]

CORE_NUMERIC: list[str] = [
    "accommodates",
    "bedrooms",
    "bedrooms_missing",
    "minimum_nights",
    "maximum_nights",
    "amenity_count",
    "latitude",
    "longitude",
    "host_total_listings_count",
    "host_tenure_years",
    "host_response_rate",
    "host_acceptance_rate",
    "host_response_rate_missing",
    "host_acceptance_rate_missing",
    "host_response_time_rank",
    "host_is_superhost",
    "host_has_profile_pic",
    "host_identity_verified",
    "instant_bookable",
    "is_multilisting_host",
    *AMENITY_FEATURES,
]

CORE_CATEGORICAL: list[str] = ["city", "room_type", "property_family"]

REVIEW_NUMERIC: list[str] = [
    "review_count",
    "reviewer_count",
    "reviews_last_365d",
    "days_since_last_review",
    "listing_age_years",
    "review_activity_per_year",
    "reviews_per_month",
    "review_scores_rating",
    "review_scores_accuracy",
    "review_scores_cleanliness",
    "review_scores_checkin",
    "review_scores_communication",
    "review_scores_location",
    "review_scores_value",
    "has_reviews",
]

EXTENDED_NUMERIC = CORE_NUMERIC + REVIEW_NUMERIC


def prepare_model_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Add modelling-only derived fields and the USD target."""
    out = df.copy()
    out["host_response_rate_missing"] = out["host_response_rate"].isna().astype(np.int8)
    out["host_acceptance_rate_missing"] = out["host_acceptance_rate"].isna().astype(np.int8)
    out["is_multilisting_host"] = (out["host_total_listings_count"] > 1).astype(np.int8)
    out["target_log_usd"] = np.log1p(out["price_usd"])
    for col in CORE_CATEGORICAL:
        out[col] = out[col].astype("string").fillna("unknown").astype("category")
    for col in CORE_NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    for col in REVIEW_NUMERIC:
        out[col] = pd.to_numeric(out[col], errors="coerce").astype("float32")
    return out


class MedianCategoryPrep(BaseEstimator, TransformerMixin):
    """Median-impute numeric columns and freeze categorical categories.

    Returning a DataFrame (rather than a NumPy array) preserves ``category``
    dtype so ``HistGradientBoostingRegressor`` can use native categorical
    handling without a second encoding step.
    """

    def __init__(self, numeric: Sequence[str], categorical: Sequence[str]):
        self.numeric = numeric
        self.categorical = categorical

    def fit(self, X: pd.DataFrame, y=None):
        self.numeric_ = list(self.numeric)
        self.categorical_ = list(self.categorical)
        num = X[self.numeric_].apply(pd.to_numeric, errors="coerce")
        self.medians_ = num.median()
        self.categories_ = {}
        for col in self.categorical_:
            values = pd.Series(X[col]).astype("string").fillna("unknown").unique().tolist()
            self.categories_[col] = sorted(values)
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        out = X[self.numeric_ + self.categorical_].copy()
        for col in self.numeric_:
            values = pd.to_numeric(out[col], errors="coerce")
            out[col] = values.fillna(self.medians_.get(col, np.nan))
        for col in self.categorical_:
            values = pd.Series(out[col]).astype("string").fillna("unknown")
            out[col] = pd.Categorical(values, categories=self.categories_[col])
        return out


def make_pipeline(kind: str, numeric: Sequence[str], categorical: Sequence[str]) -> Pipeline:
    prep = MedianCategoryPrep(numeric, categorical)
    if kind == "ridge":
        encoder = ColumnTransformer(
            [
                ("num", Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler())]), make_column_selector(dtype_include=np.number)),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=True), make_column_selector(dtype_include="category")),
            ],
            sparse_threshold=1.0,
        )
        return Pipeline([("prep", prep), ("encode", encoder), ("model", Ridge(alpha=1.0))])

    if kind == "forest":
        encoder = ColumnTransformer(
            [
                ("num", SimpleImputer(strategy="median"), make_column_selector(dtype_include=np.number)),
                ("cat", OneHotEncoder(handle_unknown="ignore", min_frequency=10, sparse_output=False), make_column_selector(dtype_include="category")),
            ],
            sparse_threshold=0.0,
        )
        model = RandomForestRegressor(
            n_estimators=120,
            min_samples_leaf=10,
            max_features=0.5,
            max_samples=0.8,
            n_jobs=2,
            random_state=42,
        )
        return Pipeline([("prep", prep), ("encode", encoder), ("model", model)])

    if kind == "hist_gb":
        model = HistGradientBoostingRegressor(
            max_iter=400,
            learning_rate=0.06,
            max_leaf_nodes=31,
            min_samples_leaf=40,
            l2_regularization=1.0,
            early_stopping=True,
            validation_fraction=0.1,
            n_iter_no_change=30,
            categorical_features="from_dtype",
            random_state=42,
        )
        return Pipeline([("prep", prep), ("model", model)])

    raise ValueError(f"Unknown pipeline kind: {kind!r}")
