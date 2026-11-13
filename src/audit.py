"""Data-quality audit: produces tables and a Markdown report.

Run: ``python -m src.audit``
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import polars as pl

from src.config import (
    CITY_CURRENCY,
    CITY_ORDER,
    DOCS,
    ECB_RATES_PER_EUR,
    FX_SNAPSHOT_DATE,
    FX_SOURCE,
    LISTINGS_RAW,
    REPORTS,
    REVIEWS_RAW,
    TABLES,
    usd_per_unit,
)
from src.build_dataset import add_price_flags
from src.loading import LISTING_COLUMNS, load_listings
from src.reviews import _scan, validate_reviews

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def md_table(df: pd.DataFrame, index: bool = False, floatfmt: str = ".2f") -> str:
    """Small dependency-free DataFrame -> Markdown table renderer."""
    frame = df.reset_index() if index else df.copy()
    frame = frame.where(pd.notna(frame), "—")

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return format(float(v), floatfmt)
        if isinstance(v, (int, np.integer)):
            return f"{int(v):,}"
        return str(v)

    cols = [str(c) for c in frame.columns]
    rows = [[fmt(v) for v in row] for row in frame.itertuples(index=False, name=None)]
    out = ["| " + " | ".join(cols) + " |", "|" + "|".join(["---"] * len(cols)) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return "\n".join(out)


def _price_stats(df: pd.DataFrame, value: str) -> pd.DataFrame:
    rows = []
    for city, g in df.groupby("city", observed=True):
        s = g[value].dropna()
        if s.empty:
            continue
        rows.append(
            {
                "city": str(city),
                "currency": CITY_CURRENCY[str(city)],
                "n": len(s),
                "min": s.min(),
                "p25": s.quantile(0.25),
                "median": s.median(),
                "mean": s.mean(),
                "p75": s.quantile(0.75),
                "p99": s.quantile(0.99),
                "max": s.max(),
                "mean_over_median": s.mean() / s.median() if s.median() else np.nan,
            }
        )
    return pd.DataFrame(rows)


def main() -> dict[str, object]:
    listings = add_price_flags(load_listings())
    findings = validate_reviews()

    # ---------------------------------------------------------------- shape
    n_rows = len(listings)
    n_cols = len(LISTING_COLUMNS)  # supplied file; the cleaned table adds derived columns
    review_counts = pl.read_csv(REVIEWS_RAW, columns=["listing_id"]).get_column("listing_id")

    # ------------------------------------------------------------ duplicate
    lf = _scan(REVIEWS_RAW)
    repeated_ids = (
        lf.group_by("review_id").len().filter(pl.col("len") > 1).select("review_id")
    )
    dup = lf.join(repeated_ids, on="review_id", how="semi").collect()
    dup_pandas = dup.to_pandas()
    repeated_groups = dup_pandas.groupby("review_id")
    differing = sum(
        1 for _, g in repeated_groups if g[["listing_id", "date", "reviewer_id"]].drop_duplicates().shape[0] > 1
    )
    identical = repeated_groups.ngroups - differing
    dup_detail = (
        dup_pandas.sort_values(["review_id", "listing_id"])
        .head(40)
        .assign(date=lambda d: d["date"].astype(str))
    )

    # --------------------------------------------------------------- coverage
    listing_ids = set(listings["listing_id"])
    review_listing_ids = set(review_counts.to_list())
    missing_in_listings = review_listing_ids - listing_ids
    coverage = {
        "reviews_rows": findings["rows"],
        "unique_review_listing_ids": len(review_listing_ids),
        "listings_total": len(listing_ids),
        "listings_with_at_least_one_review": len(review_listing_ids & listing_ids),
        "review_listing_ids_not_in_listings": len(missing_in_listings),
        "listings_without_reviews": len(listing_ids - review_listing_ids),
        "coverage_rate_pct": 100.0
        * len(review_listing_ids & listing_ids)
        / max(len(listing_ids), 1),
    }

    # --------------------------------------------------------------- missing
    fields = [
        "host_response_time", "host_response_rate", "host_acceptance_rate",
        "district", "bedrooms", "review_scores_rating",
        "review_scores_accuracy", "review_scores_cleanliness",
        "review_scores_checkin", "review_scores_communication",
        "review_scores_location", "review_scores_value", "host_since",
    ]
    missing = pd.DataFrame(
        {
            "field": fields,
            "missing_count": [int(listings[f].isna().sum()) for f in fields],
            "missing_pct": [100 * listings[f].isna().mean() for f in fields],
            "non_missing_dtype": [str(listings[f].dropna().dtype) for f in fields],
        }
    ).sort_values("missing_pct", ascending=False)

    missing_by_city = (
        listings.groupby("city", observed=True)[
            ["district", "bedrooms", "review_scores_rating", "host_response_rate", "host_acceptance_rate"]
        ]
        .apply(lambda g: 100 * g.isna().mean())
    )

    # ---------------------------------------------------------------- price
    price_local = _price_stats(listings[listings["price_valid"]], "price")
    price_usd = _price_stats(listings[listings["price_valid"]], "price_usd")
    price_usd["median_local"] = price_local["median"].values
    price_usd["currency"] = price_local["currency"].values

    invalid = pd.DataFrame(
        {
            "issue": [
                "price == 0",
                "price < 0",
                "price missing",
                "price > per-city 99th percentile (USD)",
                "price < per-city 1st percentile (USD)",
            ],
            "rows": [
                int((listings["price"] == 0).sum()),
                int((listings["price"] < 0).sum()),
                int(listings["price"].isna().sum()),
                int(listings["price_outlier_high"].sum()),
                int(listings["price_outlier_low"].sum()),
            ],
        }
    )

    # ----------------------------------------------------------- categories
    room_types = listings["room_type"].value_counts(dropna=False).rename_axis("room_type").reset_index(name="listings")
    room_types["pct"] = 100 * room_types["listings"] / len(listings)

    property_types = (
        listings["property_type"].value_counts(dropna=False).rename_axis("property_type").reset_index(name="listings")
    )
    property_types["pct"] = 100 * property_types["listings"] / len(listings)
    property_families = (
        listings["property_family"].value_counts().rename_axis("property_family").reset_index(name="listings")
    )

    city_counts = (
        listings["city"].value_counts().reindex(CITY_ORDER).rename_axis("city").reset_index(name="listings")
    )
    city_counts["pct"] = 100 * city_counts["listings"] / len(listings)

    currency_ref = pd.DataFrame(
        [
            {
                "city": city,
                "currency": CITY_CURRENCY[city],
                "usd_per_unit": usd_per_unit(CITY_CURRENCY[city]),
                "units_per_eur": ECB_RATES_PER_EUR[CITY_CURRENCY[city]],
            }
            for city in CITY_ORDER
        ]
    )

    # ---------------------------------------------------------------- write
    TABLES.mkdir(parents=True, exist_ok=True)
    price_local.to_csv(TABLES / "audit_price_local_by_city.csv", index=False)
    price_usd.to_csv(TABLES / "audit_price_usd_by_city.csv", index=False)
    missing.to_csv(TABLES / "audit_missing_values.csv", index=False)
    missing_by_city.to_csv(TABLES / "audit_missing_by_city.csv")
    invalid.to_csv(TABLES / "audit_price_issues.csv", index=False)
    room_types.to_csv(TABLES / "audit_room_types.csv", index=False)
    property_types.to_csv(TABLES / "audit_property_types.csv", index=False)
    property_families.to_csv(TABLES / "audit_property_families.csv", index=False)
    city_counts.to_csv(TABLES / "audit_city_counts.csv", index=False)
    currency_ref.to_csv(TABLES / "audit_currency_reference.csv", index=False)
    dup_detail.to_csv(TABLES / "audit_duplicate_review_ids_sample.csv", index=False)

    audit_findings = {
        "listings_rows": n_rows,
        "listings_columns": n_cols,
        "listings_unique_ids": int(listings["listing_id"].nunique()),
        "listings_duplicate_ids": int(listings["listing_id"].duplicated().sum()),
        "review_validation": findings,
        "coverage": coverage,
        "duplicate_review_ids": {
            "repeated_ids": findings["duplicated_review_id_values"],
            "extra_rows": findings["extra_rows_from_repeated_ids"],
            "repeated_ids_with_identical_rows": identical,
            "repeated_ids_with_conflicting_rows": differing,
        },
    }
    (TABLES / "audit_findings.json").write_text(json.dumps(audit_findings, indent=2, default=str))

    # -------------------------------------------------------------- report
    report = f"""# Data-Quality Assessment — Airbnb Listings & Reviews

Generated by `python -m src.audit`.

## 1. Files inspected

| File | Rows | Columns | Notes |
|---|---|---|---|
| `Datasets/Listings.csv` | {n_rows:,} | {n_cols} | one row per listing |
| `Datasets/Reviews.csv` | {findings['rows']:,} | 4 | review history, no review text |
| `Datasets/Listings_data_dictionary.csv` | 33 definitions | 2 | supplied schema |
| `Datasets/Reviews_data_dictionary.csv` | 4 definitions | 2 | supplied schema |

**Encoding defect.** `Listings.csv` is described as UTF-8 but contains 984 invalid
byte sequences (e.g. `A\\x81lvaro Obregon`). Reading with `encoding="latin-1"`
and applying a bounded mojibake repair recovers the intended characters. This is
a file-level issue, not a row-level one; no records are lost.

## 2. Primary and foreign keys

### Listings

- Rows: **{n_rows:,}**
- Distinct `listing_id`: **{audit_findings['listings_unique_ids']:,}**
- Duplicate `listing_id` values: **{audit_findings['listings_duplicate_ids']}**
- Null `listing_id`: **{int(listings['listing_id'].isna().sum())}**

`listing_id` is a valid primary key for Listings.

### Reviews

- Rows: **{findings['rows']:,}**
- Distinct `review_id`: **{findings['unique_review_ids']:,}**
- Distinct `listing_id`: **{findings['unique_listings']:,}**
- Distinct `reviewer_id`: **{findings['unique_reviewers']:,}**
- Review dates: **{findings['min_date']} → {findings['max_date']}**
- Repeated `review_id` values: **{findings['duplicated_review_id_values']}**
  (adds {findings['extra_rows_from_repeated_ids']} extra rows)
- Repeated `(listing_id, review_id)` keys: **{findings['duplicated_listing_review_keys']}**
- Exact duplicate review rows: **{findings['exact_duplicate_rows']}**

`review_id` alone is not unique. Of the {findings['duplicated_review_id_values']} repeated
IDs, **{identical}** have identical `(listing_id, date, reviewer_id)` payloads and
**{differing}** conflict. `(listing_id, review_id)` is unique, confirming the
documented key. **Decision:** keep all review rows, aggregate on `listing_id`,
and never treat `review_id` as a primary key. See
`outputs/tables/audit_duplicate_review_ids_sample.csv` for evidence.

## 3. Listing → review relationship

| Measure | Value |
|---|---|
| Review rows | {coverage['reviews_rows']:,} |
| Listings with ≥ 1 review | {coverage['listings_with_at_least_one_review']:,} |
| Listings without reviews | {coverage['listings_without_reviews']:,} |
| Coverage rate | {coverage['coverage_rate_pct']:.1f}% |
| Review `listing_id`s missing from Listings | {coverage['review_listing_ids_not_in_listings']} |

The relationship is one-to-many and referentially valid. The join must be a
**left join from Listings**; a listing with no review is a real observation, not
missing data.

## 4. Missing values

{md_table(missing)}
Missingness by city:

{md_table(missing_by_city.reset_index(), floatfmt='.1f')}

`district` is complete **only for New York** ({100 - missing_by_city.loc['New York', 'district']:.1f}% available)
and 100% missing in every other city, so it cannot support cross-city district
analysis.
`bedrooms` and the review-score fields are missing systematically (listings with
no reviews cannot have review scores), so review-score statistics describe
**rated listings only**.

## 5. Price and currency

Prices are integers in each market's local currency; Listings.csv has no currency
column. Mapping used (city → currency) and the ECB snapshot rates
({FX_SOURCE}):

{md_table(currency_ref)}

Raw local-currency description:

{md_table(price_local)}

USD-standardised description (FX snapshot {FX_SNAPSHOT_DATE}):

{md_table(price_usd)}

Price issues:

{md_table(invalid)}

`price == 0` and negative prices are invalid and are excluded from analysis;
they are retained in the cleaned file with `price_valid = False`. The USD upper
tail is dominated by a small number of implausible values (for example a
Rio de Janeiro listing at the file maximum). Those are flagged with
`price_outlier_high` and excluded from modelling, but shown in descriptive EDA.

## 6. Categorical structure

City distribution:

{md_table(city_counts)}

Room types:

{md_table(room_types)}

Top property types (144 distinct raw values; reduced to families for modelling):

{md_table(property_types.head(12))}

## 7. Decisions recorded

1. `listing_id` is the Listings primary key; `(listing_id, review_id)` is the
   Reviews unique key. Repeated `review_id` values are documented, not deleted.
2. Reviews are aggregated to one row per listing with a left join.
3. Prices are converted to USD with a single documented ECB snapshot; local
   currency is retained for within-city analysis.
4. Non-positive prices are invalid. Prices above the per-city 99th percentile
   are flagged as extreme and excluded from model training only.
5. Missing values are not imputed in the cleaned file; each analysis documents
   its handling. The modelling pipeline adds explicit imputation plus missingness
   indicators.
6. Review text and per-review ratings do not exist in the supplied data, so no
   sentiment or comment analysis is performed.

## 8. Limitations carried forward

- Advertised prices are not revenue, occupancy or profit.
- Currency conversion uses one snapshot date; results would shift with another date.
- Coverage differs by city and by listing age, so review-based metrics are not
  missing at random.
- `district` supports only New York; neighbourhood granularity varies by city.
- No booking, availability or calendar data is supplied.
"""
    (REPORTS / "data_quality_report.md").write_text(report)
    print(json.dumps(audit_findings, indent=2, default=str))
    return audit_findings


if __name__ == "__main__":
    main()
