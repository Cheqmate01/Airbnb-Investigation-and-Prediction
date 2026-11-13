"""Create the four analysis notebooks and optionally execute them.

Usage:
    python scripts/make_notebooks.py            # build only
    python scripts/make_notebooks.py --execute  # build and run
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf

ROOT = Path(__file__).resolve().parents[1]
NOTEBOOKS = ROOT / "notebooks"


def md(text: str):
    return nbf.v4.new_markdown_cell(text.strip())


def code(text: str):
    return nbf.v4.new_code_cell(text.strip())


def notebook_01() -> nbf.NotebookNode:
    cells = [
        md("""
# 01 — Data Quality Audit

**Project:** Airbnb Market Intelligence (Data Science + Machine Learning)
**Purpose:** inspect the four supplied files, validate keys and relationships,
quantify missingness, and document the price/currency issues before any analysis.

The heavy review-file checks use Polars' lazy scanner so the 5.37m-row file is
never loaded into a spreadsheet-style structure.
"""),
        code("""
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)

from src.config import LISTINGS_RAW, REVIEWS_RAW, FX_SOURCE
from src.loading import load_listings
from src.build_dataset import add_price_flags

listings = add_price_flags(load_listings())
print("Listings shape:", listings.shape)
print("Files:")
for path in [LISTINGS_RAW, REVIEWS_RAW]:
    print(f"  {path.name}: {path.stat().st_size / 1e6:,.1f} MB")
print("\\nFX assumption:", FX_SOURCE)
listings.head(3)
"""),
        md("## 1. Primary key: Listings"),
        code("""
pd.DataFrame({
    "rows": [len(listings)],
    "distinct_listing_id": [listings["listing_id"].nunique()],
    "duplicate_listing_id": [int(listings["listing_id"].duplicated().sum())],
    "null_listing_id": [int(listings["listing_id"].isna().sum())],
})
"""),
        md("## 2. Missing values"),
        code("""
fields = ["host_response_time", "host_response_rate", "host_acceptance_rate",
          "district", "bedrooms", "review_scores_rating", "review_scores_value", "host_since"]
missing = pd.DataFrame({
    "missing_count": listings[fields].isna().sum(),
    "missing_pct": (100 * listings[fields].isna().mean()).round(2),
})
missing.sort_values("missing_pct", ascending=False)
"""),
        code("""
(listings.groupby("city", observed=True)[["district", "bedrooms", "review_scores_rating"]]
         .apply(lambda g: 100 * g.isna().mean()).round(1))
"""),
        md("""
`district` is populated only for New York. Review scores exist only for listings
that have been reviewed. Neither gap is missing at random, so every later
statistic states which sub-population it describes.
"""),
        md("## 3. Price validity and currency structure"),
        code("""
pd.DataFrame({
    "check": ["price missing", "price == 0", "price < 0",
              "above per-city 99th pct (USD)", "below per-city 1st pct (USD)"],
    "rows": [int(listings["price"].isna().sum()),
             int((listings["price"] == 0).sum()),
             int((listings["price"] < 0).sum()),
             int(listings["price_outlier_high"].sum()),
             int(listings["price_outlier_low"].sum())],
})
"""),
        code("""
currency_ref = (listings.groupby("city", observed=True)
                        .agg(currency=("currency", "first"),
                             listings=("listing_id", "size"),
                             usd_per_unit=("fx_usd_per_unit", "first"))
                        .reset_index())
currency_ref
"""),
        code("""
(listings[listings["price_valid"]]
    .groupby("city", observed=True)
    .agg(currency=("currency", "first"),
         n=("listing_id", "size"),
         median_local=("price", "median"),
         p99_local=("price", lambda s: s.quantile(0.99)),
         max_local=("price", "max"),
         median_usd=("price_usd", "median"))
    .round(2))
"""),
        md("""
Every price statistic above is computed **inside** a city. The local median and
the USD median are different quantities and are never compared as if they were
the same currency. The huge local maxima (Rio de Janeiro, Istanbul, Bangkok) are
implausible nightly rates and are handled as extreme observations.
"""),
        md("## 4. Reviews: keys, duplicates and join coverage"),
        code("""
from src.reviews import validate_reviews

findings = validate_reviews(REVIEWS_RAW)
pd.Series(findings).to_frame("value")
"""),
        code("""
from src.reviews import duplicate_review_evidence

dup = duplicate_review_evidence(REVIEWS_RAW)
print(f"Rows involved in repeated review_id values: {len(dup):,}")
print(f"Distinct repeated review_id values: {dup['review_id'].nunique()}")
dup.head(10)
"""),
        md("""
The duplicated `review_id` values are real in the supplied file. Of the repeated
IDs, some carry identical payloads and some conflict. Because
`(listing_id, review_id)` is unique, the reviews are aggregated by `listing_id`
and `review_id` is never used as a primary key. No review rows are deleted, so
the 5.37m review records remain fully accounted for.
"""),
        md("## 5. Categorical structure"),
        code("""
display(listings["room_type"].value_counts().to_frame("listings"))
display(listings["property_family"].value_counts().to_frame("listings"))
print("Raw property_type values:", listings["property_type"].nunique())
"""),
        md("""
## 6. Decisions and limitations

1. `listing_id` is the Listings primary key; `(listing_id, review_id)` is the
   Reviews unique key.
2. Reviews are left-joined to listings: no review is a real zero, not a missing row.
3. Prices are local currency; USD conversion uses one documented ECB snapshot.
4. Non-positive prices are invalid. Values above each city's 99th percentile are
   flagged as extreme and excluded from model training only.
5. No review text exists, so sentiment analysis is out of scope.
6. Advertised price is not revenue, occupancy or profit.

The full generated report is `reports/data_quality_report.md`.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells)


def notebook_02() -> nbf.NotebookNode:
    cells = [
        md("""
# 02 — Exploratory Data Analysis

**Research problem:** how do advertised prices, property characteristics and
review activity differ across the ten Airbnb markets?

**Currency rule:** raw price distributions are shown per city in local currency.
Cross-city comparisons use a dimensionless price index (price ÷ city median).
"""),
        code("""
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
pd.set_option("display.width", 180)
pd.set_option("display.max_columns", 60)

from src.eda import load, run
from IPython.display import Image, display

df = load()
print(f"{len(df):,} listings with a usable price across {df['city'].nunique()} cities")
display(df[["price", "price_usd", "accommodates", "review_count"]].describe().round(1))
"""),
        md("## Market overview"),
        code("""
paths = run(df)
display(Image(paths["listings_by_city"]))
display(pd.read_csv("../outputs/tables/eda_city_summary.csv"))
"""),
        md("""
Paris is the largest market in the extract (64,690 listings), Hong Kong the
smallest (7,087). Listing counts describe the supplied extract, not the live
market.
"""),
        md("## Price distributions — local currency, one panel per city"),
        code("""
display(Image(paths["price_distribution"]))
"""),
        md("""
All ten distributions are right-skewed with a long upper tail. Because the
panels use different currencies, only the *shape* is comparable; levels are not.
This is why the analysis uses within-city tests and relative indices.
"""),
        md("## Room type and price"),
        code("""
display(Image(paths["price_index_heatmap"]))
display(Image(paths["roomtype_mix"]))
display(pd.read_csv("../outputs/tables/eda_roomtype_by_city.csv"))
"""),
        md("""
`Entire place` sits above each city's median everywhere; private and shared
rooms sit below it. The heatmap is unitless, so the pattern is comparable even
though the underlying prices are not.
"""),
        md("## Property types"),
        code("""
display(Image(paths["property_family"]))
"""),
        md("## Capacity and relative price"),
        code("""
display(Image(paths["capacity_price"]))
"""),
        md("""
Relative price rises monotonically with capacity up to roughly 6–8 guests and
then flattens or falls as large listings are discounted per guest. This is a
strong candidate for the price model and a hypothesis test.
"""),
        md("## Review activity"),
        code("""
display(Image(paths["review_activity"]))
display(Image(paths["reviews_per_listing"]))
display(pd.read_csv("../outputs/tables/eda_review_summary.csv"))
"""),
        md("""
Review volume grew strongly until the 2020 disruption and recovered unevenly.
Review counts are cumulative and confounded with listing age, so they cannot be
read as current demand. Listings without reviews are included as zeros in the
aggregate review analysis and excluded from rating analysis.
"""),
        md("## Ratings and host indicators"),
        code("""
display(Image(paths["ratings_roomtype"]))
display(Image(paths["host_response"]))
"""),
        md("""
Ratings are compressed near the top of the scale, which limits how much
separation any group comparison can show. Host response rate is missing for
~46% of listings; the boxplots describe the listings where it is reported.

## What this EDA does not show

- It does not measure occupancy, bookings or revenue.
- It does not measure guest sentiment: there is no review text and no per-review rating.
- It does not establish causation between any attribute and price.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells)


def notebook_03() -> nbf.NotebookNode:
    cells = [
        md("""
# 03 — Hypothesis Testing

Three core hypotheses (with two supporting tests) are pre-specified, tested with
non-parametric methods, corrected for multiple comparisons, and reported with
effect sizes so that statistical and practical significance are separated.

| # | Hypothesis | Test |
|---|---|---|
| H1 | Advertised price differs across room types within a city | Kruskal–Wallis + Dunn |
| H2 | Capacity / bedrooms are associated with price within a city | Spearman ρ |
| H3 | Listing ratings differ across room types / property families | Kruskal–Wallis |
| H4 | Review activity differs across room types | Kruskal–Wallis |
| H5 | Host response rate differs across cities | Kruskal–Wallis |
"""),
        code("""
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

from src.config import ANALYSIS_DATASET, CITY_ORDER
from src.statistics import hypothesis_suite

df = pd.read_parquet(ANALYSIS_DATASET)
df["city"] = pd.Categorical(df["city"].astype(str), categories=CITY_ORDER, ordered=True)
print(f"Rows: {len(df):,} | price-analysable: {int(df['price_analysable'].sum()):,}")
"""),
        md("""
## H1 — Price by room type (within city, local currency)

**H₀:** the price distribution is identical across room types.
**H₁:** at least one room type differs.
Kruskal–Wallis is used because prices are heavily right-skewed; Dunn's test
provides pairwise comparisons with Bonferroni control; ε² is the effect size.
"""),
        code("""
res = hypothesis_suite(df, CITY_ORDER)
h1 = res["h1_kruskal_price_roomtype"].sort_values("epsilon_squared", ascending=False)
h1[["city", "currency", "n", "H", "p_value", "p_bh", "epsilon_squared", "effect"]]
"""),
        code("""
dunn = res["h1_dunn_price_roomtype"]
print(f"Pairwise comparisons: {len(dunn):,} | significant after Bonferroni: {int(dunn['significant_5pct'].sum()):,}")
dunn[dunn["significant_5pct"]].sort_values("p_bonferroni").head(15)
"""),
        md("""
**Interpretation.** If H1 is rejected in most cities, room type is a reliable
price differentiator, and the post-hoc pairs identify which categories separate.
With tens of thousands of listings, p-values will be tiny; ε² is the guard
against overstating a trivial difference.
"""),
        md("## H2 — Capacity and bedrooms vs price (within city)"),
        code("""
res["h2_spearman_capacity_price"], res["h2_spearman_bedrooms_price"]
"""),
        code("""
pd.concat([
    res["h2_spearman_capacity_price"].assign(variable="accommodates"),
    res["h2_spearman_bedrooms_price"].assign(variable="bedrooms"),
]).pivot(index="city", columns="variable", values="rho").round(3)
"""),
        md("""
**Interpretation.** ρ measures monotonic association, not a causal effect. A
moderate ρ means capacity explains part of the price signal; the rest comes from
location, property type, amenities and host positioning.
"""),
        md("## H3 — Ratings by room type and property family"),
        code("""
res["h3_kruskal_rating_roomtype"].sort_values("city")
"""),
        code("""
res["h3b_kruskal_rating_property"].sort_values("city")
"""),
        md("""
**Interpretation.** Ratings are compressed near the top of the scale, so even a
statistically significant difference can be practically small. Effect sizes here
are expected to be small; that is itself a finding, not a failed test.
"""),
        md("## H4 — Review activity by room type"),
        code("""
res["h4_kruskal_reviews_roomtype"].sort_values("city")
"""),
        md("""
**Interpretation.** Review count is a proxy for booking-related activity, not
satisfaction. It is confounded by listing age and market size.
"""),
        md("## H5 — Host response rate across cities"),
        code("""
res["h5_kruskal_host_response_city"]
"""),
        md("""
**Interpretation.** A significant result describes differences in reported host
behaviour across markets. It is not a service-quality measure.

## Method notes

- Benjamini–Hochberg controls the false-discovery rate across the repeated
  per-city tests.
- Kruskal–Wallis tests stochastic dominance and assumes broadly similar
  distribution shapes; it is not a test of equal medians.
- Listings are treated as independent. Multi-listing hosts violate this mildly,
  so p-values are optimistic.
- Excluding prices above the 99th percentile prevents a handful of data errors
  from dominating rank-based tests.
- Association is not causation.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells)


def notebook_04() -> nbf.NotebookNode:
    cells = [
        md("""
# 04 — Price Prediction Model

Reproduces the evaluation recorded in `reports/model_report.md`. Model training
is executed by `python -m src.modelling` (see `make model`); this notebook reads
the saved artefacts, inspects error behaviour and demonstrates inference.

**Target:** advertised nightly price in USD, trained on `log1p` and reported
after the inverse transform. The model is a price *estimator*, not a revenue,
occupancy or profit forecast.
"""),
        code("""
import sys
from pathlib import Path

PROJECT_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(PROJECT_ROOT))

import json
import pandas as pd
from IPython.display import Image, display

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 60)

comparison = pd.read_csv("../outputs/tables/model_comparison.csv")
metadata = json.loads(Path("../outputs/models/model_metadata.json").read_text())
print("Best model:", metadata["model_family"])
print("FX assumption:", metadata["currency_assumption"])
comparison.sort_values("MAE_usd")
"""),
        md("""
## Baseline comparison

The per-city median is the honest baseline: it already accounts for market level.
A model earns its place only if it improves on that.
"""),
        code("""
comparison.set_index("model")[["MAE_usd", "RMSE_usd", "R2", "MedAE_usd", "median_abs_pct_error"]].round(3)
"""),
        md("## Cross-validation"),
        code("""
pd.read_csv("../outputs/tables/model_cross_validation.csv")
"""),
        md("## Where the model works and where it fails"),
        code("""
display(pd.read_csv("../outputs/tables/model_error_by_city.csv"))
display(pd.read_csv("../outputs/tables/model_error_by_price_band.csv"))
"""),
        code("""
display(Image("../outputs/figures/ml_predicted_vs_actual.png"))
display(Image("../outputs/figures/ml_residuals.png"))
"""),
        md("""
Residuals are wider at the top of the price range: expensive and unusual
properties are genuinely harder to price from the available attributes. The
largest individual errors are listed in
`outputs/tables/model_largest_errors.csv`.
"""),
        md("## Feature importance"),
        code("""
display(Image("../outputs/figures/ml_feature_importance.png"))
importance = pd.read_csv("../outputs/tables/feature_importance_permutation.csv")
importance.head(15)
"""),
        md("""
Permutation importance measures predictive contribution, not causal effect.
Location and capacity dominate, consistent with the statistical tests.
"""),
        md("## Inference demonstration"),
        code("""
from src.inference import predict_price

listing = {
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
predict_price(listing)
"""),
        md("""
## Limitations

- Predicted advertised price is not realised revenue.
- USD conversion depends on one ECB snapshot; local-currency outputs inherit that assumption.
- Extreme advertised prices were excluded from training; the model is conservative at the top of the market.
- CORE features exclude review history so the model works for a new listing.
- No bathrooms field exists; bedrooms is missing for ~10% of rows and is imputed.
- Feature importance is predictive, not causal.
"""),
    ]
    return nbf.v4.new_notebook(cells=cells)


def write_all() -> list[Path]:
    NOTEBOOKS.mkdir(parents=True, exist_ok=True)
    made = []
    for name, factory in [
        ("01_data_quality_audit.ipynb", notebook_01),
        ("02_exploratory_analysis.ipynb", notebook_02),
        ("03_hypothesis_testing.ipynb", notebook_03),
        ("04_price_prediction.ipynb", notebook_04),
    ]:
        path = NOTEBOOKS / name
        nbf.write(factory(), path)
        made.append(path)
        print("wrote", path)
    return made


def execute(paths: list[Path]) -> None:
    from nbconvert.preprocessors import ExecutePreprocessor

    ep = ExecutePreprocessor(timeout=3600, kernel_name="python3")
    for path in paths:
        print("executing", path.name, flush=True)
        nb = nbf.read(path, as_version=4)
        ep.preprocess(nb, {"metadata": {"path": str(NOTEBOOKS)}})
        nbf.write(nb, path)
        print("  done", flush=True)


if __name__ == "__main__":
    paths = write_all()
    if "--execute" in sys.argv:
        execute(paths)
