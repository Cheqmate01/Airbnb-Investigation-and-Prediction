# Airbnb Market Intelligence — Pricing, Location and Listing Characteristics

**AnalystLab Africa Experience Lab — Week 1, Batch G**
**Tracks covered: B (Data Science) and C (Machine Learning)**

This repository investigates how Airbnb listings are distributed, priced and
reviewed across ten cities, and develops a reproducible model that estimates
advertised nightly prices from listing characteristics. It contains the complete
Data Science track and Machine Learning track deliverables: cleaned data, a
data-quality audit, exploratory analysis, hypothesis tests, a trained price model,
an inference utility, tests, executed notebooks and written reports.

---

## 1. Headline findings

- **Paris** is the largest market in the extract (64,690 listings, 23.1%), **Hong Kong**
  the smallest (7,087, 2.5%).
- Advertised prices are **right-skewed in every city**; the mean sits 27–87% above
  the median depending on the market, so medians are the fairer comparison.
- **Room type is the strongest categorical price differentiator.** Kruskal–Wallis
  rejects equality of price distributions in all 10 cities (ε² = 0.19 median);
  shared rooms price below private rooms, which price below entire places.
  Hotel rooms are the exception, pricing above entire places in five markets
  (Bangkok, Cape Town, Mexico City, New York and Paris).
- **Capacity is the strongest continuous driver**: Spearman ρ with price is 0.52
  (median across cities) for `accommodates` and 0.47 for `bedrooms`, significant
  in 10/10 cities after FDR correction.
- **Ratings do not separate listings meaningfully.** Room-type and property-family
  differences are statistically significant but practically negligible
  (median ε² ≈ 0.005–0.007): ratings cluster near the top of the scale.
- **Review activity** differs by room type, but the effect is modest (median ε² = 0.047)
  and is confounded with listing age and market size.
- A **gradient-boosted model** improves held-out MAE by **37.5%** over the
  per-city median baseline, with test MAE **$31.75**, RMSE **$67.99** and
  R² **0.581**. City, bedrooms, longitude, capacity and room type dominate
  importance.

> All prices are **advertised prices**, not revenue, occupancy or profit.
> All findings are **associations**, not causal effects.

---

## 2. Repository layout

```
.
├── Datasets/                     supplied files (never modified)
├── data/
│   ├── interim/                  intermediate parquet
│   └── processed/                cleaned listings, review aggregates, analysis dataset
├── src/
│   ├── config.py                 paths, city→currency map, ECB FX snapshot, seeds
│   ├── loading.py                listings loader, cleaning, derived features
│   ├── reviews.py                Polars validation + review aggregation
│   ├── build_dataset.py          builds the analysis dataset
│   ├── audit.py                  data-quality tables + report
│   ├── eda.py                    exploratory figures + summary tables
│   ├── statistics.py             Kruskal–Wallis, Dunn, Spearman, BH correction
│   ├── analysis.py               hypothesis suite + statistical report
│   ├── features.py               feature sets and sklearn pipelines
│   ├── modelling.py              training, evaluation, artifacts
│   ├── inference.py              input validation + prediction API
│   └── reporting.py              model report generation
├── notebooks/                    four executed analysis notebooks
├── scripts/                      notebook generator and pipeline runner
├── outputs/
│   ├── figures/                  10 EDA + 4 model figures
│   ├── tables/                   audit, EDA, hypothesis and model result tables
│   └── models/                   joblib model, metadata, test predictions
├── reports/                      data-quality, statistical, model and executive reports
├── tests/                        pytest suite
├── docs/                         ECB rates snapshot and supporting evidence
├── requirements.txt              pinned Python 3.9 environment
└── Makefile                      one-command reproducibility
```

---

## 3. Setup

Requires **Python 3.9**.

```bash
python3.9 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name airbnb-week1 --display-name "Airbnb Week 1"
```

Or simply:

```bash
make setup
```

**Note on Polars:** this machine's CPU lacks AVX/AVX2, so the requirements use
`polars-lts-cpu`. On a standard x86-64 machine, `polars` can be substituted.

## 4. Reproduce everything

```bash
make all        # build → audit → EDA → statistics → model → notebooks
make test       # pytest suite
```

Individual stages:

| Command | Produces |
|---|---|
| `make build` | cleaned listings, review aggregates, analysis dataset |
| `make audit` | `reports/data_quality_report.md` + audit tables |
| `make eda` | 10 figures + city/room/review summary tables |
| `make analyse` | hypothesis tables + `reports/statistical_analysis.md` |
| `make model` | model comparison, error analysis, saved model + model report |
| `make notebooks` | rebuilds and executes the four notebooks |

Runtime on a 2-core / 3 GB machine: build ≈ 4 min (review file), audit ≈ 3 min,
EDA ≈ 30 s, statistics ≈ 10 s, modelling ≈ 12–15 min, notebooks ≈ 6 min.

---

## 5. Data

| File | Rows | Columns | Role |
|---|---|---|---|
| `Listings.csv` | 279,712 | 33 | listing attributes, host profile, price, review scores |
| `Reviews.csv` | 5,373,143 | 4 | `listing_id`, `review_id`, `date`, `reviewer_id` |
| `Listings_data_dictionary.csv` | 33 | 2 | field definitions |
| `Reviews_data_dictionary.csv` | 4 | 2 | field definitions |

Snapshot: the latest review date is **2021-03-01**; reviews span 2008-11-16 to that date.

### Data-quality issues found and handled

| Issue | Evidence | Treatment |
|---|---|---|
| Mixed encoding | 984 invalid UTF-8 byte sequences in `Listings.csv` | read as Latin-1 + bounded mojibake repair; no rows lost |
| Repeated `review_id` | 160 IDs repeat, adding 160 rows; all 160 payloads conflict | keep all rows; `(listing_id, review_id)` is the unique key; aggregate by `listing_id` |
| Non-positive prices | 113 rows at price 0 | flagged `price_nonpositive`, excluded from analysis |
| Extreme prices | 2,753 rows above their city's 99th percentile | flagged `price_outlier_high`, excluded from model training only |
| Missing `district` | 86.8% missing — complete only for New York | not used for cross-city district analysis |
| Missing host rates | response 46.0%, acceptance 40.4% | missingness indicators in the model; excluded from descriptive means where stated |
| Missing ratings | 32.7% (no reviews) | rating statistics describe rated listings only |
| Missing bedrooms | 10.5% | median imputation + missing indicator |
| No review text or per-review ratings | schema | no sentiment analysis; review analysis is frequency/timing only |
| No bookings, calendars, revenue | schema | no occupancy/revenue claims anywhere |

All review `listing_id`s match a listing (0 orphans). 193,556 listings (69.2%) have
at least one review; 86,156 have none. The join is a left join, and zero reviews is
a real observation.

---

## 6. Currency methodology

`Listings.csv` has **no currency column**. Prices are in each market's local
currency, so raw prices are never pooled or compared across cities.

| City | Currency | USD per unit | Units per EUR |
|---|---|---|---|
| Paris, Rome | EUR | 1.112700 | 1.0000 |
| New York | USD | 1.000000 | 1.1127 |
| Sydney | AUD | 0.685982 | 1.6222 |
| Rio de Janeiro | BRL | 0.180000 | 6.1827 |
| Istanbul | TRY | 0.019909 | 55.8897 |
| Mexico City | MXN | 0.053599 | 20.7623 |
| Bangkok | THB | 0.029682 | 37.4870 |
| Cape Town | ZAR | 0.060372 | 18.4307 |
| Hong Kong | HKD | 0.127459 | 8.7298 |

**Source:** European Central Bank euro foreign-exchange reference rates,
snapshot **2026-11-11**, archived in `docs/ecb_reference_rates_2026-11-11.xml`.

**Assumption:** one conversion date is applied to all listings because the file
carries no price date. FX movements between the listing snapshot (2021) and the
conversion date are not modelled, and this is stated as a limitation. Every
within-city test uses local currency and is therefore unaffected by this choice.

---

## 7. Data Science track (B)

### Research design

**Primary research problem:** how do advertised prices, room types, capacity and
review activity differ across the ten Airbnb markets, and which listing
characteristics are associated with higher advertised prices?

**Supporting questions**

1. How does advertised price vary across room types within a city?
2. Are capacity and bedroom count associated with higher prices?
3. Do listing-level ratings and review activity differ across categories?

**Intended users:** market analysts, hosts considering a listing, and platform
teams evaluating market positioning.

**Decisions supported:** market screening, segment targeting, property-size
positioning, and interpreting review metrics correctly.

### Hypotheses and methods

| ID | Null hypothesis | Test | Effect size |
|---|---|---|---|
| H1 | Price distributions are equal across room types within a city | Kruskal–Wallis + Dunn (Bonferroni) | ε² |
| H2 | No monotonic association between capacity/bedrooms and price | Spearman ρ with Fisher-z CI | ρ |
| H3 | Ratings are equal across room types / property families | Kruskal–Wallis | ε² |
| H4 | Review activity is equal across room types | Kruskal–Wallis | ε² |
| H5 | Host response rate is equal across cities | Kruskal–Wallis | ε² |

Non-parametric methods are used because prices are heavily right-skewed and
heteroscedastic. Benjamini–Hochberg controls the false-discovery rate across the
ten per-city tests, and effect sizes separate statistical from practical
significance. Full results: `reports/statistical_analysis.md`.

### Visualisations

Ten figures in `outputs/figures/`, including per-city price distributions in
local currency (separate panels), a unitless room-type price index heatmap,
room-type composition, property families, review activity over time, review-volume
skew, ratings by room type, host response by city and relative price vs capacity.

---

## 8. Machine Learning track (C)

### Problem definition

**Target:** advertised nightly price in USD (`price_usd`), trained as
`log1p(price_usd)` and reported after the inverse transform. This is a price
*estimate*, not a revenue, occupancy or pricing recommendation.

**Sample:** 276,846 listings with a positive price inside their city's 1st–99th
percentile band.

### Feature sets

| Set | Contents | Purpose |
|---|---|---|
| **CORE** (deployed) | city, room type, property family, accommodates, bedrooms + missing flag, stay policy, amenity count and 11 amenity flags, latitude/longitude, host portfolio size, host tenure, response/acceptance rates + missing flags, response-time rank, superhost/profile/identity flags, instant-bookable | available when a listing is created |
| **EXTENDED** (sensitivity) | CORE + review count, reviewer count, reviews last 365 days, recency, listing age, review intensity, 7 rating scores | more accurate but unavailable for new listings |

Review-derived and target-derived fields are excluded from CORE to prevent
leakage and to make the deployed model usable for a brand-new listing.

### Models and evaluation

- Baseline 1: global median price.
- Baseline 2: **per-city median** (the fair baseline).
- Ridge regression (linear, one-hot encoded).
- Random forest (120 trees, minimum leaf size 10).
- Histogram gradient boosting (native categorical handling).
- 80/20 stratified split by city, `random_state=42`; 3-fold stratified CV as a
  stability check.
- Metrics: MAE, RMSE, MedAE, R² and percentage error, overall and per city.

Results, error analysis, feature importance and limitations are in
`reports/model_report.md`. The packaged model is
`outputs/models/price_model_core.joblib`.

### Using the model

```python
from src.inference import predict_price

predict_price({
    "city": "Rome", "room_type": "Entire place", "property_family": "apartment",
    "accommodates": 4, "bedrooms": 2, "minimum_nights": 2, "maximum_nights": 365,
    "amenity_count": 20, "latitude": 41.9028, "longitude": 12.4964,
    "host_total_listings_count": 3, "host_tenure_years": 6,
    "host_response_rate": 0.98, "host_acceptance_rate": 0.9,
    "host_response_time_rank": 1, "host_is_superhost": 1,
    "host_has_profile_pic": 1, "host_identity_verified": 1, "instant_bookable": 1,
})
# → {'predicted_price_usd': ..., 'predicted_price_local': ..., 'local_currency': 'EUR', ...}
```

Input validation rejects unknown cities, missing required fields and negative
capacity. Tests cover determinism, conversion consistency and failure modes.

---

## 9. Testing and validation

```bash
make test
```

- **Cleaning unit tests:** percentage/boolean parsing, mojibake repair, property
  family mapping, currency rates.
- **Data-contract tests:** unique `listing_id`, ten cities, positive prices,
  one currency per city, non-negative review counts, ratings within documented
  scales.
- **Inference tests:** positive prediction, USD→local conversion consistency,
  determinism, rejection of invalid input.
- **Analysis validation:** hypothesis results are reproducible from the stored
  analysis dataset; totals reconcile with the source row counts (5,373,143
  reviews, 279,712 listings).

### Success criteria and results

| Criterion | Target | Result |
|---|---|---|
| Held-out MAE improvement over the per-city median baseline | ≥ 20% | **37.5%** |
| Effect size reported for every hypothesis | 100% | 5/5 hypotheses |
| All ten cities covered in per-city tests | 10/10 | 10/10 |
| Notebooks execute end to end from the processed data | No errors | `make notebooks` |
| Test suite passes | 0 failures | `make test` |
| Calculations reconcile with source totals | Exact | 279,712 listings; 5,373,143 reviews |

---

## 10. Limitations

1. Advertised price ≠ revenue, occupancy or profit.
2. No review text or per-review ratings: sentiment and comment analysis are impossible.
3. One FX snapshot; cross-city USD comparisons inherit that assumption.
4. Review metrics are confounded by listing age and are not missing at random.
5. `district` is usable only for New York; neighbourhood granularity varies.
6. No bathrooms field; `bedrooms` missing for 10.5%.
7. Listings are observational and not a random market sample; p-values are optimistic
   because multi-listing hosts violate independence mildly.
8. Extreme advertised prices are excluded from training, so the model is
   conservative at the top of the market.
9. Associations are not causal effects.

---

## 11. Deliverables map

| Requirement (Track B) | Where |
|---|---|
| Reproducible Python notebook | `notebooks/02_exploratory_analysis.ipynb`, `notebooks/03_hypothesis_testing.ipynb` |
| Data-quality summary | `reports/data_quality_report.md` |
| ≥ 5 visualisations | `outputs/figures/ds_*.png` (10 figures) |
| Statistical analysis report | `reports/statistical_analysis.md` |
| Findings and recommendations | `reports/executive_summary.md` |
| Reproduction instructions | this README, `Makefile` |

| Requirement (Track C) | Where |
|---|---|
| Reproducible notebook / source | `notebooks/04_price_prediction.ipynb`, `src/modelling.py` |
| Preparation and feature documentation | §5, §8 above and `src/features.py` |
| Target and feature definitions | `outputs/models/model_metadata.json` |
| Baseline and comparison report | `reports/model_report.md`, `outputs/tables/model_comparison.csv` |
| Metrics and error analysis | `reports/model_report.md`, `outputs/tables/model_*.csv` |
| Feature importance | `outputs/tables/feature_importance_permutation.csv` |
| Saved model | `outputs/models/price_model_core.joblib` |
| Inference prototype | `src/inference.py` |
| Tests and README | `tests/`, this file |

---

## 12. Attribution and tools

- **Data:** Airbnb Listings and Reviews dataset supplied by AnalystLab Africa.
- **FX rates:** European Central Bank reference rates, 2026-11-11 snapshot.
- **Tools:** Python 3.9, pandas, Polars, NumPy, SciPy, scikit-learn,
  Matplotlib, Seaborn, Jupyter, pytest. No Microsoft Excel was used at any stage.

## 13. Professional development checklist

- [ ] Push the repository to GitHub.
- [ ] Publish a LinkedIn post tagging AnalystLab Africa with #AnalystLabAfrica.
- [ ] Publish an X post tagging @analystlabafric with #AnalystLabAfrica.
- [ ] Record a 5–10 minute presentation covering method, results, limitations and contribution.
