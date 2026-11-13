# Target and Feature Definitions — Price Model

Companion to `src/features.py` and `outputs/models/model_metadata.json`.

## Target

| Field | Definition | Scale |
|---|---|---|
| `price` | Advertised nightly price exactly as supplied | Local currency of the listing's city |
| `price_usd` | `price × usd_per_unit(currency)` using the ECB 2026-11-11 snapshot | USD |
| `target_log_usd` | `log1p(price_usd)` — training target only | log USD |

Predictions are produced on the log scale and converted back with `expm1`.
The model estimates an **advertised price**, never revenue, occupancy or profit.

## Feature sets

### CORE — deployable, available when a listing is created

| Feature | Type | Source / derivation | Why it may matter |
|---|---|---|---|
| `city` | categorical (10) | supplied | Market level and currency context |
| `room_type` | categorical (4) | supplied | Privacy and exclusive use |
| `property_family` | categorical (~17) | derived from `property_type` | Dwelling type (apartment, house, hotel, …) |
| `accommodates` | numeric | supplied | Capacity |
| `bedrooms` | numeric | supplied | Size; imputed with the training median |
| `bedrooms_missing` | binary | `bedrooms.isna()` | Missingness can carry information |
| `minimum_nights` | numeric | supplied | Booking policy |
| `maximum_nights` | numeric | supplied | Booking policy |
| `amenity_count` | numeric | count of parsed amenity list | Breadth of offering |
| `has_wifi` … `has_long_term_stay` | 11 binary | keyword match on amenity list | Presence of key amenities |
| `latitude`, `longitude` | numeric | supplied | Fine-grained location |
| `host_total_listings_count` | numeric | supplied | Professional vs occasional host |
| `is_multilisting_host` | binary | count > 1 | Portfolio host indicator |
| `host_tenure_years` | numeric | snapshot − `host_since` | Experience |
| `host_response_rate` | numeric | parsed percentage | Responsiveness |
| `host_acceptance_rate` | numeric | parsed percentage | Selectivity |
| `host_response_rate_missing`, `host_acceptance_rate_missing` | binary | `isna()` | Missingness itself is informative |
| `host_response_time_rank` | ordinal 1–4 | mapped from `host_response_time` | Responsiveness |
| `host_is_superhost`, `host_has_profile_pic`, `host_identity_verified` | binary | `t`/`f` parsed | Trust signals |
| `instant_bookable` | binary | `t`/`f` parsed | Booking friction |

### EXTENDED — sensitivity analysis only (not deployable for a new listing)

Adds: `review_count`, `reviewer_count`, `reviews_last_365d`,
`days_since_last_review`, `listing_age_years`, `review_activity_per_year`,
`reviews_per_month`, `has_reviews`, and the seven `review_scores_*` fields.

These fields accrue only after a listing has been live and booked, so they are
excluded from the deployed model. The comparison quantifies how much accuracy
is lost by keeping the model deployable.

## Explicitly excluded fields

| Field | Reason |
|---|---|
| `name` | Free text; contains decorative and promotional language, and no text model is in scope |
| `host_location` | High-cardinality free text; host base is not the listing location |
| `district` | 86.8% missing; complete only for New York |
| `neighbourhood` | High cardinality, inconsistent across cities; location is carried by lat/long and city |
| `city_price_p01/p50/p99` | Target-derived; would leak the target |
| `price_above_city_median` | Target-derived |
| `price_valid`, `price_analysable` | Sample-selection flags, not features |
| `host_id`, `listing_id` | Identifiers; no predictive meaning and privacy risk |
| `amenities` (raw) | Replaced by count and binary flags |

## Preprocessing

1. Numeric: median imputation (median learned on the training fold only).
2. Categorical: category levels frozen on the training fold; unseen levels at
   inference map to `unknown`.
3. Encoding: one-hot for Ridge and Random Forest; native categorical bins for
   histogram gradient boosting.
4. No scaling is required for tree models; Ridge uses `StandardScaler`.
5. No target encoding is used, avoiding fold leakage.

## Split and validation

- 80/20 random split stratified by `city`, `random_state = 42`.
- 3-fold stratified cross-validation on the training portion for Ridge and
  gradient boosting.
- Metrics on the held-out 20%, reported overall and per city.

The split is stratified rather than grouped by host: host attributes are
legitimately available at prediction time, but the p-values and error bars
should be read with mild clustering by host in mind.
