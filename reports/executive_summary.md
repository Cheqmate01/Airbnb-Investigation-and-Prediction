# Executive Summary — Airbnb Market Intelligence

**Scope:** Data Science (Track B) and Machine Learning (Track C)
**Data:** 279,712 listings and 5,373,143 review records across 10 cities; review history ends 2021-03-01.
**Currency:** local prices converted to USD at the ECB 2026-11-11 reference snapshot; all statistical price tests run within a city and its local currency.

---

## What the analysis answers

| Question | Answer |
|---|---|
| Which cities have the most listings? | Paris (64,690; 23.1%) and New York (37,012; 13.2%). Hong Kong is smallest (7,087; 2.5%) — a 9.1× range. |
| How do prices vary? | Right-skewed everywhere. Median advertised price ranges from about **$5/night in Istanbul** to **$99 in New York** (USD-standardised), but local medians are the correct basis for within-market comparison. |
| Which room types dominate? | Entire places are 65.1% of listings, private rooms 31.1%, hotel rooms 2.1%, shared rooms 1.7%. |
| What drives price? | Room type (ε² ≈ 0.19) and capacity (ρ ≈ 0.52) are the largest and most consistent signals. Location adds substantially in the model. |
| Do ratings differ by category? | Statistically yes, practically barely: median ε² ≈ 0.005–0.007. Ratings cluster near the top of the 0–100 scale. |
| Can price be predicted? | Yes, reasonably: held-out MAE **$31.75**, R² **0.581**, a **37.5%** MAE improvement over the per-city median baseline. |

---

## Six findings

**1. The market is concentrated, and listing counts are not demand.**
Paris alone contributes almost a quarter of the extract. Counts describe the
supplied sample, not live market share, and cannot be read as occupancy.

**2. Advertised prices are heavily skewed in every market.**
The mean sits 27–28% above the median in Paris and New York and 80–87% above it
in Rio de Janeiro and Cape Town. Means therefore overstate the typical listing;
the analysis uses medians and rank-based tests throughout.

**3. Room type is the strongest categorical price differentiator.**
Kruskal–Wallis rejects equal price distributions in **10/10** cities, with a
median ε² of **0.188** (large). Shared rooms price below private rooms, which
price below entire places. Hotel rooms break the pattern, pricing above entire
places in Bangkok, Cape Town, Mexico City, New York and Paris.

**4. Capacity is the strongest continuous driver, but it is not the whole story.**
Spearman ρ between `accommodates` and price is significant in **10/10** cities,
median **0.52** (Sydney 0.68, Bangkok 0.31); bedrooms reach a median of **0.47**.
Moderate correlations mean capacity explains part of the price signal; location,
property type and positioning explain the rest.

**5. Ratings are a weak discriminator; review volume is not satisfaction.**
Rating differences across room types and property families are significant only
because the sample is large — median ε² is **0.005** and **0.007**, i.e.
negligible. Review activity differs more (median ε² **0.047**) but is confounded
with listing age, city demand and host behaviour. Only **69.2%** of listings have
any review, and ratings exist for **67.3%**, so both metrics describe a
self-selected subset.

**6. A transparent model meaningfully beats the naive alternative.**
Against the per-city median baseline (MAE $50.81), gradient boosting reaches
MAE **$31.75**, RMSE **$67.99**, R² **0.581** and median absolute percentage
error **26.0%**. It uses only attributes available at listing creation. Adding
review history improves MAE to **$30.87** (R² 0.600) but is not deployable for a
new listing. Errors are heteroscedastic: median absolute percentage error is
**36.3%** in the cheapest quintile versus **20.4%** in the middle and **30.3%**
in the dearest. The largest individual errors are implausibly cheap listings
($7–11 per night) predicted at $160–200, pointing to price-entry anomalies rather
than model failure alone.

---

## Four recommendations

**R1 — Compare markets with medians, inside local currency, and segment by room
type and capacity.**
*Evidence:* skew in every city (Finding 2), room-type effect (H1), capacity
effect (H2). *Action:* make "city × room type × capacity band" the default
reporting grain; never average raw prices across cities. *Evaluation:* next
refresh should reproduce the same rank ordering of cities by median price within
each room type; flag any market where the ordering flips without a documented
cause.

**R2 — Stop treating ratings as a price signal or a quality ranking.**
*Evidence:* negligible effect sizes for ratings (H3), small effect for review
activity (H4), and 33% of listings have no rating. *Action:* keep ratings as
context in listing profiles, but do not use them to justify price levels or to
rank markets. Report review volume separately and label it as activity, not
satisfaction. *Evaluation:* an internal review of any pricing memo should find no
rating-based price claim; dashboards should label review metrics unambiguously.

**R3 — Ship the price model as a bounded estimate, not a pricing instruction.**
*Evidence:* MAE $31.75 and 26% median percentage error; per-city MAE ranges from
$3.17 (Istanbul) to $44.55 (Sydney); the cheapest quintile has the largest
percentage error. *Action:* present a prediction interval (for example ±1 MAE)
alongside the point estimate, suppress or flag estimates for listings priced far
below their city's first percentile, and retrain per city if local precision
matters more than pooled sample size. *Evaluation:* track realised prediction
error after deployment and compare against the $31.75 benchmark; investigate any
city whose error drifts by more than 20%.

**R4 — Request the data that the current extract cannot provide.**
*Evidence:* no bathrooms, calendar/availability, booking outcomes, review text
or price date; `bedrooms` missing for 10.5%; district usable only for New York.
*Action:* request bathrooms/bed count, availability and booking records, review
text with per-review ratings, and the date each price was observed. *Evaluation:*
re-run the model with bathrooms and availability; a material R² gain would
confirm that the missing structural features — not model choice — are the binding
constraint.

---

## What the evidence cannot support

- **Revenue, occupancy or profit.** The data contain advertised prices and review
  events, not bookings or stays.
- **Guest sentiment.** The reviews file has dates and IDs only; there is no text
  and no per-review rating.
- **Causal claims.** Room type and capacity are *associated* with price. Hosts
  choose their listing attributes, so selection effects are unavoidable.
- **Cross-city price levels as observed facts.** USD figures depend on one FX
  snapshot; the local-currency tests do not.
- **A pricing recommendation for an individual host.** The model is a market-level
  estimator with material error, especially at the cheap end.

---

## Success criteria achieved

| Criterion | Target | Result |
|---|---|---|
| Beat the per-city median baseline on held-out MAE | ≥ 20% reduction | **37.5%** |
| Report effect sizes, not just p-values | Every hypothesis | 5/5 hypotheses include ε² or ρ |
| Cover all ten cities | 10/10 | **10/10** in every per-city test |
| Reproducible end to end | One command | `make all` |
| Tests pass | No failures | See `make test` |
| Honest limitations | Documented | Data-quality, statistical and model reports |
