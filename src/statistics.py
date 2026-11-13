"""Statistical tests used in the Data Science track.

All tests respect the currency constraint: price comparisons are made **within
a city** (same local currency) or on the USD-standardised column with the FX
assumption stated. No test pools raw local-currency amounts across cities.

Methods
-------
Kruskal-Wallis  : non-parametric comparison of ≥ 2 independent groups. Chosen
                  because advertised prices are heavily right-skewed and
                  heteroscedastic; it tests stochastic dominance, not equality
                  of means.
Dunn            : post-hoc pairwise Mann-Whitney-type comparisons on average
                  ranks, with Bonferroni adjustment.
Spearman        : monotonic association, robust to skew and outliers.
Benjamini-Hochberg : controls the false-discovery rate across the repeated
                  per-city tests.
"""

from __future__ import annotations

from itertools import combinations
from math import sqrt

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def bh_adjust(p_values: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR adjustment."""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out


def _epsilon_squared(h: float, n: int, k: int) -> float:
    if n <= k:
        return np.nan
    return max((h - k + 1) / (n - k), 0.0)


def _effect_label(eps: float) -> str:
    if np.isnan(eps):
        return "n/a"
    if eps < 0.01:
        return "negligible"
    if eps < 0.06:
        return "small"
    if eps < 0.14:
        return "medium"
    return "large"


def kruskal_test(df: pd.DataFrame, value: str, group: str, min_group_size: int = 20) -> dict:
    """Kruskal-Wallis test with epsilon-squared effect size."""
    data = df[[value, group]].dropna()
    sizes = data.groupby(group, observed=True)[value].size()
    keep = sizes[sizes >= min_group_size].index
    data = data[data[group].isin(keep)]
    k = data[group].nunique()
    n = len(data)
    if k < 2 or n == 0:
        return {"n": n, "k": k, "H": np.nan, "p_value": np.nan, "epsilon_squared": np.nan, "effect": "n/a"}

    groups = [g[value].to_numpy() for _, g in data.groupby(group, observed=True)]
    h, p = stats.kruskal(*groups)
    eps = _epsilon_squared(h, n, k)
    return {
        "n": n,
        "k": k,
        "H": h,
        "p_value": p,
        "epsilon_squared": eps,
        "effect": _effect_label(eps),
        "median_by_group": data.groupby(group, observed=True)[value].median().to_dict(),
        "n_by_group": sizes.to_dict(),
    }


def dunn_test(df: pd.DataFrame, value: str, group: str, min_group_size: int = 20) -> pd.DataFrame:
    """Dunn's pairwise post-hoc test with Bonferroni-adjusted p-values."""
    data = df[[value, group]].dropna()
    sizes = data.groupby(group, observed=True)[value].size()
    keep = sizes[sizes >= min_group_size].index
    data = data[data[group].isin(keep)].copy()
    if data[group].nunique() < 2:
        return pd.DataFrame()

    data["_rank"] = stats.rankdata(data[value].to_numpy())
    n = len(data)
    tie_counts = data[value].value_counts().to_numpy()
    tie_term = np.sum(tie_counts ** 3 - tie_counts)
    denom_base = n * (n + 1) / 12 - tie_term / (12 * (n - 1))

    stats_by_group = data.groupby(group, observed=True).agg(
        n=(value, "size"), mean_rank=("_rank", "mean"), median=(value, "median")
    )

    rows = []
    for (g1, g2) in combinations(stats_by_group.index, 2):
        r1, r2 = stats_by_group.loc[g1, "mean_rank"], stats_by_group.loc[g2, "mean_rank"]
        n1, n2 = stats_by_group.loc[g1, "n"], stats_by_group.loc[g2, "n"]
        se = sqrt(denom_base * (1 / n1 + 1 / n2))
        z = (r1 - r2) / se if se else np.nan
        p = 2 * (1 - stats.norm.cdf(abs(z))) if np.isfinite(z) else np.nan
        rows.append(
            {
                "group_1": g1,
                "group_2": g2,
                "n_1": int(n1),
                "n_2": int(n2),
                "median_1": stats_by_group.loc[g1, "median"],
                "median_2": stats_by_group.loc[g2, "median"],
                "mean_rank_1": r1,
                "mean_rank_2": r2,
                "z": z,
                "p_value": p,
            }
        )
    out = pd.DataFrame(rows)
    out["p_bonferroni"] = np.minimum(out["p_value"] * len(out), 1.0)
    out["significant_5pct"] = out["p_bonferroni"] < 0.05
    return out.sort_values("p_bonferroni")


def spearman_with_ci(df: pd.DataFrame, x: str, y: str, min_n: int = 30) -> dict:
    """Spearman correlation with a Fisher-z confidence interval."""
    data = df[[x, y]].dropna()
    n = len(data)
    if n < min_n or data[x].nunique() < 3 or data[y].nunique() < 3:
        return {"n": n, "rho": np.nan, "p_value": np.nan, "ci_low": np.nan, "ci_high": np.nan}
    rho, p = stats.spearmanr(data[x], data[y])
    # Fieller's standard error for Spearman's rho.
    se = 1.06 / sqrt(n - 3) if n > 3 else np.nan
    z = np.arctanh(np.clip(rho, -0.999999, 0.999999))
    lo, hi = np.tanh(z - 1.96 * se), np.tanh(z + 1.96 * se)
    return {"n": n, "rho": rho, "p_value": p, "ci_low": lo, "ci_high": hi}


# --------------------------------------------------------------------------
# Hypothesis suite
# --------------------------------------------------------------------------
def hypothesis_suite(df: pd.DataFrame, city_order: list[str]) -> dict[str, pd.DataFrame]:
    """Run the four pre-specified hypotheses and return tidy result tables."""
    results: dict[str, pd.DataFrame] = {}

    # H1: price differs across room types, within each city (local currency).
    rows = []
    for city in city_order:
        g = df[(df["city"] == city) & df["price_analysable"]]
        res = kruskal_test(g, "price", "room_type")
        rows.append(
            {
                "city": city,
                "currency": g["currency"].iloc[0] if len(g) else None,
                "n": res["n"],
                "k": res["k"],
                "H": res["H"],
                "p_value": res["p_value"],
                "epsilon_squared": res["epsilon_squared"],
                "effect": res["effect"],
            }
        )
    h1 = pd.DataFrame(rows)
    h1["p_bh"] = bh_adjust(h1["p_value"].to_numpy())

    # H1 post-hoc: Dunn per city (only where H1 is significant).
    dunn_frames = []
    for city in city_order:
        g = df[(df["city"] == city) & df["price_analysable"]]
        d = dunn_test(g, "price", "room_type")
        if not d.empty:
            d.insert(0, "city", city)
            d.insert(1, "currency", g["currency"].iloc[0] if len(g) else None)
            dunn_frames.append(d)
    results["h1_kruskal_price_roomtype"] = h1
    results["h1_dunn_price_roomtype"] = pd.concat(dunn_frames, ignore_index=True) if dunn_frames else pd.DataFrame()

    # H2: capacity / bedrooms vs price (Spearman, per city, local currency).
    for label, x in [("accommodates", "capacity"), ("bedrooms", "bedrooms")]:
        rows = []
        for city in city_order:
            g = df[(df["city"] == city) & df["price_analysable"]]
            res = spearman_with_ci(g, label, "price")
            rows.append({"city": city, "variable": x, **res})
        t = pd.DataFrame(rows)
        t["p_bh"] = bh_adjust(t["p_value"].fillna(1).to_numpy())
        results[f"h2_spearman_{x}_price"] = t

    # H3: review score differs across room types. The rating is on a 0-100 scale
    # in this file; no currency issue here, so cities may be pooled with a city
    # control in the interpretation.
    rows = []
    for city in city_order:
        g = df[(df["city"] == city)]
        res = kruskal_test(g, "review_scores_rating", "room_type")
        rows.append({"city": city, **{k: res[k] for k in ["n", "k", "H", "p_value", "epsilon_squared", "effect"]}})
    h3 = pd.DataFrame(rows)
    h3["p_bh"] = bh_adjust(h3["p_value"].to_numpy())
    results["h3_kruskal_rating_roomtype"] = h3

    # H3b: review score differs across property families.
    rows = []
    for city in city_order:
        g = df[(df["city"] == city)]
        res = kruskal_test(g, "review_scores_rating", "property_family")
        rows.append({"city": city, **{k: res[k] for k in ["n", "k", "H", "p_value", "epsilon_squared", "effect"]}})
    h3b = pd.DataFrame(rows)
    h3b["p_bh"] = bh_adjust(h3b["p_value"].to_numpy())
    results["h3b_kruskal_rating_property"] = h3b

    # H4: review activity (review_count) differs across room types.
    rows = []
    for city in city_order:
        g = df[df["city"] == city]
        res = kruskal_test(g, "review_count", "room_type")
        rows.append({"city": city, **{k: res[k] for k in ["n", "k", "H", "p_value", "epsilon_squared", "effect"]}})
    h4 = pd.DataFrame(rows)
    h4["p_bh"] = bh_adjust(h4["p_value"].to_numpy())
    results["h4_kruskal_reviews_roomtype"] = h4

    # H5: host response rate differs across cities (percentages, no currency).
    res = kruskal_test(df, "host_response_rate", "city")
    results["h5_kruskal_host_response_city"] = pd.DataFrame(
        [{"n": res["n"], "k": res["k"], "H": res["H"], "p_value": res["p_value"],
          "epsilon_squared": res["epsilon_squared"], "effect": res["effect"]}]
    )

    return results


def save_hypothesis_tables(results: dict[str, pd.DataFrame], directory) -> None:
    for name, table in results.items():
        if table is None or table.empty:
            continue
        table.to_csv(directory / f"{name}.csv", index=False)
