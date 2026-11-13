"""Train, evaluate and package the Airbnb advertised-price model.

Target: ``log1p(price_usd)`` -- advertised price per night, standardised to USD
with the documented ECB snapshot. Predictions are returned in USD after the
inverse transform and can be converted to any market currency with the
documented rate.

Run: ``python -m src.modelling``
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.inspection import permutation_importance  # noqa: F401  (kept for reference)
from sklearn.metrics import mean_absolute_error, mean_squared_error, median_absolute_error, r2_score
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split

from src.config import (
    ANALYSIS_DATASET,
    CITY_ORDER,
    FIGURES,
    FX_SNAPSHOT_DATE,
    MODELS,
    RANDOM_STATE,
    TABLES,
    TEST_SIZE,
)
from src.features import (
    CORE_CATEGORICAL,
    CORE_NUMERIC,
    EXTENDED_NUMERIC,
    make_pipeline,
    prepare_model_frame,
)

sns.set_theme(style="whitegrid", context="notebook")
plt.rcParams["figure.dpi"] = 120


# --------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------
def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mask = y_true > 0
    mape = float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)
    return {
        "n": int(len(y_true)),
        "MAE_usd": mae,
        "RMSE_usd": rmse,
        "MedAE_usd": median_absolute_error(y_true, y_pred),
        "R2": r2_score(y_true, y_pred),
        "MAPE_pct": mape,
        "median_abs_pct_error": float(np.median(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100),
    }


def predict_usd(model, X: pd.DataFrame) -> np.ndarray:
    return np.expm1(model.predict(X))


def permutation_importance_usd(
    model, X: pd.DataFrame, y_usd: np.ndarray, n_repeats: int = 5, sample_n: int = 5000, seed: int = RANDOM_STATE
) -> pd.DataFrame:
    """Permutation importance measured on the USD scale.

    sklearn's helper scores the *raw* pipeline output, which is on the log
    scale, against a USD target. That mismatch makes the resulting numbers
    meaningless, so the shuffling is done explicitly here and every prediction
    is inverted with ``expm1`` before the MAE is computed.
    """
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(X), size=min(sample_n, len(X)), replace=False)
    Xs = X.iloc[idx].reset_index(drop=True)
    ys = y_usd[idx]
    baseline = float(np.mean(np.abs(predict_usd(model, Xs) - ys)))

    rows = []
    for column in Xs.columns:
        original = Xs[column].to_numpy(copy=True)
        deltas = []
        for _ in range(n_repeats):
            shuffled = Xs.copy()
            shuffled[column] = original[rng.permutation(len(Xs))]
            deltas.append(float(np.mean(np.abs(predict_usd(model, shuffled) - ys)) - baseline))
        rows.append(
            {
                "feature": column,
                "importance_mae_usd": float(np.mean(deltas)),
                "std": float(np.std(deltas)),
            }
        )
    out = pd.DataFrame(rows).sort_values("importance_mae_usd", ascending=False).reset_index(drop=True)
    out.attrs["baseline_mae_usd"] = baseline
    return out


def select_best_model(comparison: pd.DataFrame, candidates: list[str]) -> str:
    """Pick the deployed model with a documented, pre-registered rule.

    1. Models within 1% of the best holdout MAE are treated as statistically
       indistinguishable in absolute error.
    2. Among those, prefer the highest R² (better fit to the price variance),
       then the lowest MAE.
    3. Remaining ties break on fit time and artifact size.

    This stops a 0.5% MAE difference from choosing a model that is 10× more
    expensive to train and far larger to ship while fitting the variance worse.
    """
    table = comparison[comparison["model"].isin(candidates)].copy()
    best_mae = table["MAE_usd"].min()
    tied = table[table["MAE_usd"] <= best_mae * 1.01]
    tied = tied.sort_values(["R2", "MAE_usd", "fit_seconds"], ascending=[False, True, True])
    return str(tied.iloc[0]["model"])


def evaluate(model, X, y, label, train_metrics=None) -> dict:
    pred = predict_usd(model, X)
    row = {"model": label, **regression_metrics(y, pred)}
    if train_metrics is not None:
        row["train_MAE_usd"] = train_metrics["MAE_usd"]
        row["train_R2"] = train_metrics["R2"]
        row["MAE_gap_pct"] = 100 * (row["MAE_usd"] - train_metrics["MAE_usd"]) / train_metrics["MAE_usd"]
    return row


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def main() -> dict:
    started = time.time()
    df = pd.read_parquet(ANALYSIS_DATASET)
    df = prepare_model_frame(df)

    model_df = df[df["price_analysable"] & df["target_log_usd"].notna()].copy()
    model_df = model_df[model_df["price_usd"] > 0]
    del df  # free the full frame; the modelling sample is self-contained

    X_core = model_df[CORE_NUMERIC + CORE_CATEGORICAL]
    y_usd = model_df["price_usd"].to_numpy()
    y_log = model_df["target_log_usd"].to_numpy()
    city = model_df["city"].astype(str).to_numpy()

    idx_train, idx_test = train_test_split(
        np.arange(len(model_df)),
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=city,
    )
    Xtr, Xte = X_core.iloc[idx_train], X_core.iloc[idx_test]
    ytr_usd, yte_usd = y_usd[idx_train], y_usd[idx_test]
    city_tr, city_te = city[idx_train], city[idx_test]

    results: list[dict] = []

    # ------------------------------------------------------------- baselines
    global_median = float(np.median(ytr_usd))
    city_medians = pd.Series(ytr_usd).groupby(city_tr).median()
    base_global = np.full(len(yte_usd), global_median)
    base_city = pd.Series(city_te).map(city_medians).fillna(global_median).to_numpy()

    results.append({"model": "baseline_global_median", **regression_metrics(yte_usd, base_global)})
    results.append({"model": "baseline_city_median", **regression_metrics(yte_usd, base_city)})

    # --------------------------------------------------------------- models
    trained: dict[str, object] = {}
    train_metrics: dict[str, dict] = {}

    for kind, label in [("ridge", "ridge_regression"), ("forest", "random_forest"), ("hist_gb", "hist_gradient_boosting")]:
        print(f"Training {label} ...", flush=True)
        pipe = make_pipeline(kind, CORE_NUMERIC, CORE_CATEGORICAL)
        t0 = time.time()
        pipe.fit(Xtr, np.log1p(ytr_usd))
        elapsed = time.time() - t0
        train_pred = predict_usd(pipe, Xtr)
        tr_m = regression_metrics(ytr_usd, train_pred)
        result = evaluate(pipe, Xte, yte_usd, label, tr_m)
        result["fit_seconds"] = round(elapsed, 1)
        results.append(result)
        trained[label] = pipe
        train_metrics[label] = tr_m
        print(f"  {label}: test MAE ${result['MAE_usd']:.2f}, R2 {result['R2']:.3f}, {elapsed:.1f}s", flush=True)

    comparison = pd.DataFrame(results).sort_values("MAE_usd")
    comparison.to_csv(TABLES / "model_comparison.csv", index=False)

    # ---------------------------------------------------- extended feature set
    best_label = select_best_model(comparison, list(trained))
    best_baseline_mae = comparison[comparison["model"] == "baseline_city_median"]["MAE_usd"].min()
    kind_by_label = {
        "ridge_regression": "ridge",
        "random_forest": "forest",
        "hist_gradient_boosting": "hist_gb",
    }
    extended_row = None
    if best_label in kind_by_label:
        print("Training extended feature set (gradient boosting) ...", flush=True)
        # A single fast, strong learner isolates the effect of the extra review
        # features without conflating feature-set and model-family comparisons.
        kind = "hist_gb"
        X_ext = model_df[EXTENDED_NUMERIC + CORE_CATEGORICAL]
        pipe_ext = make_pipeline(kind, EXTENDED_NUMERIC, CORE_CATEGORICAL)
        pipe_ext.fit(X_ext.iloc[idx_train], np.log1p(ytr_usd))
        tr_ext = regression_metrics(ytr_usd, predict_usd(pipe_ext, X_ext.iloc[idx_train]))
        extended_row = evaluate(pipe_ext, X_ext.iloc[idx_test], yte_usd, "hist_gb_extended_features", tr_ext)
        comparison = pd.concat([comparison, pd.DataFrame([extended_row])], ignore_index=True)
        comparison.to_csv(TABLES / "model_comparison.csv", index=False)
        trained["hist_gb_extended_features"] = pipe_ext

    # ------------------------------------------------------ cross-validation
    cv_rows = []
    skf = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)
    for kind, label in [("ridge", "ridge_regression"), ("hist_gb", "hist_gradient_boosting")]:
        pipe = make_pipeline(kind, CORE_NUMERIC, CORE_CATEGORICAL)
        scores = cross_val_score(
            pipe, X_core, y_log, cv=skf.split(X_core, city), scoring="neg_mean_absolute_error", n_jobs=1
        )
        cv_rows.append(
            {
                "model": label,
                "cv_folds": 3,
                "cv_MAE_log_usd_mean": float(-scores.mean()),
                "cv_MAE_log_usd_std": float(scores.std()),
            }
        )
    cv_table = pd.DataFrame(cv_rows)
    cv_table.to_csv(TABLES / "model_cross_validation.csv", index=False)

    # ------------------------------------------------------- per-city results
    city_rows = []
    for label in ["baseline_city_median", *trained.keys()]:
        if label == "baseline_city_median":
            pred = base_city
        else:
            model = trained[label]
            cols = EXTENDED_NUMERIC + CORE_CATEGORICAL if "extended" in label else CORE_NUMERIC + CORE_CATEGORICAL
            pred = predict_usd(model, model_df[cols].iloc[idx_test])
        for c in CITY_ORDER:
            mask = city_te == c
            if mask.sum() == 0:
                continue
            city_rows.append({"model": label, "city": c, **regression_metrics(yte_usd[mask], pred[mask])})
    by_city = pd.DataFrame(city_rows)
    by_city.to_csv(TABLES / "model_metrics_by_city.csv", index=False)

    # ---------------------------------------------------------- interpretation
    best_model = trained[best_label]
    importance = permutation_importance_usd(best_model, Xte, yte_usd)
    importance.to_csv(TABLES / "feature_importance_permutation.csv", index=False)

    # ------------------------------------------------------------ error analysis
    best_pred = predict_usd(best_model, Xte)
    errors = model_df.iloc[idx_test][["listing_id", "city", "room_type", "property_family", "price_usd", "currency"]].copy()
    errors["prediction_usd"] = best_pred
    errors["residual_usd"] = errors["price_usd"] - errors["prediction_usd"]
    errors["abs_pct_error"] = (errors["residual_usd"].abs() / errors["price_usd"]) * 100
    errors.to_parquet(MODELS / "test_predictions.parquet", index=False)

    error_by_city = (
        errors.groupby("city", observed=True)
        .agg(
            n=("residual_usd", "size"),
            median_residual=("residual_usd", "median"),
            mean_abs_error=("residual_usd", lambda s: s.abs().mean()),
            median_abs_pct_error=("abs_pct_error", "median"),
            p90_abs_pct_error=("abs_pct_error", lambda s: s.quantile(0.90)),
        )
        .reset_index()
    )
    error_by_city.to_csv(TABLES / "model_error_by_city.csv", index=False)

    errors["price_band"] = pd.qcut(
        errors["price_usd"], q=5, labels=["Q1 cheapest", "Q2", "Q3", "Q4", "Q5 dearest"], duplicates="drop"
    )
    error_by_band = (
        errors.groupby("price_band", observed=True)
        .agg(n=("residual_usd", "size"), mae=("residual_usd", lambda s: s.abs().mean()),
             median_ape=("abs_pct_error", "median"))
        .reset_index()
    )
    error_by_band.to_csv(TABLES / "model_error_by_price_band.csv", index=False)

    top_errors = errors.reindex(errors["abs_pct_error"].sort_values(ascending=False).index).head(25)
    top_errors.to_csv(TABLES / "model_largest_errors.csv", index=False)

    # --------------------------------------------------------------- figures
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.scatter(yte_usd, best_pred, s=4, alpha=0.25, edgecolors="none")
    lim = np.percentile(np.concatenate([yte_usd, best_pred]), 99.5)
    ax.plot([0, lim], [0, lim], "r--", lw=1)
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Actual price (USD)")
    ax.set_ylabel("Predicted price (USD)")
    ax.set_title(f"Predicted vs actual — {best_label}")
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_predicted_vs_actual.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    sns.histplot(errors["residual_usd"], bins=80, ax=ax)
    ax.set_xlim(errors["residual_usd"].quantile(0.01), errors["residual_usd"].quantile(0.99))
    ax.set_xlabel("Residual (USD), actual − predicted")
    ax.set_title("Residual distribution on the held-out test set")
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_residuals.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8, 6))
    top_imp = importance.head(15).iloc[::-1]
    ax.barh(top_imp["feature"], top_imp["importance_mae_usd"], color="#4C72B0")
    ax.set_xlabel("Increase in test MAE when shuffled (USD)")
    ax.set_title("Permutation importance — core feature set")
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_feature_importance.png")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(9, 4))
    plot_df = by_city[by_city["model"] == best_label]
    sns.barplot(data=plot_df, x="city", y="MAE_usd", ax=ax, color="#55A868")
    ax.set_title("Test MAE by city (USD-standardised target)")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(FIGURES / "ml_mae_by_city.png")
    plt.close(fig)

    # ---------------------------------------------------------- save artifacts
    artifact = MODELS / "price_model_core.joblib"
    joblib.dump(best_model, artifact)

    metadata = {
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_family": best_label,
        "feature_set": "core",
        "target": "advertised price per night in USD (trained on log1p, reported after expm1)",
        "currency_assumption": f"ECB reference rates snapshot {FX_SNAPSHOT_DATE}",
        "features": CORE_NUMERIC + CORE_CATEGORICAL,
        "n_training_rows": int(len(idx_train)),
        "n_test_rows": int(len(idx_test)),
        "test_metrics": comparison.set_index("model").loc[best_label].to_dict(),
        "city_median_baseline_MAE_usd": float(best_baseline_mae),
        "mae_reduction_vs_city_median_pct": float(
            100 * (best_baseline_mae - comparison.set_index("model").loc[best_label, "MAE_usd"]) / best_baseline_mae
        ),
        "artifacts": {
            "model": str(artifact.relative_to(MODELS.parent.parent)),
            "comparison": "outputs/tables/model_comparison.csv",
            "by_city": "outputs/tables/model_metrics_by_city.csv",
            "importance": "outputs/tables/feature_importance_permutation.csv",
        },
        "limitations": [
            "Predicts advertised price, not revenue, occupancy or profit.",
            "USD conversion uses a single ECB snapshot and ignores FX volatility.",
            "Listings are observational; the model describes association, not causation.",
            "New listings lack review history, so the core feature set is used for deployment.",
            "No bathrooms / bed-count field exists in the supplied schema.",
        ],
    }
    (MODELS / "model_metadata.json").write_text(json.dumps(metadata, indent=2, default=str))

    print("\nModel comparison:\n", comparison.to_string(index=False))
    print(f"\nTotal runtime: {time.time() - started:.1f}s")
    return metadata


if __name__ == "__main__":
    main()
