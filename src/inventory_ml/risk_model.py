"""
Stockout risk classifier for src.inventory_ml.

Builds WALK-FORWARD training examples: for each product, at multiple
snapshot dates through its history, features use only data up to that
date, and the label looks only at the following LOOKAHEAD_DAYS -- a
genuine point-in-time forecasting-style setup, not a single static
per-product label.

Forecast-related features are trailing-window statistics (slope, mean,
std, total of PAST demand), not a literal re-run of Phase 8's
naive/ETS/ARIMA pipeline at every snapshot -- that would require
thousands of ARIMA fits and be impractically slow. This is a documented
proxy, not the real forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

from src.inventory_ml.labeling import build_daily_stock_series, detect_stockout_dates

LOOKBACK_DAYS = 30      # how much trailing history to compute features from
LOOKAHEAD_DAYS = 14     # the actual prediction horizon: stockout within the next 14 days?
SNAPSHOT_INTERVAL_DAYS = 7  # sample one snapshot per product per week (keeps training set manageable, reduces redundant near-identical rows)
TEST_SIZE = 0.2
RANDOM_STATE = 42
RISK_HIGH_THRESHOLD = 0.66
RISK_MEDIUM_THRESHOLD = 0.33


def build_risk_table(model, feature_table: pd.DataFrame) -> pd.DataFrame:
    """
    Produces a standardized risk table: product_id, risk_label,
    risk_score, method -- same shape rule_based_risk.py produces, so
    consumers (recommendation_engine) never need to know this came
    from a trained model, predict_proba, or any feature details.
    Uses each product's MOST RECENT snapshot as its current risk.
    """
    feature_cols = ["days_of_inventory", "demand_volatility", "trailing_demand_slope", "trailing_total_demand"]

    latest_snapshots = (
        feature_table.sort_values("snapshot_date")
        .groupby("product_id")
        .tail(1)
    )

    probabilities = model.predict_proba(latest_snapshots[feature_cols])[:, 1]

    rows = []
    for (_, row), proba in zip(latest_snapshots.iterrows(), probabilities):
        if proba >= RISK_HIGH_THRESHOLD:
            label = "high"
        elif proba >= RISK_MEDIUM_THRESHOLD:
            label = "medium"
        else:
            label = "low"

        rows.append({
            "product_id": row["product_id"],
            "risk_label": label,
            "risk_score": float(proba),
            "method": "ml_classifier",
        })

    return pd.DataFrame(rows)


def _build_daily_demand_series(product_df: pd.DataFrame) -> pd.Series:
    daily = product_df.groupby(product_df["date"].dt.date)["quantity_sold"].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_range, fill_value=0)


def _compute_snapshot_features(daily_demand: pd.Series, daily_stock: pd.Series, snapshot_date) -> dict | None:
    """Features computed using ONLY data up to (and including) snapshot_date."""
    trailing_start = snapshot_date - pd.Timedelta(days=LOOKBACK_DAYS)
    trailing_demand = daily_demand[(daily_demand.index > trailing_start) & (daily_demand.index <= snapshot_date)]

    if len(trailing_demand) < LOOKBACK_DAYS // 2:  # need reasonably complete trailing history
        return None

    stock_at_snapshot = daily_stock.get(snapshot_date)
    if stock_at_snapshot is None or pd.isna(stock_at_snapshot):
        return None

    mean_demand = trailing_demand.mean()
    std_demand = trailing_demand.std()
    volatility = (std_demand / mean_demand) if mean_demand > 0 else 0.0

    x = np.arange(len(trailing_demand))
    slope = scipy_stats.linregress(x, trailing_demand.values).slope if len(trailing_demand) > 1 else 0.0

    days_of_inventory = (stock_at_snapshot / mean_demand) if mean_demand > 0 else np.inf

    return {
        "days_of_inventory": min(days_of_inventory, 365),  # cap extreme values
        "demand_volatility": volatility,
        "trailing_demand_slope": float(slope),
        "trailing_total_demand": float(trailing_demand.sum()),
    }


def build_feature_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Builds walk-forward (product, snapshot_date) training examples.
    Each row's features use only data up to snapshot_date; the label
    indicates whether a stockout event occurs in the following
    LOOKAHEAD_DAYS -- genuinely point-in-time, no future leakage.
    """
    rows = []

    for product_id, group in df.groupby("product_id"):
        daily_demand = _build_daily_demand_series(group)
        daily_stock = build_daily_stock_series(group)

        if daily_stock.empty:
            continue

        stockout_dates = set(detect_stockout_dates(group))

        # candidate snapshots: need enough trailing history AND enough
        # remaining future data to observe the full lookahead window
        earliest_valid = daily_stock.index.min() + pd.Timedelta(days=LOOKBACK_DAYS)
        latest_valid = daily_stock.index.max() - pd.Timedelta(days=LOOKAHEAD_DAYS)

        if earliest_valid > latest_valid:
            continue

        snapshot_dates = pd.date_range(earliest_valid, latest_valid, freq=f"{SNAPSHOT_INTERVAL_DAYS}D")

        for snapshot_date in snapshot_dates:
            features = _compute_snapshot_features(daily_demand, daily_stock, snapshot_date)
            if features is None:
                continue

            lookahead_end = snapshot_date + pd.Timedelta(days=LOOKAHEAD_DAYS)
            label = 1 if any(snapshot_date < d <= lookahead_end for d in stockout_dates) else 0

            features["product_id"] = product_id
            features["snapshot_date"] = snapshot_date
            features["stockout_occurred"] = label
            rows.append(features)

    return pd.DataFrame(rows)


def train_risk_model(feature_table: pd.DataFrame):
    """
    Trains a Random Forest classifier and returns (model, evaluation_metrics).
    Returns (None, {"error": ...}) if the feature table is too small or
    has only one class present.
    """
    feature_cols = ["days_of_inventory", "demand_volatility", "trailing_demand_slope", "trailing_total_demand"]
    X = feature_table[feature_cols]
    y = feature_table["stockout_occurred"]

    if y.nunique() < 2:
        return None, {"error": "All examples have the same outcome -- cannot train a classifier."}

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    model = RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, max_depth=5)
    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "n_train": len(X_train),
        "n_test": len(X_test),
    }

    return model, metrics