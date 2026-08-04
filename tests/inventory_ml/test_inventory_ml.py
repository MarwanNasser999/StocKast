"""Unit tests for src.inventory_ml."""

import numpy as np
import pandas as pd

from src.inventory_ml.labeling import (
    build_daily_stock_series, detect_stockout_dates, can_build_ml_classifier,
)
from src.inventory_ml.overview import run_inventory_ml
from src.inventory_ml.risk_model import build_feature_table, train_risk_model
from src.inventory_ml.explainability import build_explainer, explain_product_risk


def make_repeated_stockout_df(n_products=5, n_days=400):
    """Multiple products, each cycling through restock -> deplete ->
    stockout repeatedly, enough to generate real walk-forward examples."""
    rng = np.random.default_rng(11)
    rows = []
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")

    for p in range(n_products):
        product_id = f"SKU-{p}"
        stock = 100.0
        for d in dates:
            demand = max(0, rng.normal(8, 3))
            stock = max(0, stock - demand)
            if stock <= 0 and rng.random() < 0.3:
                stock = 100.0  # restock after depletion, sometimes
            rows.append({
                "date": d, "product_id": product_id,
                "quantity_sold": demand, "current_stock": stock,
            })
    return pd.DataFrame(rows)


def make_no_variation_df():
    dates = pd.date_range("2024-01-01", periods=60, freq="D")
    return pd.DataFrame({
        "date": dates, "product_id": ["A"] * 60,
        "quantity_sold": [10] * 60, "current_stock": [50] * 60,
    })


def test_build_daily_stock_series_forward_fills():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-05"]),
        "product_id": ["A", "A"],
        "current_stock": [50, 30],
    })
    series = build_daily_stock_series(df)

    assert series[pd.Timestamp("2024-01-03")] == 50  # forward-filled
    assert series[pd.Timestamp("2024-01-05")] == 30


def test_detect_stockout_dates_finds_real_transitions():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-10", "2024-01-12", "2024-01-20"]),
        "product_id": ["A"] * 4,
        "current_stock": [50, 5, 0, 40],
    })
    dates = detect_stockout_dates(df)

    assert pd.Timestamp("2024-01-12") in dates


def test_can_build_ml_classifier_true_with_enough_events():
    df = make_repeated_stockout_df()
    can_build, reason = can_build_ml_classifier(df)

    assert can_build is True


def test_can_build_ml_classifier_false_without_variation():
    df = make_no_variation_df()
    can_build, reason = can_build_ml_classifier(df)

    assert can_build is False
    assert "does not vary" in reason


def test_build_feature_table_produces_multiple_rows_per_product():
    df = make_repeated_stockout_df()
    table = build_feature_table(df)

    assert not table.empty
    assert table["stockout_occurred"].nunique() == 2  # both classes present
    # each product should contribute more than one snapshot row
    assert (table.groupby("product_id").size() > 1).any()


def test_train_risk_model_produces_metrics():
    df = make_repeated_stockout_df()
    table = build_feature_table(df)
    model, metrics = train_risk_model(table)

    assert model is not None
    assert 0 <= metrics["accuracy"] <= 1
    assert metrics["n_train"] > 0
    assert metrics["n_test"] > 0


def test_run_inventory_ml_unavailable_without_current_stock():
    df = make_no_variation_df().drop(columns=["current_stock"])
    result = run_inventory_ml(df)

    assert result["path"] == "unavailable"


def test_run_inventory_ml_end_to_end():
    df = make_repeated_stockout_df()
    result = run_inventory_ml(df)

    assert result["path"] == "ml_classifier"
    assert "accuracy" in result["metrics"]


def test_explain_product_risk_produces_summary():
    df = make_repeated_stockout_df()
    result = run_inventory_ml(df)

    assert result["path"] == "ml_classifier"
    feature_table = result["feature_table"]
    explainer = result["explainer"]

    sample_row = feature_table.iloc[0]
    explanation = explain_product_risk(explainer, sample_row)

    assert "top_factor" in explanation
    assert explanation["top_factor"] in [
        "days_of_inventory", "demand_volatility", "trailing_demand_slope", "trailing_total_demand"
    ]
    assert len(explanation["contributions"]) == 4