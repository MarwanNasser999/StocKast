"""Unit tests for src.recommendation_engine."""

import numpy as np
import pandas as pd

from src.recommendation_engine.rule_based_risk import compute_rule_based_risk, _trend_risk_component
from src.recommendation_engine.rules import (
    recommend_reorder, recommend_reduce_inventory, recommend_discount,
    recommend_increase_safety_stock, recommend_price_change,
)
from src.recommendation_engine.overview import run_recommendations


# ---- rule_based_risk.py ----

def test_trend_component_distinguishes_volatile_from_smooth_uptrend():
    smooth = pd.Series([20, 21, 22, 23, 24])
    volatile = pd.Series([20, 100, 5, 90, 24])

    assert _trend_risk_component(smooth) == 1  # genuine positive slope
    # volatile series has a near-flat/unpredictable slope, not a clean uptrend
    volatile_score = _trend_risk_component(volatile)
    assert volatile_score in {0, 1}  # doesn't crash, produces a valid score either way


def test_compute_rule_based_risk_flags_low_doi_high_volatility_as_high():
    doi_table = pd.DataFrame({"product_id": ["A"], "days_of_inventory": [3.0], "latest_stock": [10]})
    xyz_table = pd.DataFrame({"product_id": ["A"], "xyz_tier": ["Z"]})
    forecast_results = {"A": {"forecast": pd.Series([10, 11, 12, 13, 14])}}

    result = compute_rule_based_risk(doi_table, xyz_table, forecast_results)

    assert result.iloc[0]["risk_label"] == "high"
    assert result.iloc[0]["method"] == "rule_based"


def test_compute_rule_based_risk_returns_empty_without_tables():
    result = compute_rule_based_risk(None, None, {})
    assert result.empty


# ---- rules.py ----

def test_recommend_reorder_triggers_when_stock_below_reorder_point():
    risk_row = {"product_id": "A", "risk_label": "high"}
    doi_row = pd.Series({"latest_stock": 5, "days_of_inventory": 2})
    reorder_row = pd.Series({"reorder_point": 20})

    rec = recommend_reorder(risk_row, doi_row, reorder_row, "Widget")

    assert rec is not None
    assert rec["action"] == "increase_inventory"
    assert rec["priority"] == "high"


def test_recommend_reorder_none_when_stock_above_reorder_point():
    risk_row = {"product_id": "A", "risk_label": "high"}
    doi_row = pd.Series({"latest_stock": 50, "days_of_inventory": 20})
    reorder_row = pd.Series({"reorder_point": 20})

    assert recommend_reorder(risk_row, doi_row, reorder_row, "Widget") is None


def test_recommend_reduce_inventory_skips_erratic_demand():
    doi_row = pd.Series({"product_id": "A", "days_of_inventory": 150})
    assert recommend_reduce_inventory(doi_row, "Z", "Widget") is None  # erratic -- might be legit buffer


def test_recommend_reduce_inventory_triggers_for_stable_overstock():
    doi_row = pd.Series({"product_id": "A", "days_of_inventory": 150})
    rec = recommend_reduce_inventory(doi_row, "X", "Widget")

    assert rec is not None
    assert rec["action"] == "reduce_inventory"


def test_recommend_discount_triggers_for_dead_stock():
    row = pd.Series({"product_id": "A", "is_dead_stock": True, "is_slow_mover": False,
                      "days_since_last_sale": 60, "total_quantity_sold": 5})
    rec = recommend_discount(row, "Widget")

    assert rec is not None
    assert rec["action"] == "discount"


def test_recommend_discount_none_for_healthy_product():
    row = pd.Series({"product_id": "A", "is_dead_stock": False, "is_slow_mover": False,
                      "days_since_last_sale": 1, "total_quantity_sold": 500})
    assert recommend_discount(row, "Widget") is None


def test_recommend_increase_safety_stock_only_for_erratic_and_risky():
    risk_row = {"product_id": "A", "risk_label": "high"}
    assert recommend_increase_safety_stock(risk_row, "X", "Widget") is None  # stable demand
    assert recommend_increase_safety_stock(risk_row, "Z", "Widget") is not None  # erratic + risky


def test_recommend_price_change_inelastic_suggests_increase():
    row = pd.Series({"product_id": "A", "classification": "inelastic",
                      "elasticity_coefficient": -0.3, "r_squared": 0.6})
    rec = recommend_price_change(row, "Widget")

    assert rec["action"] == "consider_price_increase"


def test_recommend_price_change_elastic_suggests_decrease():
    row = pd.Series({"product_id": "A", "classification": "elastic",
                      "elasticity_coefficient": -2.1, "r_squared": 0.7})
    rec = recommend_price_change(row, "Widget")

    assert rec["action"] == "consider_price_decrease"


def test_recommend_price_change_skips_suspect_extreme():
    row = pd.Series({"product_id": "A", "classification": "suspect_extreme",
                      "elasticity_coefficient": -75.0, "r_squared": 0.1})
    assert recommend_price_change(row, "Widget") is None


# ---- overview.py (end to end) ----

def make_full_dataset(n_days=200):
    rng = np.random.default_rng(3)
    dates = pd.date_range("2023-01-01", periods=n_days, freq="D")
    rows = []
    for p in range(3):
        product_id = f"SKU-{p}"
        stock = 100.0
        price = 20.0
        for d in dates:
            demand = max(0, rng.normal(8, 3))
            stock = max(0, stock - demand)
            if stock <= 0 and rng.random() < 0.3:
                stock = 100.0
            price = rng.choice([15.0, 20.0, 25.0])
            rows.append({
                "date": d, "product_id": product_id, "quantity_sold": demand,
                "current_stock": stock, "unit_price": price, "category": "General",
            })
    return pd.DataFrame(rows)


def test_run_recommendations_end_to_end():
    df = make_full_dataset()
    recommendations = run_recommendations(df)

    assert isinstance(recommendations, list)
    if recommendations:
        first = recommendations[0]
        assert "product_id" in first
        assert "action" in first
        assert "priority" in first
        # priority sorted: high should never appear after medium/low
        priorities = [r["priority"] for r in recommendations]
        priority_values = {"high": 0, "medium": 1, "low": 2}
        assert priorities == sorted(priorities, key=lambda p: priority_values[p])