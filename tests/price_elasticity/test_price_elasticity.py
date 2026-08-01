"""Unit tests for src.price_elasticity."""

import numpy as np
import pandas as pd

from src.price_elasticity.elasticity import compute_price_elasticity
from src.price_elasticity.overview import run_price_elasticity
from src.price_elasticity.what_if import project_price_change


def make_elastic_df(n_days=90):
    """Product with clear price variation and a real inverse price-demand
    relationship (elastic: price up -> demand down noticeably)."""
    rng = np.random.default_rng(5)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    prices = rng.choice([15.0, 20.0, 25.0, 30.0], size=n_days)
    # demand inversely related to price, plus small noise
    quantities = (500 / prices + rng.normal(0, 1, size=n_days)).clip(min=1)
    return pd.DataFrame({
        "date": dates, "product_id": ["A"] * n_days,
        "quantity_sold": quantities, "unit_price": prices,
    })


def make_constant_price_df(n_days=60):
    """Product sold at the exact same price every time -- no variation."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    return pd.DataFrame({
        "date": dates, "product_id": ["B"] * n_days,
        "quantity_sold": [10] * n_days, "unit_price": [20.0] * n_days,
    })


def test_compute_price_elasticity_returns_none_without_unit_price():
    df = make_elastic_df().drop(columns=["unit_price"])
    assert compute_price_elasticity(df) is None


def test_compute_price_elasticity_detects_negative_relationship():
    df = make_elastic_df()
    result = compute_price_elasticity(df)

    assert result is not None
    row = result[result["product_id"] == "A"].iloc[0]
    assert row["elasticity_coefficient"] < 0
    assert row["classification"] in {"elastic", "inelastic"}


def test_compute_price_elasticity_skips_products_without_price_variation():
    df = pd.concat([make_elastic_df(), make_constant_price_df()], ignore_index=True)
    result = compute_price_elasticity(df)

    assert "A" in result["product_id"].values
    assert "B" not in result["product_id"].values  # no variation, correctly excluded


def test_run_price_elasticity_reports_unavailable_reason_when_no_price():
    df = make_elastic_df().drop(columns=["unit_price"])
    result = run_price_elasticity(df)

    assert result.elasticity_table is None
    assert result.unavailable_reason is not None


def test_project_price_change_end_to_end():
    df = make_elastic_df()
    result = project_price_change(df, product_id="A", price_change_pct=10)

    assert "error" not in result
    assert result["new_price"] > result["current_price"]
    # elastic/negative relationship -> price increase should reduce projected units
    assert result["projected_units"] < result["baseline_units"]


def test_project_price_change_price_decrease_increases_projected_units():
    df = make_elastic_df()
    result = project_price_change(df, product_id="A", price_change_pct=-10)

    assert result["new_price"] < result["current_price"]
    assert result["projected_units"] > result["baseline_units"]


def test_project_price_change_errors_for_unknown_product():
    df = make_elastic_df()
    result = project_price_change(df, product_id="DOES_NOT_EXIST", price_change_pct=10)

    assert "error" in result


def test_project_price_change_never_goes_negative_on_extreme_input():
    df = make_elastic_df()
    result = project_price_change(df, product_id="A", price_change_pct=1000)

    assert result["projected_units"] >= 0