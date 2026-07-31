"""Unit tests for src.analytics.correlation.analysis."""

import pandas as pd

from src.analytics.correlation.analysis import compute_correlation, compute_all_correlations


def test_strong_negative_correlation_detected():
    df = pd.DataFrame({
        "unit_price": [10, 20, 30, 40, 50],
        "quantity_sold": [100, 80, 60, 40, 20],  # perfectly inverse
    })
    result = compute_correlation(df, "unit_price", "quantity_sold")

    assert result["pearson_r"] < -0.9
    assert "negative" in result["interpretation"]


def test_constant_field_returns_error():
    df = pd.DataFrame({
        "unit_price": [10, 20, 30, 40, 50],
        "current_stock": [5, 5, 5, 5, 5],  # constant, no variance
    })
    result = compute_correlation(df, "unit_price", "current_stock")

    assert "error" in result


def test_compute_all_correlations_only_uses_available_fields():
    df = pd.DataFrame({
        "quantity_sold": [10, 20, 30],
        "unit_price": [5, 10, 15],
        # no unit_cost, no current_stock
    })
    results = compute_all_correlations(df)

    assert len(results) == 1  # only one possible pair
    assert {results[0]["field_a"], results[0]["field_b"]} == {"quantity_sold", "unit_price"}