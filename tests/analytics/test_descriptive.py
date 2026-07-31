"""Unit tests for src.analytics.descriptive.stats."""

import pandas as pd

from src.analytics.descriptive.stats import compute_descriptive_stats


def make_df():
    return pd.DataFrame({
        "quantity_sold": [10, 12, 8, 15, 9, 11, 100],  # includes one outlier
        "unit_price": [19.99] * 7,
    })


def test_computes_stats_for_required_field():
    df = make_df()
    result = compute_descriptive_stats(df)

    assert "quantity_sold" in result
    assert result["quantity_sold"]["mean"] > 0
    assert "skewness" in result["quantity_sold"]
    assert "kurtosis" in result["quantity_sold"]


def test_skips_missing_optional_field():
    df = make_df()  # no unit_cost column at all
    result = compute_descriptive_stats(df)

    assert "unit_cost" not in result


def test_outlier_produces_positive_skew():
    df = make_df()  # 100 is a big upward outlier
    result = compute_descriptive_stats(df)

    assert result["quantity_sold"]["skewness"] > 0