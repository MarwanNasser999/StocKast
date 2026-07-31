"""Unit tests for src.analytics.overview.run_analytics."""

import numpy as np
import pandas as pd

from src.analytics.overview import run_analytics


def make_full_df(n=100):
    rng = np.random.default_rng(1)
    return pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=n, freq="D"),
        "product_id": [f"SKU-{i%10}" for i in range(n)],
        "quantity_sold": rng.integers(5, 50, size=n),
        "unit_price": rng.uniform(5, 50, size=n),
        "category": (["A"] * 60) + (["B"] * 40),
    })


def test_run_analytics_produces_all_sections():
    df = make_full_df()
    result = run_analytics(df)

    assert "quantity_sold" in result.descriptive_stats
    assert len(result.correlations) >= 1
    assert len(result.hypothesis_tests) >= 1  # category is available
    assert len(result.distribution_tests) >= 1
    assert result.has_figure("correlation_heatmap")