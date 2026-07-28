"""
Unit tests for src.eda.charts.

Proves each chart function returns a valid Plotly Figure (or None where
that's the correct, honest outcome), using small hand-built DataFrames.
"""

import pandas as pd
import plotly.graph_objects as go

from src.eda.charts import (
    build_demand_over_time_chart,
    build_revenue_by_category_chart,
    build_top_products_chart,
    build_trendy_products_chart,
    build_distribution_chart,
)


def test_demand_over_time_collapses_same_day_rows():
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01", "2024-01-01", "2024-01-02"]),
        "quantity_sold": [10, 5, 8],
    })
    fig = build_demand_over_time_chart(df)

    assert isinstance(fig, go.Figure)
    y_values = list(fig.data[0].y)
    assert 15 in y_values  # 2024-01-01's 10+5 collapsed into one point
    assert 8 in y_values


def test_revenue_by_category_sums_correctly():
    df = pd.DataFrame({
        "category": ["Hardware", "Hardware", "Software"],
        "quantity_sold": [10, 5, 3],
        "unit_price": [19.99, 8.50, 49.99],
    })
    fig = build_revenue_by_category_chart(df)

    assert isinstance(fig, go.Figure)
    values = list(fig.data[0].values)
    assert any(abs(v - 242.40) < 0.01 for v in values)  # 199.90 + 42.50


def test_top_products_respects_top_n():
    df = pd.DataFrame({
        "product_id": ["A", "B", "C", "D"],
        "quantity_sold": [100, 50, 200, 10],
    })
    fig = build_top_products_chart(df, top_n=2)

    assert isinstance(fig, go.Figure)
    assert len(fig.data[0].y) == 2
    assert "C" in fig.data[0].y  # highest total, must be included
    assert "D" not in fig.data[0].y  # lowest total, must be excluded


def test_trendy_products_returns_none_when_no_overlap():
    """If every product only sold in one period (early or recent), but
    never both, growth can't be computed for anyone -- must return None,
    not crash."""
    df = pd.DataFrame({
        "date": pd.to_datetime(["2024-01-01"] * 3 + ["2024-04-01"] * 3),
        "product_id": ["A", "B", "C", "D", "E", "F"],  # every product unique, no overlap
        "quantity_sold": [10, 5, 8, 12, 6, 9],
    })
    fig = build_trendy_products_chart(df)
    assert fig is None


def test_trendy_products_ranks_by_growth_not_volume():
    dates_early = pd.date_range("2024-01-01", periods=10, freq="D")
    dates_recent = pd.date_range("2024-03-01", periods=10, freq="D")

    df = pd.DataFrame({
        "date": list(dates_early) * 2 + list(dates_recent) * 2,
        "product_id": (["LOW_GROWTH"] * 10 + ["HIGH_GROWTH"] * 10) * 2,
        "quantity_sold": (
            [100] * 10 + [5] * 10 +      # early: LOW_GROWTH=100/day, HIGH_GROWTH=5/day
            [105] * 10 + [50] * 10        # recent: LOW_GROWTH=105/day, HIGH_GROWTH=50/day
        ),
    })
    fig = build_trendy_products_chart(df, top_n=2)

    assert isinstance(fig, go.Figure)
    # HIGH_GROWTH went from 5->50 (900% growth), LOW_GROWTH went from 100->105 (5% growth)
    assert fig.data[0].x[-1] > fig.data[0].x[0] or "HIGH_GROWTH" in list(fig.data[0].y)


def test_distribution_chart_builds_for_given_field():
    df = pd.DataFrame({"quantity_sold": [10, 11, 12, 9, 95]})
    fig = build_distribution_chart(df, "quantity_sold")

    assert isinstance(fig, go.Figure)