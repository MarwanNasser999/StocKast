"""
Unit tests for src.eda.overview.

Proves run_eda() correctly includes/excludes charts and stats based on
which optional fields are actually present in the dataset.
"""

import pandas as pd

from src.eda.overview import run_eda


def make_full_df(n=90):
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "product_id": [f"SKU-{i%10:03d}" for i in range(n)],
        "product_name": [f"Product {i%10}" for i in range(n)],
        "category": ["Hardware", "Software"] * (n // 2),
        "quantity_sold": [10 + (i % 5) for i in range(n)],
        "unit_price": [19.99] * n,
        "unit_cost": [8.50] * n,
    })


def test_run_eda_reports_available_and_unavailable_fields():
    df = make_full_df()
    result = run_eda(df)

    assert "unit_price" in result.available_fields
    assert "warehouse_id" in result.unavailable_fields


def test_run_eda_includes_revenue_by_category_when_fields_present():
    df = make_full_df()
    result = run_eda(df)

    assert result.has_figure("revenue_by_category")
    assert "total_revenue" in result.stats


def test_run_eda_excludes_revenue_by_category_when_category_missing():
    df = make_full_df().drop(columns=["category"])
    result = run_eda(df)

    assert not result.has_figure("revenue_by_category")


def test_run_eda_always_includes_demand_over_time():
    minimal_df = pd.DataFrame({
        "date": pd.date_range("2024-01-01", periods=60, freq="D"),
        "product_id": [f"SKU-{i%5:03d}" for i in range(60)],
        "quantity_sold": [10] * 60,
    })
    result = run_eda(minimal_df)

    assert result.has_figure("demand_over_time")
    assert result.has_figure("top_products")


def test_run_eda_uses_product_name_as_label_when_available():
    df = make_full_df()
    result = run_eda(df)

    labels = list(result.figures["top_products"].data[0].y)
    assert any("Product" in str(label) for label in labels)