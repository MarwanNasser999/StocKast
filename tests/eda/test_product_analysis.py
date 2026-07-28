"""
Unit tests for src.eda.product_analysis.
"""

import pandas as pd
import pytest

from src.eda.product_analysis import run_product_eda, get_product_display_options


def make_df():
    dates = pd.date_range("2024-01-01", periods=20, freq="D")
    return pd.DataFrame({
        "date": list(dates) + list(dates),
        "product_id": ["SKU-001"] * 20 + ["SKU-002"] * 20,
        "product_name": ["Wireless Mouse"] * 20 + ["USB Cable"] * 20,
        "quantity_sold": [10] * 20 + [5] * 20,
        "unit_price": [20.0] * 20 + [10.0] * 20,
        "unit_cost": [8.0] * 20 + [4.0] * 20,
    })


def test_get_product_display_options_uses_names():
    df = make_df()
    options = get_product_display_options(df)

    assert options["SKU-001"] == "Wireless Mouse"
    assert options["SKU-002"] == "USB Cable"


def test_run_product_eda_filters_to_correct_product():
    df = make_df()
    result = run_product_eda(df, "SKU-001")

    assert result.product_id == "SKU-001"
    assert result.display_name == "Wireless Mouse"
    assert result.stats["total_units_sold"] == 200  # 10 * 20 days


def test_run_product_eda_computes_profit_when_price_and_cost_available():
    df = make_df()
    result = run_product_eda(df, "SKU-001")

    # (20.0 - 8.0) * 10 * 20 days = 2400
    assert abs(result.stats["total_profit"] - 2400.0) < 0.01


def test_run_product_eda_raises_for_unknown_product():
    df = make_df()
    with pytest.raises(ValueError):
        run_product_eda(df, "DOES-NOT-EXIST")