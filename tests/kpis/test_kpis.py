"""Unit tests for src.kpis (Phase 7a)."""

import pandas as pd

from src.kpis.abc_analysis import compute_abc_analysis
from src.kpis.days_of_inventory import compute_days_of_inventory
from src.kpis.overview import run_kpis
from src.kpis.turnover import compute_turnover
from src.kpis.xyz_analysis import compute_xyz_analysis


def make_df_with_stock():
    return pd.DataFrame({
        "date": pd.to_datetime(
            ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-01", "2024-01-02"]
        ),
        "product_id": ["A", "A", "A", "B", "B"],
        "quantity_sold": [10, 12, 8, 5, 6],
        "unit_price": [20.0, 20.0, 20.0, 50.0, 50.0],
        "current_stock": [100, 90, 82, 40, 34],
    })


def make_df_without_stock():
    df = make_df_with_stock()
    return df.drop(columns=["current_stock"])


def test_turnover_returns_none_without_current_stock():
    df = make_df_without_stock()
    assert compute_turnover(df) is None


def test_turnover_computes_ratio_with_stock():
    df = make_df_with_stock()
    result = compute_turnover(df)

    assert result is not None
    assert set(result["product_id"]) == {"A", "B"}
    assert (result["turnover_ratio"] > 0).all()


def test_days_of_inventory_uses_latest_stock():
    df = make_df_with_stock()
    result = compute_days_of_inventory(df)

    product_a = result[result["product_id"] == "A"].iloc[0]
    assert product_a["latest_stock"] == 82  # last recorded value for A, not the average


def test_abc_analysis_uses_revenue_when_price_available():
    df = make_df_with_stock()
    result = compute_abc_analysis(df)

    assert (result["basis"] == "revenue").all()
    assert set(result["tier"]).issubset({"A", "B", "C"})


def test_abc_analysis_falls_back_to_quantity_without_price():
    df = make_df_with_stock().drop(columns=["unit_price"])
    result = compute_abc_analysis(df)

    assert (result["basis"] == "quantity_sold").all()


def test_xyz_analysis_assigns_tiers():
    df = make_df_with_stock()
    result = compute_xyz_analysis(df)

    assert set(result["xyz_tier"]).issubset({"X", "Y", "Z"})
    assert "coefficient_of_variation" in result.columns


def test_run_kpis_produces_all_sections():
    df = make_df_with_stock()
    result = run_kpis(df)

    assert result.turnover is not None
    assert result.days_of_inventory is not None
    assert result.abc_analysis is not None
    assert result.xyz_analysis is not None