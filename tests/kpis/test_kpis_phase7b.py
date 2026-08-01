"""Unit tests for src.kpis Phase 7b (safety stock, reorder point,
dead/slow stock, seasonality)."""

import numpy as np
import pandas as pd

from src.kpis.dead_stock import compute_dead_and_slow_stock
from src.kpis.reorder_point import compute_reorder_point
from src.kpis.safety_stock import compute_safety_stock
from src.kpis.seasonality import detect_seasonality


def make_df(n_days=60):
    rng = np.random.default_rng(1)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    rows = []
    for d in dates:
        for product in ["A", "B"]:
            rows.append({"date": d, "product_id": product,
                         "quantity_sold": max(0, int(rng.normal(10, 3)))})
    return pd.DataFrame(rows)


def test_safety_stock_returns_none_without_lead_time():
    df = make_df()
    assert compute_safety_stock(df) is None


def test_safety_stock_uses_default_lead_time():
    df = make_df()
    result = compute_safety_stock(df, default_lead_time_days=5)

    assert result is not None
    assert (result["safety_stock"] >= 0).all()


def test_reorder_point_builds_on_safety_stock():
    df = make_df()
    result = compute_reorder_point(df, default_lead_time_days=5)

    assert result is not None
    assert (result["reorder_point"] > result["safety_stock"]).all()


def test_dead_and_slow_stock_flags_correctly():
    df = make_df()
    # zero out product B's last 20 days entirely -> should be dead stock
    df = df[~((df["product_id"] == "B") & (df["date"] > df["date"].max() - pd.Timedelta(days=20)))]

    result = compute_dead_and_slow_stock(df)
    product_b = result[result["product_id"] == "B"].iloc[0]

    assert product_b["is_dead_stock"] == True


def test_seasonality_detects_insufficient_data_for_yearly():
    df = make_df(n_days=60)
    result = detect_seasonality(df)

    assert "error" not in result["weekly"]
    assert "error" in result["yearly"]  # 60 days is nowhere near 730 needed