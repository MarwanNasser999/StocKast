"""
Reorder Point for src.kpis.

Reorder Point = (Average Daily Demand * Lead Time) + Safety Stock

Reuses compute_safety_stock() rather than recomputing lead time/std
logic separately, since reorder point is safety stock plus one more term.
"""

from __future__ import annotations

import pandas as pd

from src.kpis.safety_stock import compute_safety_stock


def compute_reorder_point(df: pd.DataFrame, service_level_z: float = 1.65,
                           default_lead_time_days: float | None = None) -> pd.DataFrame | None:
    """
    Returns a DataFrame with one row per product_id: avg_daily_demand,
    lead_time_days_used, safety_stock, and reorder_point. Returns None
    under the same conditions compute_safety_stock() would (no usable
    lead time available for any product).
    """
    safety = compute_safety_stock(df, service_level_z=service_level_z,
                                   default_lead_time_days=default_lead_time_days)
    if safety is None:
        return None

    daily = df.groupby(["product_id", df["date"].dt.date])["quantity_sold"].sum().reset_index()
    avg_demand = daily.groupby("product_id")["quantity_sold"].mean().reset_index()
    avg_demand.columns = ["product_id", "avg_daily_demand"]

    result = safety.merge(avg_demand, on="product_id", how="inner")

    result["reorder_point"] = (
        result["avg_daily_demand"] * result["lead_time_days_used"] + result["safety_stock"]
    )

    return result[["product_id", "avg_daily_demand", "lead_time_days_used", "safety_stock", "reorder_point"]]