"""
Safety Stock for src.kpis.

Safety Stock = Z * StdDev(daily demand) * sqrt(lead time in days)

Z reflects the desired service level (e.g. 1.65 for ~95%). Lead time
comes from the optional canonical field lead_time_days when available
per product; otherwise falls back to a caller-supplied default -- never
fabricated or silently assumed.
"""

from __future__ import annotations

import math

import pandas as pd

from src.common.canonical_schema import field_is_available


def compute_safety_stock(df: pd.DataFrame, service_level_z: float = 1.65,
                          default_lead_time_days: float | None = None) -> pd.DataFrame | None:
    """
    Returns a DataFrame with one row per product_id: std_daily_demand,
    lead_time_days_used, and safety_stock. Products with no available
    lead time (neither in the data nor a default provided) are excluded,
    not fabricated.
    """
    daily = df.groupby(["product_id", df["date"].dt.date])["quantity_sold"].sum().reset_index()
    std_demand = daily.groupby("product_id")["quantity_sold"].std().reset_index()
    std_demand.columns = ["product_id", "std_daily_demand"]
    std_demand["std_daily_demand"] = std_demand["std_daily_demand"].fillna(0)

    has_lead_time_field = field_is_available(df, "lead_time_days")

    if has_lead_time_field:
        lead_times = df.groupby("product_id")["lead_time_days"].mean().reset_index()
        lead_times.columns = ["product_id", "lead_time_days_used"]
        result = std_demand.merge(lead_times, on="product_id", how="left")

        if default_lead_time_days is not None:
            result["lead_time_days_used"] = result["lead_time_days_used"].fillna(default_lead_time_days)
    else:
        if default_lead_time_days is None:
            return None
        result = std_demand.copy()
        result["lead_time_days_used"] = default_lead_time_days

    result = result.dropna(subset=["lead_time_days_used"])
    if result.empty:
        return None

    result["safety_stock"] = (
        service_level_z * result["std_daily_demand"] * result["lead_time_days_used"].apply(math.sqrt)
    )

    return result