"""
Days of Inventory (DOI) for src.kpis.

DOI = how many days current stock will last at the recent average daily
demand rate. Uses the MOST RECENT current_stock reading per product
(not an average across history), since DOI is inherently about "how
much do I have right now", unlike turnover which looks at the whole period.
"""

from __future__ import annotations

import pandas as pd

from src.common.canonical_schema import field_is_available


def compute_days_of_inventory(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Returns a DataFrame with one row per product_id: latest_stock,
    avg_daily_demand, days_of_inventory. Returns None if current_stock
    isn't available in this dataset.
    """
    if not field_is_available(df.columns, "current_stock"):
        return None

    df = df.sort_values("date")

    latest_stock = df.groupby("product_id")["current_stock"].last().reset_index()
    latest_stock.columns = ["product_id", "latest_stock"]

    date_span_days = (df["date"].max() - df["date"].min()).days
    date_span_days = max(date_span_days, 1)  # avoid division by zero on single-day data

    total_sold = df.groupby("product_id")["quantity_sold"].sum().reset_index()
    total_sold.columns = ["product_id", "total_quantity_sold"]
    total_sold["avg_daily_demand"] = total_sold["total_quantity_sold"] / date_span_days

    result = latest_stock.merge(total_sold, on="product_id")
    result = result[result["avg_daily_demand"] > 0].copy()
    result["days_of_inventory"] = result["latest_stock"] / result["avg_daily_demand"]

    return result[["product_id", "latest_stock", "avg_daily_demand", "days_of_inventory"]]