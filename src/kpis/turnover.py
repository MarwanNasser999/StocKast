"""
Inventory Turnover for src.kpis.

Turnover = how many times inventory is sold through, relative to
average inventory held.

Two supported cases, since datasets vary in how much stock-history
detail they provide:
1. current_stock varies meaningfully over time per product (real
   historical snapshots) -> true average inventory across the period.
2. current_stock is effectively constant per product (a single,
   point-in-time snapshot) -> use that single value as an approximation,
   clearly labeled as such.
"""

from __future__ import annotations

import pandas as pd

from src.common.canonical_schema import field_is_available


def compute_turnover(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Returns a DataFrame with one row per product_id: total_quantity_sold,
    avg_inventory, turnover_ratio, and inventory_data_type
    ("historical_average" or "single_snapshot_approximation").
    Returns None if current_stock isn't available in this dataset.
    """
    if not field_is_available(df, "current_stock"):
        return None

    grouped = df.groupby("product_id").agg(
        total_quantity_sold=("quantity_sold", "sum"),
        avg_inventory=("current_stock", "mean"),
        stock_variation=("current_stock", "std"),
    ).reset_index()

    grouped["stock_variation"] = grouped["stock_variation"].fillna(0)
    grouped["inventory_data_type"] = grouped["stock_variation"].apply(
        lambda std: "historical_average" if std > 0 else "single_snapshot_approximation"
    )

    grouped = grouped[grouped["avg_inventory"] > 0].copy()
    grouped["turnover_ratio"] = grouped["total_quantity_sold"] / grouped["avg_inventory"]

    
    if grouped.empty:
        return None
    
    return grouped.drop(columns=["stock_variation"])
    