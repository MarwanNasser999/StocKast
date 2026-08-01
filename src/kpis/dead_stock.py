"""
Dead Stock / Slow-Moving Product detection for src.kpis.

Dead stock: no sales in the most recent portion of the dataset's date
range (relative to the dataset's own timeline, not real-world "today").
Slow-moving: total quantity sold falls in the bottom percentile of all
products -- relative to this dataset's own scale, not a fixed threshold.
"""

from __future__ import annotations

import pandas as pd

RECENT_WINDOW_FRACTION = 0.25   # last 25% of the date range counts as "recent"
SLOW_MOVER_PERCENTILE = 0.20     # bottom 20% of total quantity sold


def compute_dead_and_slow_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per product_id: total_quantity_sold,
    last_sale_date, days_since_last_sale, is_dead_stock, is_slow_mover.
    """
    date_max = df["date"].max()
    date_min = df["date"].min()
    recent_cutoff = date_max - (date_max - date_min) * RECENT_WINDOW_FRACTION

    totals = df.groupby("product_id")["quantity_sold"].sum().reset_index()
    totals.columns = ["product_id", "total_quantity_sold"]

    last_sale = df.groupby("product_id")["date"].max().reset_index()
    last_sale.columns = ["product_id", "last_sale_date"]

    result = totals.merge(last_sale, on="product_id")
    result["days_since_last_sale"] = (date_max - result["last_sale_date"]).dt.days
    result["is_dead_stock"] = result["last_sale_date"] < recent_cutoff

    slow_threshold = result["total_quantity_sold"].quantile(SLOW_MOVER_PERCENTILE)
    result["is_slow_mover"] = result["total_quantity_sold"] <= slow_threshold

    return result