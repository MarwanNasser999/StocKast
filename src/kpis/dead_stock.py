"""
Dead Stock / Slow-Moving Product detection for src.kpis.

Dead stock: no sales in the most recent portion of the dataset's date
range (relative to the dataset's own timeline, not real-world "today").
Slow-moving: GROSS quantity sold (positive sales only, excluding
returns) falls in the bottom percentile of all products.

Returns (negative quantity_sold values) are tracked separately from
gross sales, so a product with heavy sales AND heavy returns doesn't
show a confusing negative or misleadingly low net number -- and isn't
unfairly flagged as "slow" just because returns offset its real sales.
"""

from __future__ import annotations

import pandas as pd

RECENT_WINDOW_FRACTION = 0.25   # last 25% of the date range counts as "recent"
SLOW_MOVER_PERCENTILE = 0.20     # bottom 20% of GROSS quantity sold


def compute_dead_and_slow_stock(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per product_id: gross_quantity_sold
    (positive sales only), returns (absolute value of negative
    quantities), net_quantity_sold (gross minus returns), last_sale_date,
    days_since_last_sale, is_dead_stock, is_slow_mover.
    """
    date_max = df["date"].max()
    date_min = df["date"].min()
    recent_cutoff = date_max - (date_max - date_min) * RECENT_WINDOW_FRACTION

    gross_sold = df[df["quantity_sold"] > 0].groupby("product_id")["quantity_sold"].sum()
    returns = df[df["quantity_sold"] < 0].groupby("product_id")["quantity_sold"].sum().abs()
    net_sold = df.groupby("product_id")["quantity_sold"].sum()

    totals = pd.DataFrame({
        "gross_quantity_sold": gross_sold,
        "returns": returns,
        "net_quantity_sold": net_sold,
    }).fillna(0).reset_index()

    last_sale = df.groupby("product_id")["date"].max().reset_index()
    last_sale.columns = ["product_id", "last_sale_date"]

    result = totals.merge(last_sale, on="product_id")
    result["days_since_last_sale"] = (date_max - result["last_sale_date"]).dt.days
    result["is_dead_stock"] = result["last_sale_date"] < recent_cutoff

    slow_threshold = result["gross_quantity_sold"].quantile(SLOW_MOVER_PERCENTILE)
    result["is_slow_mover"] = result["gross_quantity_sold"] <= slow_threshold

    return result