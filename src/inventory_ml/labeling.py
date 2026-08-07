"""
Stockout label construction for src.inventory_ml.

Detects genuine stockout events (stock > 0 -> stock <= 0) using a
forward-filled DAILY stock series per product, not raw sparse readings
-- this correctly handles irregular reading schedules and gives an
accurate day-by-day view of stock level over time.
"""

from __future__ import annotations

import pandas as pd

from src.common.canonical_schema import field_is_available

MIN_STOCK_READINGS_PER_PRODUCT = 5
MIN_STOCKOUT_EVENTS = 30


def build_daily_stock_series(product_df: pd.DataFrame) -> pd.Series:
    """Forward-filled daily stock level: the stock stays at its last
    known reading until a new reading changes it."""
    readings = product_df.dropna(subset=["current_stock"])
    if readings.empty:
        return pd.Series(dtype=float)

    daily = readings.groupby(readings["date"].dt.date)["current_stock"].last()
    daily.index = pd.to_datetime(daily.index)

    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_range).ffill()


def detect_stockout_dates(product_df: pd.DataFrame) -> pd.DatetimeIndex:
    """Dates where a genuine stock > 0 -> stock <= 0 transition occurred."""
    daily_stock = build_daily_stock_series(product_df)
    if len(daily_stock) < 2:
        return pd.DatetimeIndex([])

    previous_positive = daily_stock.shift(1) > 0
    currently_zero_or_below = daily_stock <= 0
    return daily_stock.index[previous_positive & currently_zero_or_below]


def count_total_stockout_events(df: pd.DataFrame) -> int:
    """Total genuine stockout events across all products -- used for the
    upfront 'is there enough data' gate, not for per-snapshot labeling."""
    if not field_is_available(df, "current_stock"):
        return 0

    total = 0
    for _, group in df.groupby("product_id"):
        total += len(detect_stockout_dates(group))
    return total


def can_build_ml_classifier(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Returns (True, "") if there's enough data to train a real classifier,
    or (False, reason) explaining why to fall back to a rule-based
    approach instead.
    """
    if not field_is_available(df, "current_stock"):
        return False, "current_stock is not available in this dataset."

    readings_per_product = df.groupby("product_id")["current_stock"].apply(
        lambda s: s.dropna().nunique()
    )
    products_with_real_history = (readings_per_product >= MIN_STOCK_READINGS_PER_PRODUCT).sum()

    if products_with_real_history == 0:
        return False, ("current_stock does not vary over time -- appears to be a single "
                        "snapshot, not historical tracking.")

    stockout_events = count_total_stockout_events(df)
    if stockout_events < MIN_STOCKOUT_EVENTS:
        return False, (
            f"Only {stockout_events} historical stockout event(s) found "
            f"(need at least {MIN_STOCKOUT_EVENTS}) -- not enough examples to train a reliable model."
        )

    return True, ""