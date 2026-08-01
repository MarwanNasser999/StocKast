"""
Shared seasonality-period selection for src.forecasting.

Used by both naive.py and exponential_smoothing.py, so the logic for
"which period should this product's forecast use" is defined exactly
once, not duplicated or reached into from another module's internals.
"""

from __future__ import annotations

PERIOD_DAYS = {"weekly": 7, "monthly": 30, "yearly": 365}


def select_seasonal_period(seasonality_result: dict) -> int | None:
    """
    Picks the shortest period that was actually detected as seasonal,
    from kpis.seasonality.detect_seasonality()'s full result dict.
    Shorter periods are more reliably estimated with less data; if a
    product is both weekly- and monthly-seasonal, weekly captures the
    pattern with a smaller, more data-efficient model. Returns None if
    no period was confirmed seasonal (or none could be tested).
    """
    for label in ["weekly", "monthly", "yearly"]:
        result = seasonality_result.get(label, {})
        if result.get("is_seasonal"):
            return PERIOD_DAYS[label]
    return None