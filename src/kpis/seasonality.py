"""
Seasonality Detection for src.kpis.

Tests weekly, monthly, and yearly seasonality using seasonal
decomposition. Each period is only tested if there's enough data for at
least 2 full cycles -- otherwise that period is reported as
'not enough data', never skipped silently or guessed.
"""

from __future__ import annotations

import pandas as pd
from statsmodels.tsa.seasonal import seasonal_decompose

SEASONAL_STRENGTH_THRESHOLD = 0.3

PERIODS_TO_TEST = {
    "weekly": 7,
    "monthly": 30,
    "yearly": 365,
}


def _build_daily_series(df: pd.DataFrame) -> pd.Series:
    """Build a complete, gap-filled daily demand series (missing days = 0)."""
    daily = df.groupby(df["date"].dt.date)["quantity_sold"].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_range, fill_value=0)


def _test_one_period(series: pd.Series, period: int, label: str) -> dict:
    min_days_needed = period * 2

    if len(series) < min_days_needed:
        return {
            "period_label": label, "period_days": period,
            "error": f"Need at least {min_days_needed} days of data to test "
                     f"{label} seasonality (have {len(series)}).",
        }

    decomposition = seasonal_decompose(series, model="additive", period=period)

    seasonal_var = decomposition.seasonal.var()
    residual_var = decomposition.resid.dropna().var()
    total_var = seasonal_var + residual_var

    seasonal_strength = seasonal_var / total_var if total_var > 0 else 0.0
    is_seasonal = seasonal_strength > SEASONAL_STRENGTH_THRESHOLD

    return {
        "period_label": label, "period_days": period,
        "seasonal_strength": float(seasonal_strength),
        "is_seasonal": bool(is_seasonal),
        "interpretation": (
            f"Demand shows a {'clear' if is_seasonal else 'weak or no'} {label} pattern "
            f"(strength={seasonal_strength:.2f})."
        ),
    }


def detect_seasonality(df: pd.DataFrame) -> dict[str, dict]:
    """
    Tests weekly, monthly, and yearly seasonality. Returns
    {period_label: result_dict} -- each result either has the full
    strength/interpretation, or an 'error' key if there wasn't enough
    data for that specific period.
    """
    series = _build_daily_series(df)

    return {
        label: _test_one_period(series, period, label)
        for label, period in PERIODS_TO_TEST.items()
    }