"""
Exponential Smoothing (ETS) forecasting for src.forecasting.

Seasonal period comes from kpis.seasonality's results (whichever
tested period actually showed real seasonality). Additive vs
multiplicative form is chosen by fitting both and keeping whichever
fits the historical data better, falling back to additive if the data
contains zeros (multiplicative is undefined there).
"""

from __future__ import annotations

import pandas as pd
import numpy as np
from statsmodels.tsa.holtwinters import ExponentialSmoothing

from src.forecasting.seasonality_utils import select_seasonal_period


def _fit_and_score(daily_series: pd.Series, trend: str | None, seasonal: str | None,
                    seasonal_periods: int | None):
    model = ExponentialSmoothing(
        daily_series, trend=trend, seasonal=seasonal, seasonal_periods=seasonal_periods,
    )
    fitted = model.fit()
    in_sample_error = np.mean(np.abs(fitted.fittedvalues - daily_series))
    return fitted, in_sample_error


def forecast_ets(daily_series: pd.Series, horizon_days: int, seasonality_result: dict) -> pd.Series | None:
    period = select_seasonal_period(seasonality_result)
    min_required = period * 2 if period else 10
    if len(daily_series) < min_required:
        return None

    has_zero_or_negative = (daily_series <= 0).any()
    forms_to_try = ["add"] if has_zero_or_negative else ["add", "mul"]

    candidates = []
    for form in forms_to_try:
        try:
            seasonal_arg = form if period else None
            fitted, error = _fit_and_score(daily_series, trend=form, seasonal=seasonal_arg,
                                            seasonal_periods=period)
            candidates.append((fitted, error))
        except Exception:
            continue

    if not candidates:
        return None

    best_fitted, _ = min(candidates, key=lambda pair: pair[1])
    return best_fitted.forecast(horizon_days).rename("forecast")