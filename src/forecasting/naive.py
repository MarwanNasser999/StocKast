"""
Naive baseline forecasting for src.forecasting.

Chooses between seasonal naive (repeat value from N days ago) and
moving-average naive, using WHICHEVER period (weekly/monthly/yearly)
kpis.seasonality actually detected -- never assumes weekly specifically.
"""

from __future__ import annotations

import pandas as pd

from src.forecasting.seasonality_utils import select_seasonal_period

MOVING_AVERAGE_WINDOW = 7


def forecast_naive(daily_series: pd.Series, horizon_days: int, seasonality_result: dict) -> pd.Series:
    period = select_seasonal_period(seasonality_result)

    if period is not None and len(daily_series) >= period * 2:
        return _forecast_seasonal_naive(daily_series, horizon_days, period)
    return _forecast_moving_average_naive(daily_series, horizon_days)


def _forecast_seasonal_naive(daily_series: pd.Series, horizon_days: int, period: int) -> pd.Series:
    last_date = daily_series.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    extended = daily_series.copy()
    forecasts = []

    for date in future_dates:
        lookback_date = date - pd.Timedelta(days=period)
        value = extended.get(lookback_date, extended.iloc[-period:].mean())
        forecasts.append(value)
        extended[date] = value

    return pd.Series(forecasts, index=future_dates, name="forecast")


def _forecast_moving_average_naive(daily_series: pd.Series, horizon_days: int) -> pd.Series:
    last_date = daily_series.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    recent_average = daily_series.iloc[-MOVING_AVERAGE_WINDOW:].mean()
    forecasts = [recent_average] * horizon_days

    return pd.Series(forecasts, index=future_dates, name="forecast")