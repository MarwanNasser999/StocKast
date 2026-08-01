"""
ARIMA forecasting for src.forecasting.

Uses auto-ARIMA (pmdarima) to search for good (p,d,q) parameters rather
than hand-picking them -- this genuinely needs a search, unlike ETS's
simpler on/off seasonal choice. Seasonal period comes from the same
select_seasonal_period() used by naive/ETS, never assumed.
"""

from __future__ import annotations

import pandas as pd
import pmdarima as pm

from src.forecasting.seasonality_utils import select_seasonal_period


def forecast_arima(daily_series: pd.Series, horizon_days: int, seasonality_result: dict) -> pd.Series | None:
    """
    Fits an auto-selected (S)ARIMA model and forecasts horizon_days
    ahead. Returns None if there isn't enough data or fitting fails.
    """
    period = select_seasonal_period(seasonality_result)
    min_required = (period * 2) if period else 15
    if len(daily_series) < min_required:
        return None

    try:
        model = pm.auto_arima(
            daily_series,
            seasonal=period is not None,
            m=period if period else 1,
            suppress_warnings=True,
            error_action="ignore",
        )
        forecast_values = model.predict(n_periods=horizon_days)
    except Exception:
        return None

    last_date = daily_series.index.max()
    future_dates = pd.date_range(start=last_date + pd.Timedelta(days=1), periods=horizon_days, freq="D")

    return pd.Series(forecast_values, index=future_dates, name="forecast")