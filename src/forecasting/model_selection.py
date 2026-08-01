"""
Model comparison and selection for src.forecasting.

Fairly evaluates naive, ETS, and ARIMA by holding back the most recent
TEST_WINDOW_DAYS of real data, training each model on everything
before that, and measuring how close each model's predictions come to
the actual held-back values (MAE). The winner is then retrained on the
FULL history to produce the real future forecast.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.forecasting.arima import forecast_arima
from src.forecasting.exponential_smoothing import forecast_ets
from src.forecasting.naive import forecast_naive

TEST_WINDOW_DAYS = 14


def _mean_absolute_error(actual: pd.Series, predicted: pd.Series) -> float:
    aligned_actual, aligned_predicted = actual.align(predicted, join="inner")
    return float(np.mean(np.abs(aligned_actual - aligned_predicted)))


def select_best_model(daily_series: pd.Series, seasonality_result: dict) -> dict:
    """
    Returns a dict with the winning technique's name, its MAE on the
    held-back test window, and every technique's MAE for transparency
    -- 'explain model selection', per the original project brief.
    Returns an 'error' key if there isn't enough data to run a fair test.
    """
    if len(daily_series) < TEST_WINDOW_DAYS * 2:
        return {"error": f"Need at least {TEST_WINDOW_DAYS * 2} days of data to fairly evaluate models."}

    train = daily_series.iloc[:-TEST_WINDOW_DAYS]
    test = daily_series.iloc[-TEST_WINDOW_DAYS:]

    candidates = {}

    naive_pred = forecast_naive(train, TEST_WINDOW_DAYS, seasonality_result)
    candidates["naive"] = _mean_absolute_error(test, naive_pred)

    ets_pred = forecast_ets(train, TEST_WINDOW_DAYS, seasonality_result)
    if ets_pred is not None:
        candidates["ets"] = _mean_absolute_error(test, ets_pred)

    arima_pred = forecast_arima(train, TEST_WINDOW_DAYS, seasonality_result)
    if arima_pred is not None:
        candidates["arima"] = _mean_absolute_error(test, arima_pred)

    if not candidates:
        return {"error": "No forecasting technique could be fit to this data."}

    best_technique = min(candidates, key=candidates.get)

    return {
        "best_technique": best_technique,
        "best_mae": candidates[best_technique],
        "all_scores": candidates,
    }