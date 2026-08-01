"""
Orchestrator for src.forecasting.

Per product: builds a daily demand series, gets seasonality info,
selects the best-performing technique via a fair train/test comparison,
then retrains that winning technique on the FULL history to produce
the real future forecast.
"""

from __future__ import annotations

import pandas as pd

from src.forecasting.arima import forecast_arima
from src.forecasting.exponential_smoothing import forecast_ets
from src.forecasting.model_selection import select_best_model
from src.forecasting.naive import forecast_naive
from src.kpis.seasonality import detect_seasonality

FORECAST_HORIZON_DAYS = 14

TECHNIQUE_FUNCTIONS = {
    "naive": lambda series, horizon, seasonality: forecast_naive(series, horizon, seasonality),
    "ets": lambda series, horizon, seasonality: forecast_ets(series, horizon, seasonality),
    "arima": lambda series, horizon, seasonality: forecast_arima(series, horizon, seasonality),
}


def _build_daily_series(product_df: pd.DataFrame) -> pd.Series:
    daily = product_df.groupby(product_df["date"].dt.date)["quantity_sold"].sum()
    daily.index = pd.to_datetime(daily.index)
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    return daily.reindex(full_range, fill_value=0)


def forecast_one_product(product_df: pd.DataFrame) -> dict:
    """
    Returns a dict with the forecast Series, which technique won, its
    test-period MAE, and all techniques' scores -- or an 'error' key if
    forecasting wasn't possible for this product.
    """
    daily_series = _build_daily_series(product_df)
    seasonality_result = detect_seasonality(product_df)

    selection = select_best_model(daily_series, seasonality_result)
    if "error" in selection:
        return selection

    winning_technique = selection["best_technique"]
    forecast_fn = TECHNIQUE_FUNCTIONS[winning_technique]
    final_forecast = forecast_fn(daily_series, FORECAST_HORIZON_DAYS, seasonality_result)

    return {
        "technique_used": winning_technique,
        "test_mae": selection["best_mae"],
        "all_scores": selection["all_scores"],
        "forecast": final_forecast,
    }


def run_forecasting(df: pd.DataFrame) -> dict[str, dict]:
    """Runs forecast_one_product for every product_id in the dataset."""
    return {
        product_id: forecast_one_product(group)
        for product_id, group in df.groupby("product_id")
    }