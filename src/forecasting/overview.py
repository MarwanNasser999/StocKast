"""
Orchestrator for src.forecasting.

Per product: builds a daily demand series, gets seasonality info,
selects the best-performing technique via a fair train/test comparison,
then retrains that winning technique on the FULL history to produce
the real future forecast.

Runs across products in parallel (ThreadPoolExecutor) since ARIMA
fitting per product is the main performance bottleneck at scale --
numpy/scipy/statsmodels release the GIL during their heavy C-level
computation, so threads give real speedup here without the
process-spawning fragility multiprocessing can have under Streamlit.

Accepts an optional on_progress callback rather than importing
Streamlit directly, keeping this module UI-agnostic and independently
testable/usable outside any specific interface.

Forecast results are cached based on the input dataset fingerprint.
If the same dataset is forecasted again, the cached results are
returned without rerunning Naive, ETS, or ARIMA.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Callable, Optional

import pandas as pd

from src.forecasting.arima import forecast_arima
from src.forecasting.exponential_smoothing import forecast_ets
from src.forecasting.forecast_cache import (
    load_cached_forecast,
    save_forecast_to_cache,
)
from src.forecasting.model_selection import select_best_model
from src.forecasting.naive import forecast_naive
from src.kpis.seasonality import detect_seasonality


FORECAST_HORIZON_DAYS = 14
MAX_WORKERS = 8


TECHNIQUE_FUNCTIONS = {
    "naive": lambda series, horizon, seasonality: forecast_naive(
        series, horizon, seasonality
    ),
    "ets": lambda series, horizon, seasonality: forecast_ets(
        series, horizon, seasonality
    ),
    "arima": lambda series, horizon, seasonality: forecast_arima(
        series, horizon, seasonality
    ),
}


def _build_daily_series(product_df: pd.DataFrame) -> pd.Series:
    daily = product_df.groupby(
        product_df["date"].dt.date
    )["quantity_sold"].sum()

    daily.index = pd.to_datetime(daily.index)

    full_range = pd.date_range(
        daily.index.min(),
        daily.index.max(),
        freq="D",
    )

    return daily.reindex(full_range, fill_value=0)


def forecast_one_product(product_df: pd.DataFrame) -> dict:
    """
    Returns a dict with the forecast Series, which technique won, its
    test-period MAE, and all techniques' scores -- or an 'error' key if
    forecasting wasn't possible for this product.
    """
    daily_series = _build_daily_series(product_df)

    seasonality_result = detect_seasonality(product_df)

    selection = select_best_model(
        daily_series,
        seasonality_result,
    )

    if "error" in selection:
        return selection

    winning_technique = selection["best_technique"]

    forecast_fn = TECHNIQUE_FUNCTIONS[winning_technique]

    final_forecast = forecast_fn(
        daily_series,
        FORECAST_HORIZON_DAYS,
        seasonality_result,
    )

    return {
        "technique_used": winning_technique,
        "test_mae": selection["best_mae"],
        "all_scores": selection["all_scores"],
        "forecast": final_forecast,
    }


def run_forecasting(
    df: pd.DataFrame,
    on_progress: Optional[Callable[[int, int], None]] = None,
) -> dict[str, dict]:
    """
    Runs forecasting for every product_id.

    Before running any models, checks whether a cached forecast exists
    for the exact input dataset.

    If a valid cache exists:
        - returns it immediately
        - skips all forecasting computation
        - reports all products as completed

    If no cache exists:
        - runs forecast_one_product for every product in parallel
        - saves the complete results to cache
        - returns the results
    """

    # ---------------------------------------------------------
    # 1. Try to load cached forecast
    # ---------------------------------------------------------
    cached = load_cached_forecast(df)

    if cached is not None:
        if on_progress:
            on_progress(len(cached), len(cached))

        return cached

    # ---------------------------------------------------------
    # 2. No cache -> run forecasting normally
    # ---------------------------------------------------------
    groups = list(df.groupby("product_id"))

    total = len(groups)

    results: dict[str, dict] = {}

    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_id = {
            executor.submit(
                forecast_one_product,
                group,
            ): product_id
            for product_id, group in groups
        }

        completed = 0

        for future in as_completed(future_to_id):
            product_id = future_to_id[future]

            try:
                results[product_id] = future.result()

            except Exception as exc:
                results[product_id] = {
                    "error": str(exc)
                }

            completed += 1

            if on_progress:
                on_progress(completed, total)

    # ---------------------------------------------------------
    # 3. Save completed forecasting results to cache
    # ---------------------------------------------------------
    save_forecast_to_cache(df, results)

    return results