"""Unit tests for src.forecasting."""

import numpy as np
import pandas as pd

from src.forecasting.model_selection import select_best_model
from src.forecasting.naive import forecast_naive
from src.forecasting.overview import forecast_one_product, run_forecasting
from src.forecasting.seasonality_utils import select_seasonal_period
from src.kpis.seasonality import detect_seasonality


def make_seasonal_df(n_days=90):
    """Strong, clean weekly pattern: weekdays ~10, weekends ~30."""
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    values = [30 if d.weekday() >= 5 else 10 for d in dates]
    return pd.DataFrame({
        "date": dates,
        "product_id": ["A"] * n_days,
        "quantity_sold": values,
    })


def make_flat_noisy_df(n_days=90):
    """No real pattern -- random noise around a constant mean."""
    rng = np.random.default_rng(3)
    dates = pd.date_range("2024-01-01", periods=n_days, freq="D")
    values = rng.normal(20, 2, size=n_days).clip(min=0)
    return pd.DataFrame({
        "date": dates,
        "product_id": ["B"] * n_days,
        "quantity_sold": values,
    })


def build_daily_series(df):
    daily = df.groupby(df["date"].dt.date)["quantity_sold"].sum()
    daily.index = pd.to_datetime(daily.index)
    return daily


def test_select_seasonal_period_picks_weekly_when_detected():
    df = make_seasonal_df()
    seasonality_result = detect_seasonality(df)
    period = select_seasonal_period(seasonality_result)

    assert period == 7


def test_select_seasonal_period_none_when_not_seasonal():
    df = make_flat_noisy_df()
    seasonality_result = detect_seasonality(df)
    period = select_seasonal_period(seasonality_result)

    assert period is None


def test_naive_uses_seasonal_pattern_when_detected():
    df = make_seasonal_df()
    series = build_daily_series(df)
    seasonality_result = detect_seasonality(df)

    forecast = forecast_naive(series, 14, seasonality_result)

    assert len(forecast) == 14
    # weekend forecasts should be noticeably higher than weekday forecasts
    weekday_avg = forecast[[d.weekday() < 5 for d in forecast.index]].mean()
    weekend_avg = forecast[[d.weekday() >= 5 for d in forecast.index]].mean()
    assert weekend_avg > weekday_avg


def test_naive_flat_when_not_seasonal():
    df = make_flat_noisy_df()
    series = build_daily_series(df)
    seasonality_result = detect_seasonality(df)

    forecast = forecast_naive(series, 14, seasonality_result)

    assert forecast.nunique() == 1  # flat moving-average forecast


def test_select_best_model_returns_scores_for_all_working_techniques():
    df = make_seasonal_df()
    series = build_daily_series(df)
    seasonality_result = detect_seasonality(df)

    result = select_best_model(series, seasonality_result)

    assert "error" not in result
    assert result["best_technique"] in result["all_scores"]
    assert result["best_mae"] == min(result["all_scores"].values())


def test_select_best_model_insufficient_data():
    df = make_seasonal_df(n_days=10)
    series = build_daily_series(df)
    seasonality_result = detect_seasonality(df)

    result = select_best_model(series, seasonality_result)

    assert "error" in result


def test_forecast_one_product_end_to_end():
    df = make_seasonal_df()
    result = forecast_one_product(df)

    assert "error" not in result
    assert len(result["forecast"]) == 14
    assert result["technique_used"] in {"naive", "ets", "arima"}


def test_run_forecasting_covers_all_products():
    df = pd.concat([make_seasonal_df(), make_flat_noisy_df()], ignore_index=True)
    results = run_forecasting(df)

    assert set(results.keys()) == {"A", "B"}
    assert "error" not in results["A"]
    assert "error" not in results["B"]