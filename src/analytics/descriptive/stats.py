"""
Descriptive statistics for src.analytics.descriptive.

Computes distribution-shape statistics (spread, skewness, kurtosis) for
each numeric canonical field present in the dataset -- deeper than
eda's totals/sums, characterizing HOW the data is spread and shaped,
not just what it adds up to.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats as scipy_stats

from src.common.canonical_schema import field_is_available

NUMERIC_FIELDS = ["quantity_sold", "unit_price", "unit_cost", "current_stock"]


def compute_descriptive_stats(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    """
    Returns {field_name: {stat_name: value}} for every numeric canonical
    field actually present in the dataset. A field is simply absent from
    the result if it's not available (e.g. unit_cost wasn't mapped).
    """
    results: dict[str, dict[str, float]] = {}

    for field in NUMERIC_FIELDS:
        if not field_is_available(df.columns, field):
            continue

        series = df[field].dropna()
        if series.empty:
            continue

        results[field] = {
            "mean": float(series.mean()),
            "median": float(series.median()),
            "std": float(series.std()),
            "min": float(series.min()),
            "max": float(series.max()),
            "q1": float(series.quantile(0.25)),
            "q3": float(series.quantile(0.75)),
            "skewness": float(scipy_stats.skew(series)),
            "kurtosis": float(scipy_stats.kurtosis(series)),
        }

    return results