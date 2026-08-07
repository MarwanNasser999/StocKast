"""
Distribution testing for src.analytics.distributions.

Formally tests whether a numeric field's distribution is approximately
normal (Shapiro-Wilk test), which matters for later modules (e.g.
forecasting) that may assume normality. Reuses skewness/kurtosis from
descriptive stats rather than recomputing them.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats as scipy_stats

from src.common.canonical_schema import field_is_available

NUMERIC_FIELDS = ["quantity_sold", "unit_price", "unit_cost", "current_stock"]

# Shapiro-Wilk becomes unreliable/overly sensitive on very large samples --
# a random subsample keeps the test meaningful without changing our
# interpretation of "is this roughly normal".
MAX_SAMPLE_SIZE_FOR_TEST = 5000
SIGNIFICANCE_THRESHOLD = 0.05


def test_normality(df: pd.DataFrame, field: str) -> dict:
    """Shapiro-Wilk normality test for one numeric field."""
    series = df[field].dropna()

    if len(series) < 3:
        return {"error": f"Not enough data in '{field}' to test normality."}

    if len(series) > MAX_SAMPLE_SIZE_FOR_TEST:
        series = series.sample(MAX_SAMPLE_SIZE_FOR_TEST, random_state=42)

    statistic, p_value = scipy_stats.shapiro(series)
    is_normal = bool(p_value >= SIGNIFICANCE_THRESHOLD)

    interpretation = (
        f"'{field}' appears approximately normally distributed (p={p_value:.4f})."
        if is_normal else
        f"'{field}' does NOT appear normally distributed (p={p_value:.4f}) -- "
        f"likely skewed or heavy-tailed, common for real sales/inventory data."
    )

    return {
        "field": field,
        "shapiro_statistic": float(statistic),
        "p_value": float(p_value),
        "is_normal": is_normal,
        "interpretation": interpretation,
    }


def test_all_distributions(df: pd.DataFrame) -> list[dict]:
    """Run the normality test for every numeric field actually present."""
    available = [f for f in NUMERIC_FIELDS if field_is_available(df, f)]
    results = []

    for field in available:
        result = test_normality(df, field)
        if "error" not in result:
            results.append(result)

    return results