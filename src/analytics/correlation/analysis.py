"""
Correlation analysis for src.analytics.correlation.

Measures how strongly pairs of numeric canonical fields move together,
using both Pearson (linear) and Spearman (monotonic, distribution-free)
correlation -- reporting both since they can reveal different things
about a relationship.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats as scipy_stats

from src.common.canonical_schema import field_is_available

NUMERIC_FIELDS = ["quantity_sold", "unit_price", "unit_cost", "current_stock"]


def _interpret_strength(coefficient: float) -> str:
    """Plain-language strength label for a correlation coefficient."""
    magnitude = abs(coefficient)
    if magnitude < 0.1:
        return "negligible"
    elif magnitude < 0.3:
        return "weak"
    elif magnitude < 0.5:
        return "moderate"
    elif magnitude < 0.7:
        return "strong"
    else:
        return "very strong"


def compute_correlation(df: pd.DataFrame, field_a: str, field_b: str) -> dict:
    """Compute Pearson and Spearman correlation between two numeric fields."""
    paired = df[[field_a, field_b]].dropna()

    if len(paired) < 3:
        return {"error": f"Not enough overlapping data between '{field_a}' and '{field_b}' to compute correlation."}

    if paired[field_a].nunique() < 2 or paired[field_b].nunique() < 2:
        return {"error": f"'{field_a}' or '{field_b}' has no variation (constant value) -- correlation is undefined."}

    pearson_r, pearson_p = scipy_stats.pearsonr(paired[field_a], paired[field_b])
    spearman_r, spearman_p = scipy_stats.spearmanr(paired[field_a], paired[field_b])

    direction = "positive" if pearson_r > 0 else "negative"
    strength = _interpret_strength(pearson_r)

    return {
        "field_a": field_a, "field_b": field_b,
        "pearson_r": float(pearson_r), "pearson_p_value": float(pearson_p),
        "spearman_r": float(spearman_r), "spearman_p_value": float(spearman_p),
        "interpretation": (
            f"'{field_a}' and '{field_b}' have a {strength} {direction} relationship "
            f"(Pearson r={pearson_r:.2f})."
        ),
    }


def compute_all_correlations(df: pd.DataFrame) -> list[dict]:
    """
    Automatically computes correlation for every pair of numeric fields
    actually present in the dataset -- correlation between a fixed,
    small set of always-relevant fields doesn't carry the same multiple-
    comparisons risk hypothesis testing does, since we're not testing for
    'significance' as a pass/fail claim per pair, just reporting strength.
    """
    available = [f for f in NUMERIC_FIELDS if field_is_available(df.columns, f)]
    results = []

    for i in range(len(available)):
        for j in range(i + 1, len(available)):
            result = compute_correlation(df, available[i], available[j])
            if "error" not in result:
                results.append(result)

    return results