"""
Hypothesis testing for src.analytics.hypothesis.

Generic, reusable group comparison: is a numeric field's distribution
genuinely different between two groups defined by a categorical field?
Uses Mann-Whitney U as the primary test (doesn't assume normality --
important since sales data is often skewed), with a p-value and a
plain-language interpretation.
"""

from __future__ import annotations

import pandas as pd
from scipy import stats as scipy_stats

SIGNIFICANCE_THRESHOLD = 0.05


def compare_two_groups(df: pd.DataFrame, numeric_field: str, group_field: str,
                        group_a: str, group_b: str) -> dict:
    """
    Compare `numeric_field` between two specific values of `group_field`
    (e.g. numeric_field="quantity_sold", group_field="category",
    group_a="Hardware", group_b="Software").

    Returns a dict with the test statistic, p-value, whether the
    difference is statistically significant, and a plain-language
    interpretation.
    """
    values_a = df.loc[df[group_field] == group_a, numeric_field].dropna()
    values_b = df.loc[df[group_field] == group_b, numeric_field].dropna()

    if len(values_a) < 2 or len(values_b) < 2:
        return {
            "group_a": group_a, "group_b": group_b,
            "error": "Not enough data in one or both groups to run a statistical test.",
        }

    statistic, p_value = scipy_stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
    is_significant = bool(p_value < SIGNIFICANCE_THRESHOLD)

    mean_a, mean_b = float(values_a.mean()), float(values_b.mean())

    if is_significant:
        direction = "higher" if mean_a > mean_b else "lower"
        interpretation = (
            f"'{group_a}' has statistically significantly {direction} {numeric_field} "
            f"than '{group_b}' (p={p_value:.4f})."
        )
    else:
        interpretation = (
            f"No statistically significant difference in {numeric_field} between "
            f"'{group_a}' and '{group_b}' (p={p_value:.4f}) -- the observed difference "
            f"could plausibly be due to random variation."
        )

    return {
        "group_a": group_a, "group_b": group_b,
        "mean_a": mean_a, "mean_b": mean_b,
        "p_value": float(p_value),
        "is_significant": is_significant,
        "interpretation": interpretation,
    }


def compare_top_two_groups(df: pd.DataFrame, numeric_field: str, group_field: str) -> dict:
    """
    Convenience wrapper: automatically picks the two LARGEST groups (by
    row count) in `group_field` and compares them -- useful when the
    caller doesn't want to specify group values manually.
    """
    top_groups = df[group_field].value_counts().nlargest(2).index.tolist()

    if len(top_groups) < 2:
        return {"error": f"'{group_field}' does not have at least 2 distinct groups to compare."}

    return compare_two_groups(df, numeric_field, group_field, top_groups[0], top_groups[1])