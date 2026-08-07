"""
Orchestrator for src.analytics.

run_analytics() is the single public entry point: runs descriptive
stats and correlation/distribution tests fully automatically (small,
fixed field sets -- no multiple-comparisons risk), and runs a small
default set of hypothesis test comparisons automatically too (Option C).
Additional user-driven hypothesis comparisons happen later, in Streamlit
(Phase 13), by calling hypothesis.tests.compare_two_groups() directly.
"""

from __future__ import annotations

import pandas as pd

from src.analytics import charts
from src.analytics.correlation.analysis import compute_all_correlations
from src.analytics.descriptive.stats import compute_descriptive_stats, NUMERIC_FIELDS
from src.analytics.distributions.tests import test_all_distributions
from src.analytics.hypothesis.tests import compare_top_two_groups
from src.analytics.result import AnalyticsResult
from src.common.canonical_schema import field_is_available

DEFAULT_HYPOTHESIS_GROUP_FIELDS = ["category", "warehouse_id"]


def run_analytics(df: pd.DataFrame) -> AnalyticsResult:
    """Run all automatic analytics against a canonical DataFrame."""
    result = AnalyticsResult()

    # 1. descriptive stats -- automatic, all available numeric fields
    result.descriptive_stats = compute_descriptive_stats(df)

    # 2. correlations -- automatic, all pairs (small, fixed field set)
    result.correlations = compute_all_correlations(df)

    # 3. hypothesis tests -- Option C: small, fixed default set, automatic
    for group_field in DEFAULT_HYPOTHESIS_GROUP_FIELDS:
        if field_is_available(df, group_field):
            test_result = compare_top_two_groups(df, "quantity_sold", group_field)
            if "error" not in test_result:
                result.hypothesis_tests.append(test_result)

    # 4. distribution tests -- automatic, all available numeric fields
    result.distribution_tests = test_all_distributions(df)

    # 5. supporting charts
    available_numeric = [f for f in NUMERIC_FIELDS if field_is_available(df, f)]
    heatmap = charts.build_correlation_heatmap(df, available_numeric)
    if heatmap is not None:
        result.figures["correlation_heatmap"] = heatmap

    for field in available_numeric:
        result.figures[f"distribution_histogram_{field}"] = charts.build_distribution_histogram(df, field)

    return result