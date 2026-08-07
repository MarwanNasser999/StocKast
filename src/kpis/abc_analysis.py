"""
ABC Analysis for src.kpis.

Classifies products into tiers A/B/C based on cumulative contribution
to a ranking metric -- revenue when unit_price is available (the
standard basis), falling back to total quantity sold when it isn't,
clearly labeled either way.
"""

from __future__ import annotations

import pandas as pd

from src.common.canonical_schema import field_is_available

TIER_A_CUTOFF = 0.80
TIER_B_CUTOFF = 0.95  # A + B combined = 95%; remainder is C


def _assign_tier(cumulative_pct: float) -> str:
    if cumulative_pct <= TIER_A_CUTOFF:
        return "A"
    elif cumulative_pct <= TIER_B_CUTOFF:
        return "B"
    else:
        return "C"


def compute_abc_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per product_id, ranked by the basis
    metric (revenue or quantity), its cumulative percentage of the
    total, its ABC tier, and which basis was actually used.
    """
    use_revenue = field_is_available(df, "unit_price")

    if use_revenue:
        df = df.copy()
        df["revenue"] = df["quantity_sold"] * df["unit_price"]
        grouped = df.groupby("product_id")["revenue"].sum().reset_index()
        grouped.columns = ["product_id", "value"]
        basis = "revenue"
    else:
        grouped = df.groupby("product_id")["quantity_sold"].sum().reset_index()
        grouped.columns = ["product_id", "value"]
        basis = "quantity_sold"

    grouped = grouped.sort_values("value", ascending=False).reset_index(drop=True)

    total_value = grouped["value"].sum()
    grouped["cumulative_value"] = grouped["value"].cumsum()
    grouped["cumulative_pct"] = grouped["cumulative_value"] / total_value

    grouped["tier"] = grouped["cumulative_pct"].apply(_assign_tier)
    grouped["basis"] = basis

    return grouped[["product_id", "value", "cumulative_pct", "tier", "basis"]]