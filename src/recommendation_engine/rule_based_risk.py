"""
Rule-based stockout risk fallback for src.recommendation_engine.

Used ONLY when src.inventory_ml reports "unavailable" for a dataset
(insufficient current_stock history to train a real classifier). This
is an explicit, transparent FORMULA -- not machine learning -- combining
days_of_inventory, XYZ demand volatility tier, and forecast trend into
one risk label. Always labeled honestly as rule-based in its output.
"""

from __future__ import annotations

import pandas as pd

DOI_HIGH_RISK_THRESHOLD = 7    # days
DOI_MEDIUM_RISK_THRESHOLD = 14  # days


def _doi_risk_component(doi: float) -> int:
    """0 = low risk, 1 = medium, 2 = high -- based purely on DOI."""
    if doi <= DOI_HIGH_RISK_THRESHOLD:
        return 2
    elif doi <= DOI_MEDIUM_RISK_THRESHOLD:
        return 1
    return 0


def _xyz_risk_component(xyz_tier: str) -> int:
    """Erratic demand (Z) adds risk; stable demand (X) reduces it."""
    return {"Z": 2, "Y": 1, "X": 0}.get(xyz_tier, 1)


def _trend_risk_component(forecast_series: pd.Series) -> int:
    """A rising trend on top of already-low stock compounds risk."""
    if forecast_series is None or len(forecast_series) < 2:
        return 0
    return 1 if forecast_series.iloc[-1] > forecast_series.iloc[0] else 0


def compute_rule_based_risk(doi_table: pd.DataFrame, xyz_table: pd.DataFrame,
                             forecast_results: dict) -> pd.DataFrame:
    """
    Combines DOI + XYZ tier + forecast trend into one risk label per
    product. Requires doi_table and xyz_table (from src.kpis) to be
    available -- returns an empty DataFrame if either is None.
    """
    if doi_table is None or xyz_table is None:
        return pd.DataFrame()

    merged = doi_table.merge(xyz_table, on="product_id", how="inner")

    rows = []
    for _, row in merged.iterrows():
        product_id = row["product_id"]
        doi_score = _doi_risk_component(row["days_of_inventory"])
        xyz_score = _xyz_risk_component(row.get("xyz_tier", "Y"))

        forecast = forecast_results.get(product_id, {})
        forecast_series = forecast.get("forecast") if "error" not in forecast else None
        trend_score = _trend_risk_component(forecast_series)

        total_score = doi_score + xyz_score + trend_score

        if total_score >= 4:
            risk_label = "high"
        elif total_score >= 2:
            risk_label = "medium"
        else:
            risk_label = "low"

        rows.append({
            "product_id": product_id,
            "risk_label": risk_label,
            "risk_score": total_score,
            "method": "rule_based",  # honest label, never presented as ML
            "days_of_inventory": row["days_of_inventory"],
            "xyz_tier": row.get("xyz_tier", "unknown"),
        })

    return pd.DataFrame(rows)