"""
Orchestrator for src.recommendation_engine.

Calls kpis, forecasting, inventory_ml, and price_elasticity, then passes
their results into the recommendation rules. inventory_ml always
returns a standardized risk_table (product_id, risk_label, risk_score,
method) regardless of whether it used real ML or the rule-based
fallback internally -- this module has zero knowledge of which.
"""

from __future__ import annotations

import pandas as pd

from src.common.canonical_schema import field_is_available
from src.forecasting.overview import run_forecasting
from src.inventory_ml.overview import run_inventory_ml
from src.kpis.overview import run_kpis
from src.price_elasticity.overview import run_price_elasticity
from src.recommendation_engine.rule_based_risk import compute_rule_based_risk
from src.recommendation_engine.rules import (
    recommend_reorder, recommend_reduce_inventory, recommend_discount,
    recommend_increase_safety_stock, recommend_price_change,
)

PRIORITY_ORDER = {"high": 0, "medium": 1, "low": 2}


def get_risk_assessment(df: pd.DataFrame, kpi_result, forecast_results: dict) -> pd.DataFrame:
    """Real ML risk table if available, otherwise the rule-based
    fallback table -- same shape either way, caller never needs to know which."""
    ml_result = run_inventory_ml(df)

    if ml_result["path"] == "ml_classifier":
        return ml_result["risk_table"]

    return compute_rule_based_risk(
        doi_table=kpi_result.days_of_inventory,
        xyz_table=kpi_result.xyz_analysis,
        forecast_results=forecast_results,
    )


def _get_display_name(df: pd.DataFrame, product_id: str) -> str:
    if not field_is_available(df.columns, "product_name"):
        return product_id
    names = df.loc[df["product_id"] == product_id, "product_name"].dropna()
    return names.iloc[0] if not names.empty else product_id


def _extract_risk_row(risk_table: pd.DataFrame, product_id: str) -> dict | None:
    if risk_table is None or risk_table.empty:
        return None
    rows = risk_table[risk_table["product_id"] == product_id]
    return rows.iloc[0].to_dict() if not rows.empty else None


def run_recommendations(df: pd.DataFrame) -> list[dict]:
    """Runs the full pipeline and returns a priority-sorted list of
    concrete recommendations across all products."""
    kpi_result = run_kpis(df)
    forecast_results = run_forecasting(df)
    elasticity_result = run_price_elasticity(df)
    risk_table = get_risk_assessment(df, kpi_result, forecast_results)

    recommendations = []

    doi_table = kpi_result.days_of_inventory
    reorder_table = kpi_result.reorder_point
    dead_stock_table = kpi_result.dead_and_slow_stock
    xyz_table = kpi_result.xyz_analysis
    elasticity_table = elasticity_result.elasticity_table

    for product_id in df["product_id"].unique():
        display_name = _get_display_name(df, product_id)

        risk_row = _extract_risk_row(risk_table, product_id)
        xyz_row = xyz_table[xyz_table["product_id"] == product_id] if xyz_table is not None else pd.DataFrame()
        xyz_tier = xyz_row.iloc[0]["xyz_tier"] if not xyz_row.empty else "Y"

        if risk_row is not None and doi_table is not None:
            doi_row_df = doi_table[doi_table["product_id"] == product_id]
            if not doi_row_df.empty:
                doi_row = doi_row_df.iloc[0]

                reorder_row = None
                if reorder_table is not None:
                    reorder_row_df = reorder_table[reorder_table["product_id"] == product_id]
                    if not reorder_row_df.empty:
                        reorder_row = reorder_row_df.iloc[0]

                rec = recommend_reorder(risk_row, doi_row, reorder_row, display_name)
                if rec:
                    recommendations.append(rec)

                rec = recommend_reduce_inventory(doi_row, xyz_tier, display_name)
                if rec:
                    recommendations.append(rec)

            rec = recommend_increase_safety_stock(risk_row, xyz_tier, display_name)
            if rec:
                recommendations.append(rec)

        if dead_stock_table is not None:
            dead_row_df = dead_stock_table[dead_stock_table["product_id"] == product_id]
            if not dead_row_df.empty:
                rec = recommend_discount(dead_row_df.iloc[0], display_name)
                if rec:
                    recommendations.append(rec)

        if elasticity_table is not None:
            elasticity_row_df = elasticity_table[elasticity_table["product_id"] == product_id]
            if not elasticity_row_df.empty:
                rec = recommend_price_change(elasticity_row_df.iloc[0], display_name)
                if rec:
                    recommendations.append(rec)

    recommendations.sort(key=lambda r: PRIORITY_ORDER.get(r["priority"], 3))
    return recommendations