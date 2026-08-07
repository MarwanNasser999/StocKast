"""
Recommendation rules for src.recommendation_engine.

Turns risk assessment (from inventory_ml or the rule-based fallback)
plus dead-stock, warehouse, and elasticity signals into concrete,
structured suggested actions. Each recommendation carries its own
reasoning and priority, so the UI never has to guess why something
was suggested.
"""

from __future__ import annotations

import pandas as pd


def _make_recommendation(product_id: str, display_name: str, action: str,
                          reasoning: str, priority: str, source_signals: dict) -> dict:
    return {
        "product_id": product_id,
        "display_name": display_name,
        "action": action,
        "reasoning": reasoning,
        "priority": priority,
        "source_signals": source_signals,
    }


def recommend_reorder(risk_row: dict, doi_row: pd.Series, reorder_point_row: pd.Series | None,
                       display_name: str) -> dict | None:
    """High/medium risk + stock at or below reorder point -> reorder now."""
    if risk_row["risk_label"] not in {"high", "medium"}:
        return None
    if reorder_point_row is None:
        return None

    if doi_row["latest_stock"] > reorder_point_row["reorder_point"]:
        return None

    priority = "high" if risk_row["risk_label"] == "high" else "medium"
    return _make_recommendation(
        product_id=risk_row["product_id"], display_name=display_name,
        action="increase_inventory",
        reasoning=(
            f"Current stock ({doi_row['latest_stock']:.0f}) is at or below the "
            f"reorder point ({reorder_point_row['reorder_point']:.0f}), and risk is "
            f"{risk_row['risk_label']}."
        ),
        priority=priority,
        source_signals={"days_of_inventory": doi_row["days_of_inventory"],
                         "reorder_point": reorder_point_row["reorder_point"]},
    )


def recommend_reduce_inventory(doi_row: pd.Series, xyz_tier: str, display_name: str,
                                overstock_doi_threshold: float = 90) -> dict | None:
    """Very high DOI + stable (low-risk) demand -> likely overstocked."""
    if doi_row["days_of_inventory"] < overstock_doi_threshold:
        return None
    if xyz_tier == "Z":  # erratic demand -- high DOI might be legitimate buffer, not waste
        return None

    return _make_recommendation(
        product_id=doi_row["product_id"], display_name=display_name,
        action="reduce_inventory",
        reasoning=(
            f"Days of inventory ({doi_row['days_of_inventory']:.0f}) is unusually high "
            f"relative to stable, predictable demand -- likely overstocked."
        ),
        priority="medium",
        source_signals={"days_of_inventory": doi_row["days_of_inventory"], "xyz_tier": xyz_tier},
    )


def recommend_discount(dead_stock_row: pd.Series, display_name: str) -> dict | None:
    """Dead or slow-moving stock -> consider discounting."""
    if not (dead_stock_row["is_dead_stock"] or dead_stock_row["is_slow_mover"]):
        return None

    reason = "no recent sales" if dead_stock_row["is_dead_stock"] else "sales volume in the bottom 20% of all products"
    return _make_recommendation(
        product_id=dead_stock_row["product_id"], display_name=display_name,
        action="discount",
        reasoning=f"Flagged as {'dead stock' if dead_stock_row['is_dead_stock'] else 'a slow mover'} ({reason}).",
        priority="low",
        source_signals={"days_since_last_sale": dead_stock_row["days_since_last_sale"],
                         "gross_quantity_sold": dead_stock_row["gross_quantity_sold"]},
    )


def recommend_increase_safety_stock(risk_row: dict, xyz_tier: str, display_name: str) -> dict | None:
    """High volatility (Z tier) + at least medium risk -> buffer is likely too thin."""
    if xyz_tier != "Z":
        return None
    if risk_row["risk_label"] not in {"high", "medium"}:
        return None

    return _make_recommendation(
        product_id=risk_row["product_id"], display_name=display_name,
        action="increase_safety_stock",
        reasoning="Demand is highly erratic (XYZ tier Z) and risk is elevated -- current safety buffer may be insufficient.",
        priority="medium",
        source_signals={"xyz_tier": xyz_tier, "risk_label": risk_row["risk_label"]},
    )


def recommend_price_change(elasticity_row: pd.Series, display_name: str) -> dict | None:
    """Elastic demand -> a price decrease could increase revenue.
    Inelastic demand -> a price increase could increase revenue."""
    classification = elasticity_row["classification"]
    if classification not in {"elastic", "inelastic"}:
        return None  # unexpected_positive or suspect_extreme -- not trustworthy enough to recommend on

    if classification == "inelastic":
        action, direction = "consider_price_increase", "raising"
    else:
        action, direction = "consider_price_decrease", "lowering"

    return _make_recommendation(
        product_id=elasticity_row["product_id"], display_name=display_name,
        action=action,
        reasoning=(
            f"Demand is {classification} (elasticity={elasticity_row['elasticity_coefficient']:.2f}) -- "
            f"{direction} price is estimated to increase revenue. This is informational, based on "
            f"historical price variation, not a guarantee."
        ),
        priority="low",
        source_signals={"elasticity_coefficient": elasticity_row["elasticity_coefficient"],
                         "r_squared": elasticity_row["r_squared"]},
    )