"""
'What-if' price change projection for src.price_elasticity.

Combines this product's own computed elasticity coefficient (from
elasticity.py) with Phase 8's demand forecast to project the effect of
a hypothetical price change. Both are fetched internally, not passed in
by the caller, to guarantee they're always the real, current values.
"""

from __future__ import annotations

import pandas as pd

from src.forecasting.overview import forecast_one_product
from src.price_elasticity.elasticity import compute_price_elasticity


def project_price_change(df: pd.DataFrame, product_id: str, price_change_pct: float) -> dict:
    """
    df: the full canonical dataset.
    product_id: which product to project for.
    price_change_pct: e.g. 10 for a 10% increase, -15 for a 15% decrease.
    """
    elasticity_table = compute_price_elasticity(df)
    if elasticity_table is None or product_id not in elasticity_table["product_id"].values:
        return {"error": f"No price elasticity estimate available for product '{product_id}'."}

    product_elasticity_row = elasticity_table[elasticity_table["product_id"] == product_id].iloc[0]
    elasticity_coefficient = float(product_elasticity_row["elasticity_coefficient"])

    product_df = df[df["product_id"] == product_id]
    current_price = float(product_df["unit_price"].iloc[-1])  # most recent known price

    forecast_result = forecast_one_product(product_df)
    if "error" in forecast_result:
        return {"error": f"Cannot project price change: {forecast_result['error']}"}

    baseline_forecast_total_units = float(forecast_result["forecast"].sum())

    price_change_ratio = price_change_pct / 100
    new_price = current_price * (1 + price_change_ratio)

    projected_demand_change_pct = elasticity_coefficient * price_change_pct
    projected_units = baseline_forecast_total_units * (1 + projected_demand_change_pct / 100)
    projected_units = max(projected_units, 0)

    baseline_revenue = baseline_forecast_total_units * current_price
    projected_revenue = projected_units * new_price
    revenue_change_pct = (
        (projected_revenue - baseline_revenue) / baseline_revenue * 100
        if baseline_revenue > 0 else 0.0
    )

    return {
        "product_id": product_id,
        "elasticity_coefficient": elasticity_coefficient,
        "current_price": current_price,
        "new_price": round(new_price, 2),
        "price_change_pct": price_change_pct,
        "baseline_units": round(baseline_forecast_total_units, 1),
        "projected_units": round(projected_units, 1),
        "projected_demand_change_pct": round(projected_demand_change_pct, 1),
        "baseline_revenue": round(baseline_revenue, 2),
        "projected_revenue": round(projected_revenue, 2),
        "revenue_change_pct": round(revenue_change_pct, 1),
    }