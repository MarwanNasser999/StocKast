"""
Price elasticity estimation for src.price_elasticity.

Log-log regression: log(quantity_sold) ~ log(unit_price). The slope
coefficient IS the elasticity directly (a well-known property of
log-log models). Requires genuine price variation within a product's
history -- without it, elasticity simply isn't estimable, and we say
so rather than guessing.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats as scipy_stats

from src.common.canonical_schema import field_is_available

MIN_ROWS_FOR_REGRESSION = 5
MIN_PRICE_VARIATION_RATIO = 0.05  # price must vary by at least 5% (max/min - 1) to attempt this
EXTREME_COEFFICIENT_MAGNITUDE = 5.0  # real elasticities are almost always within +/-5; beyond this, flag as suspect


def _classify_elasticity(coefficient: float) -> str:
    if abs(coefficient) > EXTREME_COEFFICIENT_MAGNITUDE:
        return "suspect_extreme"  # magnitude implausible for real elasticity -- likely too little/noisy data
    if coefficient > 0:
        return "unexpected_positive"  # demand rises with price -- flag for review, don't trust blindly
    magnitude = abs(coefficient)
    if magnitude < 1:
        return "inelastic"   # demand barely reacts -- raising price likely increases revenue
    else:
        return "elastic"     # demand reacts strongly -- raising price likely decreases revenue


def compute_price_elasticity(df: pd.DataFrame) -> pd.DataFrame | None:
    """
    Returns a DataFrame with one row per product_id that had enough
    price variation to estimate elasticity: elasticity_coefficient,
    r_squared, p_value, classification, n_observations.
    Returns None entirely if unit_price isn't available in this dataset.
    """
    if not field_is_available(df.columns, "unit_price"):
        return None

    results = []

    for product_id, group in df.groupby("product_id"):
        valid = group[(group["quantity_sold"] > 0) & (group["unit_price"] > 0)]

        if len(valid) < MIN_ROWS_FOR_REGRESSION:
            continue

        price_min, price_max = valid["unit_price"].min(), valid["unit_price"].max()
        if price_min <= 0 or (price_max / price_min - 1) < MIN_PRICE_VARIATION_RATIO:
            continue  # not enough real price variation to estimate anything meaningful

        log_price = np.log(valid["unit_price"])
        log_quantity = np.log(valid["quantity_sold"])

        slope, intercept, r_value, p_value, std_err = scipy_stats.linregress(log_price, log_quantity)

        results.append({
            "product_id": product_id,
            "elasticity_coefficient": float(slope),
            "r_squared": float(r_value ** 2),
            "p_value": float(p_value),
            "classification": _classify_elasticity(slope),
            "n_observations": len(valid),
        })

    if not results:
        return None

    return pd.DataFrame(results)