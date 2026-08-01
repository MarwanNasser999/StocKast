"""
XYZ Analysis for src.kpis.

Classifies products by demand predictability using the coefficient of
variation (CV = std / mean) of daily demand. Low CV ("X") = stable,
predictable demand. High CV ("Z") = erratic, hard-to-forecast demand.
"""

from __future__ import annotations

import pandas as pd

X_CUTOFF = 0.5   # CV below this -> stable
Y_CUTOFF = 1.0   # CV below this -> moderate; above -> erratic


def _assign_xyz_tier(cv: float) -> str:
    if cv < X_CUTOFF:
        return "X"
    elif cv < Y_CUTOFF:
        return "Y"
    else:
        return "Z"


def compute_xyz_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per product_id: mean_daily_demand,
    std_daily_demand, coefficient_of_variation, and xyz_tier.
    """
    daily = df.groupby(["product_id", df["date"].dt.date])["quantity_sold"].sum().reset_index()

    stats = daily.groupby("product_id")["quantity_sold"].agg(["mean", "std"]).reset_index()
    stats.columns = ["product_id", "mean_daily_demand", "std_daily_demand"]
    stats["std_daily_demand"] = stats["std_daily_demand"].fillna(0)

    stats = stats[stats["mean_daily_demand"] > 0].copy()
    stats["coefficient_of_variation"] = stats["std_daily_demand"] / stats["mean_daily_demand"]

    stats["xyz_tier"] = stats["coefficient_of_variation"].apply(_assign_xyz_tier)

    return stats