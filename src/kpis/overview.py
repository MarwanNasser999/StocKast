"""Orchestrator for src.kpis (Phase 7a + 7b)."""

from __future__ import annotations

import pandas as pd

from src.kpis.abc_analysis import compute_abc_analysis
from src.kpis.days_of_inventory import compute_days_of_inventory
from src.kpis.dead_stock import compute_dead_and_slow_stock
from src.kpis.reorder_point import compute_reorder_point
from src.kpis.result import KPIResult
from src.kpis.safety_stock import compute_safety_stock
from src.kpis.seasonality import detect_seasonality
from src.kpis.turnover import compute_turnover
from src.kpis.xyz_analysis import compute_xyz_analysis


def run_kpis(df: pd.DataFrame, service_level_z: float = 1.65,
             default_lead_time_days: float | None = None) -> KPIResult:
    """Run all KPIs (Phase 7a + 7b) against a canonical DataFrame.

    service_level_z and default_lead_time_days are passed through to
    safety_stock/reorder_point -- default_lead_time_days should come
    from the dataset if lead_time_days is available, otherwise from a
    user-provided value in the Streamlit UI (Phase 13).
    """
    return KPIResult(
        turnover=compute_turnover(df),
        days_of_inventory=compute_days_of_inventory(df),
        abc_analysis=compute_abc_analysis(df),
        xyz_analysis=compute_xyz_analysis(df),
        safety_stock=compute_safety_stock(df, service_level_z, default_lead_time_days),
        reorder_point=compute_reorder_point(df, service_level_z, default_lead_time_days),
        dead_and_slow_stock=compute_dead_and_slow_stock(df),
        seasonality=detect_seasonality(df),
    )