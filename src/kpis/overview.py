"""Orchestrator for src.kpis Phase 7a."""

from __future__ import annotations

import pandas as pd

from src.kpis.abc_analysis import compute_abc_analysis
from src.kpis.days_of_inventory import compute_days_of_inventory
from src.kpis.result import KPIResult
from src.kpis.turnover import compute_turnover
from src.kpis.xyz_analysis import compute_xyz_analysis


def run_kpis(df: pd.DataFrame) -> KPIResult:
    """Run all Phase 7a KPIs against a canonical DataFrame."""
    return KPIResult(
        turnover=compute_turnover(df),
        days_of_inventory=compute_days_of_inventory(df),
        abc_analysis=compute_abc_analysis(df),
        xyz_analysis=compute_xyz_analysis(df),
    )