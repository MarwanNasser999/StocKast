"""
KPIResult -- structured output for src.kpis (Phase 7a + 7b).

Each field holds a per-product DataFrame/dict (or None if that KPI
couldn't be computed for this dataset). Following the same "absent
means not applicable" principle used throughout.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


class KPIResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    # Phase 7a
    turnover: Optional[pd.DataFrame] = None
    days_of_inventory: Optional[pd.DataFrame] = None
    abc_analysis: Optional[pd.DataFrame] = None
    xyz_analysis: Optional[pd.DataFrame] = None

    # Phase 7b
    safety_stock: Optional[pd.DataFrame] = None
    reorder_point: Optional[pd.DataFrame] = None
    dead_and_slow_stock: Optional[pd.DataFrame] = None
    seasonality: Optional[dict] = None