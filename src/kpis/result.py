"""
KPIResult -- structured output for Phase 7a's inventory KPIs.

Each field holds a per-product DataFrame (or None if that KPI couldn't
be computed for this dataset, e.g. missing current_stock). Following
the same "absent means not applicable" principle used throughout.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


class KPIResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    turnover: Optional[pd.DataFrame] = None
    days_of_inventory: Optional[pd.DataFrame] = None
    abc_analysis: Optional[pd.DataFrame] = None
    xyz_analysis: Optional[pd.DataFrame] = None