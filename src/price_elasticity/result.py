"""
PriceElasticityResult -- structured output for src.price_elasticity.

elasticity_table holds per-product regression results (or None if
unit_price isn't available, or if no product had enough price variation
to estimate anything). what_if results are computed on-demand later
(Phase 13), not stored here, since they depend on a user-chosen
scenario at request time.
"""

from __future__ import annotations

from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict


class PriceElasticityResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    elasticity_table: Optional[pd.DataFrame] = None
    unavailable_reason: Optional[str] = None