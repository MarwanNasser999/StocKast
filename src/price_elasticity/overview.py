"""Orchestrator for src.price_elasticity."""

from __future__ import annotations

import pandas as pd

from src.price_elasticity.elasticity import compute_price_elasticity
from src.price_elasticity.result import PriceElasticityResult


def run_price_elasticity(df: pd.DataFrame) -> PriceElasticityResult:
    """Run price elasticity estimation across all products in the dataset."""
    table = compute_price_elasticity(df)

    if table is None:
        return PriceElasticityResult(
            unavailable_reason="Price elasticity could not be estimated -- either "
                                "unit_price is not available in this dataset, or no "
                                "product had enough price variation to analyze."
        )

    return PriceElasticityResult(elasticity_table=table)