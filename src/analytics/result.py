"""
AnalyticsResult -- the structured output of src.analytics.overview.run_analytics().

Bundles descriptive stats, hypothesis test results, correlations, and
distribution tests together, plus the small set of supporting charts.
Following the same dict-based pattern as EDAResult: a missing key means
"not applicable to this dataset", never an error or placeholder.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class AnalyticsResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Plotly Figure objects

    descriptive_stats: dict[str, dict[str, float]] = Field(default_factory=dict)
    hypothesis_tests: list[dict] = Field(default_factory=list)
    correlations: list[dict] = Field(default_factory=list)
    distribution_tests: list[dict] = Field(default_factory=list)

    figures: dict[str, Any] = Field(default_factory=dict)

    def has_figure(self, name: str) -> bool:
        return name in self.figures