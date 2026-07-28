"""
Result containers for src.eda.

Both EDAResult (dataset-wide) and ProductEDAResult (single-product
drill-down) hold two things: computed stats (numbers) and Plotly figures
(visuals). Both use dictionaries rather than fixed fields, because which
stats/figures exist depends on which optional canonical fields are
present in a given dataset -- a missing key means "not applicable here",
never a placeholder or an error.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class EDAResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)  # Plotly Figure objects

    available_fields: list[str] = Field(default_factory=list)
    unavailable_fields: list[str] = Field(default_factory=list)

    stats: dict[str, Any] = Field(default_factory=dict)
    figures: dict[str, Any] = Field(default_factory=dict)  # name -> plotly.graph_objects.Figure

    def has_figure(self, name: str) -> bool:
        return name in self.figures


class ProductEDAResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    product_id: str
    display_name: str  # product_name if available, otherwise product_id

    stats: dict[str, Any] = Field(default_factory=dict)
    figures: dict[str, Any] = Field(default_factory=dict)

    def has_figure(self, name: str) -> bool:
        return name in self.figures