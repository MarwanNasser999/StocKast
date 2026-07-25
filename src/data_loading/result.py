"""
LoadResult — the object src.data_loading returns from a successful load.

Bundles the raw DataFrame with metadata so downstream modules (schema_mapping,
the Streamlit UI, logging) don't need to separately track filename, format,
encoding, or row/column counts alongside the DataFrame itself.

Note: this DataFrame is RAW. No column renaming, no type coercion, no
business logic has touched it yet — that's schema_mapping's job (Phase 2).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field


class LoadResult(BaseModel):
    # pydantic doesn't know how to validate a pandas DataFrame out of the
    # box — this tells it to accept the type as-is rather than erroring.
    model_config = ConfigDict(arbitrary_types_allowed=True)

    dataframe: pd.DataFrame
    filename: str
    file_format: str  # "csv" | "excel" | "json"
    detected_encoding: Optional[str] = None  # only meaningful for csv/json
    row_count: int = Field(ge=0)
    column_count: int = Field(ge=0)
    loaded_at: datetime

    def summary(self) -> str:
        """Short human-readable description, useful for logging and for
        the Streamlit UI to show right after a successful upload."""
        encoding_part = f", encoding={self.detected_encoding}" if self.detected_encoding else ""
        return (
            f"Loaded '{self.filename}' ({self.file_format}{encoding_part}): "
            f"{self.row_count} rows x {self.column_count} columns"
        )