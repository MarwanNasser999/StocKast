"""
Canonical Schema
=================

The one data contract every module downstream of schema_mapping depends
on. Users upload data with wildly different column names -- schema_mapping
translates a raw DataFrame into this canonical shape exactly once, and
everything after that point (validation, cleaning, EDA, KPIs, forecasting,
ML, recommendations) only ever talks to these field names and types.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, field_validator


class FieldRequirement(str, Enum):
    REQUIRED = "required"
    OPTIONAL = "optional"


class CanonicalField(BaseModel):
    name: str
    dtype: str  # "datetime" | "string" | "float"
    requirement: FieldRequirement
    description: str
    default_if_missing: Optional[object] = None


CANONICAL_FIELDS: list[CanonicalField] = [
    CanonicalField(name="date", dtype="datetime", requirement=FieldRequirement.REQUIRED,
                   description="Transaction / order date."),
    CanonicalField(name="product_id", dtype="string", requirement=FieldRequirement.REQUIRED,
                   description="Unique product identifier."),
    CanonicalField(name="quantity_sold", dtype="float", requirement=FieldRequirement.REQUIRED,
                   description="Units sold on that date."),
    CanonicalField(name="product_name", dtype="string", requirement=FieldRequirement.OPTIONAL,
                   description="Human-readable product name."),
    CanonicalField(name="category", dtype="string", requirement=FieldRequirement.OPTIONAL,
                   description="Product category / family."),
    CanonicalField(name="unit_price", dtype="float", requirement=FieldRequirement.OPTIONAL,
                   description="Selling price per unit."),
    CanonicalField(name="unit_cost", dtype="float", requirement=FieldRequirement.OPTIONAL,
                   description="Cost price per unit. Never defaulted -- a fabricated "
                               "cost would silently corrupt margin-based KPIs."),
    CanonicalField(name="warehouse_id", dtype="string", requirement=FieldRequirement.OPTIONAL,
                   description="Warehouse / location identifier.", default_if_missing="main"),
    CanonicalField(name="current_stock", dtype="float", requirement=FieldRequirement.OPTIONAL,
                   description="On-hand inventory level."),
    CanonicalField(
    name="lead_time_days", dtype="float", requirement=FieldRequirement.OPTIONAL,
    description="Days between placing a reorder and receiving stock. "
                "Used for safety stock and reorder point calculations. "
                "If absent, the user must supply an assumed lead time "
                "at calculation time -- never fabricated or defaulted.",
),
]

REQUIRED_FIELDS: list[str] = [f.name for f in CANONICAL_FIELDS if f.requirement == FieldRequirement.REQUIRED]
OPTIONAL_FIELDS: list[str] = [f.name for f in CANONICAL_FIELDS if f.requirement == FieldRequirement.OPTIONAL]
ALL_FIELDS: list[str] = REQUIRED_FIELDS + OPTIONAL_FIELDS

_FIELD_LOOKUP: dict[str, CanonicalField] = {f.name: f for f in CANONICAL_FIELDS}


def get_field(name: str) -> CanonicalField:
    if name not in _FIELD_LOOKUP:
        raise KeyError(f"'{name}' is not a canonical field. Valid fields: {ALL_FIELDS}")
    return _FIELD_LOOKUP[name]


class CanonicalRecord(BaseModel):
    date: datetime
    product_id: str
    quantity_sold: float

    product_name: Optional[str] = None
    category: Optional[str] = None
    unit_price: Optional[float] = None
    unit_cost: Optional[float] = None
    warehouse_id: Optional[str] = "main"
    current_stock: Optional[float] = None

    @field_validator("quantity_sold")
    @classmethod
    def quantity_sold_not_negative(cls, v: float) -> float:
        if v < 0:
            raise ValueError("quantity_sold cannot be negative.")
        return v

    @field_validator("unit_price", "unit_cost", "current_stock")
    @classmethod
    def numeric_optional_not_negative(cls, v: Optional[float]) -> Optional[float]:
        if v is not None and v < 0:
            raise ValueError("value cannot be negative.")
        return v

    def available_optional_fields(self) -> list[str]:
        return [name for name in OPTIONAL_FIELDS if getattr(self, name, None) is not None]


def field_is_available(df_columns, field_name: str) -> bool:
    return field_name in df_columns