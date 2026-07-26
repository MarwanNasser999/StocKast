"""
Unit tests for src.schema_mapping.applier.

Proves: a confirmed mapping actually renames columns correctly, an
unconfirmed mapping is blocked, missing required fields is blocked, and
the warehouse_id default actually fills in when not mapped.
"""

import pandas as pd
import pytest

from src.schema_mapping.applier import apply_mapping
from src.schema_mapping.exceptions import MappingNotConfirmedError, MissingRequiredFieldsError
from src.schema_mapping.result import MappingResult


def make_raw_df():
    return pd.DataFrame({
        "StockCode": ["SKU-001", "SKU-002"],
        "Quantity": [10, 5],
        "InvoiceDate": ["2024-01-01", "2024-01-02"],
    })


def test_apply_mapping_renames_columns_correctly():
    raw_df = make_raw_df()
    mapping_result = MappingResult(
        mapping={"product_id": "StockCode", "quantity_sold": "Quantity", "date": "InvoiceDate"},
        is_confirmed=True,
    )

    canonical_df = apply_mapping(raw_df, mapping_result)

    assert list(canonical_df["product_id"]) == ["SKU-001", "SKU-002"]
    assert list(canonical_df["quantity_sold"]) == [10, 5]


def test_apply_mapping_fills_warehouse_default_when_unmapped():
    raw_df = make_raw_df()
    mapping_result = MappingResult(
        mapping={"product_id": "StockCode", "quantity_sold": "Quantity", "date": "InvoiceDate"},
        is_confirmed=True,
    )

    canonical_df = apply_mapping(raw_df, mapping_result)

    assert all(canonical_df["warehouse_id"] == "main")


def test_apply_mapping_leaves_unit_cost_missing_when_unmapped():
    raw_df = make_raw_df()
    mapping_result = MappingResult(
        mapping={"product_id": "StockCode", "quantity_sold": "Quantity", "date": "InvoiceDate"},
        is_confirmed=True,
    )

    canonical_df = apply_mapping(raw_df, mapping_result)

    assert canonical_df["unit_cost"].isna().all()


def test_apply_mapping_blocks_unconfirmed_mapping():
    raw_df = make_raw_df()
    mapping_result = MappingResult(
        mapping={"product_id": "StockCode", "quantity_sold": "Quantity", "date": "InvoiceDate"},
        is_confirmed=False,  # never approved
    )

    with pytest.raises(MappingNotConfirmedError):
        apply_mapping(raw_df, mapping_result)


def test_apply_mapping_blocks_missing_required_fields():
    raw_df = make_raw_df()
    mapping_result = MappingResult(
        mapping={"product_id": "StockCode"},  # quantity_sold and date missing
        is_confirmed=True,
    )

    with pytest.raises(MissingRequiredFieldsError) as exc_info:
        apply_mapping(raw_df, mapping_result)

    assert "quantity_sold" in exc_info.value.missing_fields
    assert "date" in exc_info.value.missing_fields


def test_unmapped_raw_columns_never_appear_in_canonical_output():
    raw_df = make_raw_df()
    raw_df["Country"] = ["UK", "France"]  # never mapped to anything
    mapping_result = MappingResult(
        mapping={"product_id": "StockCode", "quantity_sold": "Quantity", "date": "InvoiceDate"},
        is_confirmed=True,
    )

    canonical_df = apply_mapping(raw_df, mapping_result)

    assert "Country" not in canonical_df.columns