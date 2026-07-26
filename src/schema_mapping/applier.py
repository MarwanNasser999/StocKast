"""
Applier for schema_mapping.

matcher.py suggests a mapping. This file is where a CONFIRMED mapping
actually gets applied to a real DataFrame -- renaming raw columns into
canonical ones, filling safe defaults (e.g. warehouse_id -> "main"), and
producing a DataFrame that only ever contains canonical fields.

Unmapped raw columns (e.g. "Country") are never included in the output
DataFrame -- the canonical DataFrame's contract must stay airtight: every
column in it means something specific. Unmapped columns remain recorded
in MappingResult.unmapped_columns for future reference, but they don't
ride along inside the canonical data itself.
"""

from __future__ import annotations

import pandas as pd

from src.common.canonical_schema import ALL_FIELDS, REQUIRED_FIELDS, get_field
from src.schema_mapping.exceptions import MappingNotConfirmedError, MissingRequiredFieldsError
from src.schema_mapping.result import MappingResult


def apply_mapping(raw_df: pd.DataFrame, mapping_result: MappingResult) -> pd.DataFrame:
    """
    Apply a CONFIRMED mapping to a raw DataFrame, producing a canonical
    DataFrame containing only canonical field names.

    Raises:
        MappingNotConfirmedError: mapping_result.is_confirmed is False.
        MissingRequiredFieldsError: a required canonical field has no
            mapping, even after confirmation.
    """
    if not mapping_result.is_confirmed:
        raise MappingNotConfirmedError()

    missing_required = mapping_result.missing_required_fields(REQUIRED_FIELDS)
    if missing_required:
        raise MissingRequiredFieldsError(missing_required)

    canonical_df = pd.DataFrame()

    for canonical_field in ALL_FIELDS:
        field_spec = get_field(canonical_field)

        if canonical_field in mapping_result.mapping:
            raw_column = mapping_result.mapping[canonical_field]
            canonical_df[canonical_field] = raw_df[raw_column]
        elif field_spec.default_if_missing is not None:
            # e.g. warehouse_id -> "main" for every row
            canonical_df[canonical_field] = field_spec.default_if_missing
        else:
            # optional field with no mapping and no safe default (e.g. unit_cost)
            # -- left as missing, never fabricated
            canonical_df[canonical_field] = None

    return canonical_df