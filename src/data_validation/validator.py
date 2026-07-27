"""
Validator for src.data_validation.

The single public entry point for this module: validate() runs every
check in checks.py against a canonical DataFrame and returns one
complete ValidationReport.

Runs ALL checks regardless of earlier failures -- unlike schema_mapping,
which stops at the first blocking problem, validation deliberately keeps
going so the user sees the FULL picture of what's wrong with their data
in one pass, not one error at a time.
"""

from __future__ import annotations

import pandas as pd

from src.data_validation import checks
from src.data_validation.report import ValidationReport

REQUIRED_FIELDS = ["date", "product_id", "quantity_sold"]
NUMERIC_FIELDS = ["quantity_sold", "unit_price", "unit_cost", "current_stock"]
PRICE_COST_FIELDS = ["unit_price", "unit_cost"]
CATEGORICAL_FIELDS = ["category", "warehouse_id"]


def validate(df: pd.DataFrame) -> ValidationReport:
    """Run all data_validation checks against a canonical DataFrame."""
    report = ValidationReport()

    # structural / core checks
    checks.check_minimum_row_count(df, report)
    checks.check_date_parseable(df, report)
    checks.check_minimum_date_span(df, report)

    for field in REQUIRED_FIELDS:
        checks.check_required_field_missing(df, report, field)

    for field in NUMERIC_FIELDS:
        checks.check_numeric_field_is_numeric(df, report, field)

    # date sanity
    checks.check_dates_in_future(df, report)
    checks.check_dates_absurdly_old(df, report)

    # value sanity
    checks.check_negative_quantity_sold(df, report)
    for field in PRICE_COST_FIELDS:
        checks.check_negative_price_or_cost(df, report, field)

    # outliers / rare values
    for field in NUMERIC_FIELDS:
        checks.check_numeric_outliers(df, report, field)
    for field in CATEGORICAL_FIELDS:
        checks.check_rare_categorical_values(df, report, field)

    return report