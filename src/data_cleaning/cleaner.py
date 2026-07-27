"""
Cleaner for src.data_cleaning.

Hybrid design: always attempts the three fixable operations (date
parsing, required-field completeness, numeric coercion), regardless of
whether validation classified the issue as an ERROR or a WARNING --
even a single bad row is worth fixing, no threshold needed. Only
minimum_row_count and minimum_date_span are truly unfixable, since no
amount of row-cleaning manufactures more data; those stop the pipeline
via InsufficientDataError before any fix is attempted.
"""

from __future__ import annotations

import pandas as pd

from src.data_cleaning.exceptions import InsufficientDataError
from src.data_cleaning.report import CleaningReport
from src.data_validation.validator import validate

REQUIRED_FIELDS = ["date", "product_id", "quantity_sold"]
OPTIONAL_NUMERIC_FIELDS = ["unit_price", "unit_cost", "current_stock"]

UNFIXABLE_CHECKS = {"minimum_row_count", "minimum_date_span"}


def clean(df: pd.DataFrame) -> tuple[pd.DataFrame, CleaningReport]:
    """
    Always attempts to fix parseable dates, required-field completeness,
    and numeric type issues -- regardless of error/warning severity.
    Raises InsufficientDataError if the dataset is fundamentally too
    small or too short to be usable, since that can't be fixed here.
    """
    report = CleaningReport()
    report.rows_before = len(df)
    cleaned = df.copy()

    initial_report = validate(cleaned)
    error_names = {issue.check_name for issue in initial_report.errors}

    unfixable_present = error_names & UNFIXABLE_CHECKS
    if unfixable_present:
        reasons = [issue.message for issue in initial_report.errors
                   if issue.check_name in UNFIXABLE_CHECKS]
        raise InsufficientDataError(reasons)

    # 1. date -> coerce, drop rows where date became NaT (any count, not just >50%)
    cleaned["date"] = pd.to_datetime(cleaned["date"], errors="coerce")
    before = len(cleaned)
    cleaned = cleaned[cleaned["date"].notna()]
    dropped = before - len(cleaned)
    if dropped > 0:
        report.add_action(
            "date_parseable", "dropped_rows",
            f"Dropped {dropped} row(s) with an unparseable 'date'.",
            affected_field="date", affected_row_count=dropped,
        )

    # 2. required fields missing -> drop rows missing any of them (any count)
    before = len(cleaned)
    cleaned = cleaned.dropna(subset=REQUIRED_FIELDS)
    dropped = before - len(cleaned)
    if dropped > 0:
        report.add_action(
            "required_field_missing", "dropped_rows",
            f"Dropped {dropped} row(s) missing a required field ({', '.join(REQUIRED_FIELDS)}).",
            affected_row_count=dropped,
        )

    # 3a. quantity_sold not numeric (required) -> coerce, drop rows that fail
    before = len(cleaned)
    cleaned["quantity_sold"] = pd.to_numeric(cleaned["quantity_sold"], errors="coerce")
    cleaned = cleaned[cleaned["quantity_sold"].notna()]
    dropped = before - len(cleaned)
    if dropped > 0:
        report.add_action(
            "quantity_sold_not_numeric", "dropped_rows",
            f"Dropped {dropped} row(s) with a non-numeric 'quantity_sold'.",
            affected_field="quantity_sold", affected_row_count=dropped,
        )

    # 3b. optional numeric fields not numeric -> coerce, keep as NaN (never drop, never fabricate)
    for field in OPTIONAL_NUMERIC_FIELDS:
        if field not in cleaned.columns:
            continue
        before_non_null = cleaned[field].notna().sum()
        cleaned[field] = pd.to_numeric(cleaned[field], errors="coerce")
        after_non_null = cleaned[field].notna().sum()
        coerced_to_null = before_non_null - after_non_null
        if coerced_to_null > 0:
            report.add_action(
                f"{field}_not_numeric", "coerced_to_null",
                f"Converted {coerced_to_null} non-numeric '{field}' value(s) to missing "
                f"(row kept -- {field} is optional).",
                affected_field=field, affected_row_count=int(coerced_to_null),
            )

    report.rows_after = len(cleaned)
    return cleaned.reset_index(drop=True), report