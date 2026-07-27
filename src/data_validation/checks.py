"""
Individual validation checks for src.data_validation.

Each function here answers exactly one question about the canonical
DataFrame and records its finding into the ValidationReport. None of
these functions modify the DataFrame -- validation only reports, it
never cleans or transforms.
"""

from __future__ import annotations

from datetime import datetime

import pandas as pd

from src.data_validation.report import ValidationReport

MIN_ROWS = 50
MIN_DATE_SPAN_DAYS = 30
MAX_MISSING_RATIO_ERROR = 0.5   # >50% missing in a required field = ERROR
EARLIEST_SANE_YEAR = 1990
OUTLIER_IQR_MULTIPLIER = 1.5
RARE_CATEGORY_THRESHOLD = 0.01  # <1% of rows = rare


def check_date_parseable(df: pd.DataFrame, report: ValidationReport) -> None:
    """ERROR if the majority of `date` values fail to parse as real dates."""
    parsed = pd.to_datetime(df["date"], errors="coerce")
    unparseable_count = parsed.isna().sum()
    ratio = unparseable_count / len(df) if len(df) > 0 else 0

    if ratio > MAX_MISSING_RATIO_ERROR:
        report.add_error(
            "date_parseable",
            f"{unparseable_count} of {len(df)} rows have a 'date' value that "
            f"could not be parsed as a real date.",
            affected_field="date",
            affected_row_count=int(unparseable_count),
        )


def check_required_field_missing(df: pd.DataFrame, report: ValidationReport, field: str) -> None:
    """ERROR if >50% missing, WARNING if some missing but under that line."""
    missing_count = df[field].isna().sum()
    ratio = missing_count / len(df) if len(df) > 0 else 0

    if ratio > MAX_MISSING_RATIO_ERROR:
        report.add_error(
            f"{field}_missing", f"{missing_count} of {len(df)} rows are missing '{field}'.",
            affected_field=field, affected_row_count=int(missing_count),
        )
    elif missing_count > 0:
        report.add_warning(
            f"{field}_missing", f"{missing_count} of {len(df)} rows are missing '{field}'.",
            affected_field=field, affected_row_count=int(missing_count),
        )


def check_minimum_row_count(df: pd.DataFrame, report: ValidationReport) -> None:
    """ERROR if the dataset has too few rows to analyze meaningfully."""
    if len(df) < MIN_ROWS:
        report.add_error(
            "minimum_row_count",
            f"Dataset has only {len(df)} rows; at least {MIN_ROWS} are needed "
            f"for reliable analysis.",
            affected_row_count=len(df),
        )


def check_minimum_date_span(df: pd.DataFrame, report: ValidationReport) -> None:
    """ERROR if the date range is too short for trend/seasonality analysis."""
    parsed = pd.to_datetime(df["date"], errors="coerce").dropna()
    if parsed.empty:
        return  # already caught by check_date_parseable
    span_days = (parsed.max() - parsed.min()).days

    if span_days < MIN_DATE_SPAN_DAYS:
        report.add_error(
            "minimum_date_span",
            f"Data spans only {span_days} day(s); at least {MIN_DATE_SPAN_DAYS} "
            f"are needed for forecasting and seasonality detection.",
            affected_field="date",
        )


def check_numeric_field_is_numeric(df: pd.DataFrame, report: ValidationReport, field: str) -> None:
    """ERROR if a field that should be numeric contains non-numeric values."""
    if field not in df.columns:
        return
    non_numeric = pd.to_numeric(df[field], errors="coerce").isna() & df[field].notna()
    non_numeric_count = non_numeric.sum()

    if non_numeric_count > 0:
        report.add_error(
            f"{field}_not_numeric",
            f"{non_numeric_count} row(s) have a non-numeric value in '{field}'.",
            affected_field=field, affected_row_count=int(non_numeric_count),
        )


def check_dates_in_future(df: pd.DataFrame, report: ValidationReport) -> None:
    """WARNING if any date is after today -- likely a data entry error."""
    parsed = pd.to_datetime(df["date"], errors="coerce")
    future_count = (parsed > pd.Timestamp.now()).sum()

    if future_count > 0:
        report.add_warning(
            "dates_in_future", f"{future_count} row(s) have a 'date' in the future.",
            affected_field="date", affected_row_count=int(future_count),
        )


def check_dates_absurdly_old(df: pd.DataFrame, report: ValidationReport) -> None:
    """WARNING if any date is before a sane earliest bound."""
    parsed = pd.to_datetime(df["date"], errors="coerce")
    old_count = (parsed < pd.Timestamp(year=EARLIEST_SANE_YEAR, month=1, day=1)).sum()

    if old_count > 0:
        report.add_warning(
            "dates_absurdly_old",
            f"{old_count} row(s) have a 'date' before {EARLIEST_SANE_YEAR}, which is likely a data error.",
            affected_field="date", affected_row_count=int(old_count),
        )


def check_negative_quantity_sold(df: pd.DataFrame, report: ValidationReport) -> None:
    """WARNING (not ERROR) -- negative values could be legitimate returns/cancellations."""
    series = pd.to_numeric(df["quantity_sold"], errors="coerce")
    negative_count = (series < 0).sum()

    if negative_count > 0:
        report.add_warning(
            "negative_quantity_sold",
            f"{negative_count} row(s) have negative 'quantity_sold' -- "
            f"verify these represent returns/cancellations rather than data errors.",
            affected_field="quantity_sold", affected_row_count=int(negative_count),
        )


def check_negative_price_or_cost(df: pd.DataFrame, report: ValidationReport, field: str) -> None:
    """WARNING -- zero is fine (e.g. promos), negative is not."""
    if field not in df.columns:
        return
    series = pd.to_numeric(df[field], errors="coerce")
    negative_count = (series < 0).sum()

    if negative_count > 0:
        report.add_warning(
            f"{field}_negative", f"{negative_count} row(s) have a negative '{field}'.",
            affected_field=field, affected_row_count=int(negative_count),
        )


def check_numeric_outliers(df: pd.DataFrame, report: ValidationReport, field: str) -> None:
    """WARNING -- flags statistical outliers using the IQR method. Does not
    assume they're wrong (e.g. a real demand spike), just flags for review."""
    if field not in df.columns:
        return
    series = pd.to_numeric(df[field], errors="coerce").dropna()
    if series.empty:
        return

    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    lower_bound = q1 - OUTLIER_IQR_MULTIPLIER * iqr
    upper_bound = q3 + OUTLIER_IQR_MULTIPLIER * iqr

    outlier_count = ((series < lower_bound) | (series > upper_bound)).sum()

    if outlier_count > 0:
        report.add_warning(
            f"{field}_outliers",
            f"{outlier_count} row(s) have a '{field}' value outside the "
            f"expected range ({lower_bound:.2f} to {upper_bound:.2f}), based on this dataset's own distribution.",
            affected_field=field, affected_row_count=int(outlier_count),
        )


def check_rare_categorical_values(df: pd.DataFrame, report: ValidationReport, field: str) -> None:
    """WARNING -- flags category values that appear in less than 1% of rows,
    which could be typos or genuinely rare-but-valid cases."""
    if field not in df.columns:
        return
    non_null = df[field].dropna()
    if non_null.empty:
        return

    value_counts = non_null.value_counts(normalize=True)
    rare_values = value_counts[value_counts < RARE_CATEGORY_THRESHOLD]

    if not rare_values.empty:
        report.add_warning(
            f"{field}_rare_values",
            f"{len(rare_values)} value(s) in '{field}' each appear in less than "
            f"{RARE_CATEGORY_THRESHOLD*100:.0f}% of rows: {list(rare_values.index)}.",
            affected_field=field, affected_row_count=int(non_null.isin(rare_values.index).sum()),
        )