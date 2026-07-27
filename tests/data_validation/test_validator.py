"""
Unit tests for src.data_validation.validator.

Each test builds a small DataFrame with one specific problem (or none)
and confirms validate() correctly flags it with the right severity --
or doesn't flag it, for the "clean data" baseline.
"""

import pandas as pd
import pytest

from src.data_validation.validator import validate


def make_clean_df(n_rows: int = 60, days_span: int = 60) -> pd.DataFrame:
    """A baseline dataset that should pass validation with no errors and
    no warnings -- every other test tweaks one thing away from this."""
    dates = pd.date_range("2024-01-01", periods=n_rows, freq="D")[:n_rows]
    if days_span < n_rows:
        dates = pd.date_range("2024-01-01", periods=days_span, freq="D")
        dates = dates.repeat(n_rows // days_span + 1)[:n_rows]

    return pd.DataFrame({
        "date": dates,
        "product_id": [f"SKU-{i%10:03d}" for i in range(n_rows)],
        "quantity_sold": [10 + (i % 5) for i in range(n_rows)],
        "unit_price": [19.99] * n_rows,
        "unit_cost": [8.50] * n_rows,
        "warehouse_id": ["main"] * n_rows,
        "current_stock": [100] * n_rows,
        "category": ["Hardware"] * n_rows,
    })


def test_clean_data_has_no_errors_or_warnings():
    df = make_clean_df()
    report = validate(df)

    assert report.is_valid is True
    assert len(report.errors) == 0
    assert len(report.warnings) == 0


def test_too_few_rows_is_error():
    df = make_clean_df(n_rows=20, days_span=20)
    report = validate(df)

    assert report.is_valid is False
    assert any(i.check_name == "minimum_row_count" for i in report.errors)


def test_date_range_too_short_is_error():
    df = make_clean_df(n_rows=60, days_span=5)  # 60 rows crammed into 5 days
    report = validate(df)

    assert any(i.check_name == "minimum_date_span" for i in report.errors)


def test_unparseable_dates_is_error():
    df = make_clean_df()
    df.loc[: len(df) // 2, "date"] = "not a real date"  # >50% broken
    report = validate(df)

    assert any(i.check_name == "date_parseable" for i in report.errors)


def test_majority_missing_required_field_is_error():
    df = make_clean_df()
    df.loc[: len(df) // 2, "product_id"] = None  # >50% missing
    report = validate(df)

    assert any(i.check_name == "product_id_missing" and i.severity == "error" for i in report.errors)


def test_minority_missing_required_field_is_warning_not_error():
    df = make_clean_df()
    df.loc[0:2, "product_id"] = None  # a few missing, well under 50%
    report = validate(df)

    assert any(i.check_name == "product_id_missing" for i in report.warnings)
    assert not any(i.check_name == "product_id_missing" for i in report.errors)


def test_non_numeric_value_in_numeric_field_is_error():
    df = make_clean_df()
    df.loc[0, "quantity_sold"] = "ten"  # text in a numeric column
    report = validate(df)

    assert any(i.check_name == "quantity_sold_not_numeric" for i in report.errors)


def test_future_date_is_warning():
    df = make_clean_df()
    df.loc[0, "date"] = pd.Timestamp.now() + pd.Timedelta(days=30)
    report = validate(df)

    assert any(i.check_name == "dates_in_future" for i in report.warnings)


def test_absurdly_old_date_is_warning():
    df = make_clean_df()
    df.loc[0, "date"] = pd.Timestamp("1980-01-01")
    report = validate(df)

    assert any(i.check_name == "dates_absurdly_old" for i in report.warnings)


def test_negative_quantity_sold_is_warning():
    df = make_clean_df()
    df.loc[0, "quantity_sold"] = -5
    report = validate(df)

    assert any(i.check_name == "negative_quantity_sold" for i in report.warnings)


def test_negative_price_is_warning():
    df = make_clean_df()
    df.loc[0, "unit_price"] = -1.0
    report = validate(df)

    assert any(i.check_name == "unit_price_negative" for i in report.warnings)


def test_numeric_outlier_is_warning():
    df = make_clean_df()
    df.loc[0, "quantity_sold"] = 100000  # way outside the rest of the data
    report = validate(df)

    assert any(i.check_name == "quantity_sold_outliers" for i in report.warnings)


def test_rare_categorical_value_is_warning():
    df = make_clean_df(n_rows=150, days_span=60)
    df.loc[0, "category"] = "VeryRareCategory"  # 1 out of 150 = 0.67%, correctly rare
    report = validate(df)

    assert any(i.check_name == "category_rare_values" for i in report.warnings)