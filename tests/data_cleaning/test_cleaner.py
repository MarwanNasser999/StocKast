"""
Unit tests for src.data_cleaning.cleaner.

Each test builds a small DataFrame with one specific fixable problem
(or the unfixable case) and confirms clean() applies the correct action,
logs it, and returns the right resulting data.
"""

import pandas as pd
import pytest

from src.data_cleaning.cleaner import clean
from src.data_cleaning.exceptions import InsufficientDataError


def make_base_rows(n: int = 60) -> pd.DataFrame:
    """A large-enough, otherwise-clean baseline so individual tests can
    introduce exactly one problem without tripping the unfixable checks."""
    dates = pd.date_range("2024-01-01", periods=n, freq="D")
    return pd.DataFrame({
        "date": dates,
        "product_id": [f"SKU-{i:03d}" for i in range(n)],
        "quantity_sold": [10 + (i % 5) for i in range(n)],
        "unit_price": [19.99] * n,
        "unit_cost": [8.50] * n,
    })


def test_clean_drops_rows_with_unparseable_dates():
    df = make_base_rows()
    df.loc[0, "date"] = "not a real date"

    cleaned_df, report = clean(df)

    assert len(cleaned_df) == len(df) - 1
    assert report.rows_before == len(df)
    assert report.rows_after == len(df) - 1
    assert any(a.check_name == "date_parseable" for a in report.actions)


def test_clean_drops_rows_missing_required_fields():
    df = make_base_rows()
    df.loc[0, "product_id"] = None

    cleaned_df, report = clean(df)

    assert len(cleaned_df) == len(df) - 1
    assert cleaned_df["product_id"].isna().sum() == 0
    assert any(a.check_name == "required_field_missing" for a in report.actions)


def test_clean_drops_rows_with_non_numeric_quantity_sold():
    df = make_base_rows()
    df.loc[0, "quantity_sold"] = "ten"

    cleaned_df, report = clean(df)

    assert len(cleaned_df) == len(df) - 1
    assert any(a.check_name == "quantity_sold_not_numeric" for a in report.actions)


def test_clean_coerces_optional_numeric_field_without_dropping_row():
    df = make_base_rows()
    df.loc[0, "unit_cost"] = "expensive"  # non-numeric, but unit_cost is optional

    cleaned_df, report = clean(df)

    # row is KEPT, unlike the required-field case above
    assert len(cleaned_df) == len(df)
    assert pd.isna(cleaned_df.loc[0, "unit_cost"])
    assert any(a.check_name == "unit_cost_not_numeric" and a.action == "coerced_to_null"
               for a in report.actions)


def test_clean_raises_when_too_few_rows():
    df = make_base_rows(n=20)  # below the 50-row minimum, unfixable

    with pytest.raises(InsufficientDataError):
        clean(df)


def test_clean_raises_when_date_span_too_short():
    dates = pd.date_range("2024-01-01", periods=5, freq="D")
    n = 60
    df = pd.DataFrame({
        "date": list(dates) * (n // 5),
        "product_id": [f"SKU-{i:03d}" for i in range(n)],
        "quantity_sold": [10] * n,
    })  # 60 rows, but crammed into only 5 distinct days

    with pytest.raises(InsufficientDataError):
        clean(df)


def test_clean_on_already_clean_data_makes_no_changes():
    df = make_base_rows()

    cleaned_df, report = clean(df)

    assert len(cleaned_df) == len(df)
    assert report.rows_before == report.rows_after
    assert len(report.actions) == 0