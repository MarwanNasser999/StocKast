"""
Unit tests for src.data_loading.loaders.

These tests don't test pandas or charset-normalizer themselves — those
are already tested by their maintainers. They test OUR decisions: does
load_file() route each format correctly, raise the right custom
exceptions, and produce accurate LoadResult metadata.
"""

from pathlib import Path

import pytest

from src.data_loading.exceptions import FileLoadError, UnsupportedFileFormatError
from src.data_loading.loaders import load_file
from src.data_loading.result import LoadResult

FIXTURES_DIR = Path(__file__).parent / "fixtures"


def test_load_csv_returns_load_result():
    result = load_file(FIXTURES_DIR / "sample.csv")

    assert isinstance(result, LoadResult)
    assert result.file_format == "csv"
    assert result.row_count == 3
    assert result.column_count == 3
    assert result.detected_encoding is not None  # csv always gets an encoding guess


def test_load_excel_returns_load_result():
    result = load_file(FIXTURES_DIR / "sample.xlsx")

    assert result.file_format == "excel"
    assert result.row_count == 3
    assert result.detected_encoding is None  # excel is binary, no encoding concept


def test_load_json_returns_load_result():
    result = load_file(FIXTURES_DIR / "sample.json")

    assert result.file_format == "json"
    assert result.row_count == 3


def test_unsupported_format_raises_specific_error():
    with pytest.raises(UnsupportedFileFormatError) as exc_info:
        load_file(FIXTURES_DIR / "unsupported.txt")

    # confirm the exception carries useful, structured info, not just a message
    assert exc_info.value.filename == "unsupported.txt"
    assert "csv" in exc_info.value.supported_formats


def test_missing_file_raises_file_load_error():
    with pytest.raises(FileLoadError):
        load_file(FIXTURES_DIR / "does_not_exist.csv")


def test_dataframe_content_matches_fixture():
    """Not just row/column counts — confirm the actual values loaded correctly."""
    result = load_file(FIXTURES_DIR / "sample.csv")
    df = result.dataframe

    assert list(df.columns) == ["product_id", "quantity", "date"]
    assert df.iloc[0]["product_id"] == "SKU-001"
    assert df.iloc[0]["quantity"] == 10


def test_summary_includes_filename_and_row_count():
    result = load_file(FIXTURES_DIR / "sample.csv")
    summary = result.summary()

    assert "sample.csv" in summary
    assert "3 rows" in summary