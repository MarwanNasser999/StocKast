"""Unit tests for src.schema_mapping.result.MappingResult helper methods."""

from src.schema_mapping.result import MappingResult


def test_missing_required_fields_reports_gaps():
    result = MappingResult(mapping={"product_id": "StockCode"})
    missing = result.missing_required_fields(["date", "product_id", "quantity_sold"])
    assert missing == ["date", "quantity_sold"]


def test_missing_required_fields_empty_when_all_present():
    result = MappingResult(mapping={"date": "InvoiceDate", "product_id": "StockCode", "quantity_sold": "Quantity"})
    missing = result.missing_required_fields(["date", "product_id", "quantity_sold"])
    assert missing == []


def test_has_unresolved_ambiguity_true_when_present():
    result = MappingResult(ambiguous={"quantity_sold": [("qty", 100.0), ("quantity", 100.0)]})
    assert result.has_unresolved_ambiguity() is True


def test_has_unresolved_ambiguity_false_when_empty():
    result = MappingResult()
    assert result.has_unresolved_ambiguity() is False