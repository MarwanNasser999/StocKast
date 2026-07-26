"""
Unit tests for src.schema_mapping.matcher.

These prove our design decisions actually hold: alias-based fuzzy matching
finds obvious matches, unmapped columns are preserved (never dropped),
the greedy algorithm never double-assigns, and genuinely ambiguous cases
get flagged rather than silently guessed.
"""

import pytest

from src.schema_mapping.matcher import (
    normalize_column_name,
    score_column_against_field,
    suggest_mapping,
)


def test_normalize_handles_spacing_casing_and_punctuation():
    assert normalize_column_name("Invoice Date") == "invoice_date"
    assert normalize_column_name("InvoiceDate") == "invoicedate"
    assert normalize_column_name("  Quantity  ") == "quantity"
    assert normalize_column_name("Customer-ID#") == "customer_id"


def test_score_exact_alias_match_is_perfect():
    # "quantity" is a listed alias for quantity_sold
    score = score_column_against_field("Quantity", "quantity_sold")
    assert score == 100.0


def test_score_unrelated_column_is_low():
    score = score_column_against_field("Country", "quantity_sold")
    assert score < 75.0  # below our real matching threshold


def test_suggest_mapping_matches_online_retail_columns():
    """Real-world-ish column names, similar to Online Retail II."""
    raw_columns = ["StockCode", "Quantity", "InvoiceDate", "Price", "Country"]
    result = suggest_mapping(raw_columns, threshold=75.0)

    assert result["mapping"]["product_id"] == "StockCode"
    assert result["mapping"]["quantity_sold"] == "Quantity"
    assert result["mapping"]["date"] == "InvoiceDate"
    assert result["mapping"]["unit_price"] == "Price"

    # Country has no canonical equivalent -- must be preserved, not dropped
    assert "Country" in result["unmapped_columns"]


def test_suggest_mapping_below_threshold_stays_unmapped():
    """A column with no reasonable match for anything should not be
    force-matched just because it's the 'best available' option."""
    raw_columns = ["random_column_xyz"]
    result = suggest_mapping(raw_columns, threshold=75.0)

    assert result["mapping"] == {}
    assert "random_column_xyz" in result["unmapped_columns"]


def test_suggest_mapping_never_double_assigns_a_column():
    """Even if one raw column scores well against multiple fields, it
    can only end up claimed by exactly one."""
    raw_columns = ["Quantity", "StockCode"]
    result = suggest_mapping(raw_columns, threshold=75.0)

    assigned_columns = list(result["mapping"].values())
    assert len(assigned_columns) == len(set(assigned_columns))  # no duplicates


def test_suggest_mapping_flags_genuine_ambiguity():
    """Two columns that are both exact/near-perfect matches for the same
    field's aliases must be flagged as ambiguous, not silently resolved."""
    raw_columns = ["qty", "quantity"]  # both literal aliases of quantity_sold
    result = suggest_mapping(raw_columns, threshold=75.0)

    assert "quantity_sold" in result["ambiguous"]
    candidates = [col for col, _ in result["ambiguous"]["quantity_sold"]]
    assert "qty" in candidates
    assert "quantity" in candidates


def test_suggest_mapping_empty_columns_returns_all_unmapped_state():
    result = suggest_mapping([], threshold=75.0)
    assert result["mapping"] == {}
    assert result["unmapped_columns"] == []
    assert result["ambiguous"] == {}