"""
Alias lists for schema_mapping.

Pure string-similarity (fuzzy matching) alone isn't enough to map real-world
column names correctly -- "qty" and "quantity_sold" don't actually look
very similar character-by-character, even though they mean the same thing
in a business sense. This file supplies that missing business knowledge:
for each canonical field, a list of common real-world synonyms that raw
column names get fuzzy-matched against, in addition to the canonical name
itself.

This is intentionally kept separate from matcher.py: as we test schema_mapping
against more real datasets, we'll likely only ever need to EXTEND this file
(add new observed aliases), not touch the matching algorithm itself.
"""

from __future__ import annotations

# Canonical field name -> list of known real-world aliases (lowercase,
# no punctuation -- matcher.py is responsible for normalizing raw column
# names to the same form before comparing).
FIELD_ALIASES: dict[str, list[str]] = {
    "date": [
        "date", "invoicedate", "invoice_date", "order_date", "orderdate",
        "transaction_date", "transactiondate", "txn_date", "dt", "timestamp",
    ],
    "product_id": [
        "product_id", "productid", "stockcode", "stock_code", "sku",
        "item_id", "itemid", "item_code", "product_code",
    ],
    "quantity_sold": [
        "quantity_sold", "quantity", "qty", "units_sold", "unitssold",
        "units", "demand", "sales_qty", "sold_qty",
    ],
    "product_name": [
        "product_name", "productname", "description", "item_name",
        "item_description", "name", "title",
    ],
    "category": [
        "category", "product_category", "item_category", "family",
        "product_family", "type", "segment",
    ],
    "unit_price": [
        "unit_price", "unitprice", "price", "sale_price", "saleprice",
        "selling_price",
    ],
    "unit_cost": [
        "unit_cost", "unitcost", "cost", "cost_price", "costprice",
        "purchase_price",
    ],
    "warehouse_id": [
        "warehouse_id", "warehouseid", "warehouse", "location",
        "location_id", "store_id", "store", "site", "site_id",
    ],
    "current_stock": [
        "current_stock", "stock", "stock_level", "stocklevel",
        "inventory", "inventory_level", "on_hand", "onhand", "quantity_on_hand",
    ],
}


def get_aliases(canonical_field: str) -> list[str]:
    """All names (including the canonical name itself) that a raw column
    should be fuzzy-matched against for this canonical field."""
    if canonical_field not in FIELD_ALIASES:
        raise KeyError(f"'{canonical_field}' has no alias list defined.")
    # canonical name itself always counts as a candidate match
    return [canonical_field] + FIELD_ALIASES[canonical_field]