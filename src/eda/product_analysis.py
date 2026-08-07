"""
Per-product EDA for src.eda.

Given one product_id, computes that product's own stats (demand trend,
profit if available) and builds charts scoped to just that product,
reusing the same chart-building functions from charts.py.
"""

from __future__ import annotations

import pandas as pd

from src.common.canonical_schema import field_is_available
from src.eda import charts
from src.eda.result import ProductEDAResult

NUMERIC_FIELDS_FOR_DISTRIBUTION = [
    "quantity_sold",
    "unit_price",
    "unit_cost",
    "current_stock",
]


def get_product_display_options(df: pd.DataFrame) -> dict[str, str]:
    """
    Returns {product_id: display_label} for every distinct product.

    Uses product_name when available, otherwise falls back to product_id.
    If multiple product_ids share the same product_name, appends the
    product_id in parentheses so the user can distinguish them.
    """

    if field_is_available(df, "product_name"):
        pairs = df[["product_id", "product_name"]].drop_duplicates(
            subset="product_id"
        )

        labels = pairs["product_name"].fillna(pairs["product_id"])

        # Disambiguate duplicate product names
        counts = labels.value_counts()

        display_labels = [
            f"{label} ({pid})" if counts[label] > 1 else label
            for pid, label in zip(pairs["product_id"], labels)
        ]

        return dict(zip(pairs["product_id"], display_labels))

    return {pid: pid for pid in df["product_id"].unique()}


def run_product_eda(df: pd.DataFrame, product_id: str) -> ProductEDAResult:
    """Run EDA scoped to a single product_id."""

    product_df = df[df["product_id"] == product_id]

    if product_df.empty:
        raise ValueError(f"No rows found for product_id '{product_id}'.")

    display_name = product_id

    if field_is_available(df, "product_name"):
        name_value = product_df["product_name"].dropna()

        if not name_value.empty:
            display_name = name_value.iloc[0]

    result = ProductEDAResult(
        product_id=product_id,
        display_name=display_name,
    )

    # Stats
    result.stats["total_units_sold"] = float(
        product_df["quantity_sold"].sum()
    )

    result.stats["date_range_days"] = int(
        (
            product_df["date"].max()
            - product_df["date"].min()
        ).days
    )

    if field_is_available(df, "unit_price"):
        result.stats["total_revenue"] = float(
            (
                product_df["quantity_sold"]
                * product_df["unit_price"]
            ).sum()
        )

    if (
        field_is_available(df, "unit_price")
        and field_is_available(df, "unit_cost")
    ):
        profit = (
            product_df["unit_price"]
            - product_df["unit_cost"]
        ) * product_df["quantity_sold"]

        result.stats["total_profit"] = float(profit.sum())

    # Charts
    result.figures["demand_over_time"] = (
        charts.build_demand_over_time_chart(product_df)
    )

    for field in NUMERIC_FIELDS_FOR_DISTRIBUTION:
        if field_is_available(df, field):
            result.figures[f"distribution_{field}"] = (
                charts.build_distribution_chart(
                    product_df,
                    field,
                )
            )

    return result