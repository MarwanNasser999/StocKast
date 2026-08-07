"""
Dataset-wide EDA for src.eda.

Computes summary stats and builds whichever charts are supported by the
fields actually present in this dataset. A chart/stat is simply absent
from the result if its required fields aren't available -- never an
error, never a placeholder.
"""

from __future__ import annotations

import pandas as pd

from src.common.canonical_schema import OPTIONAL_FIELDS, field_is_available
from src.eda import charts
from src.eda.result import EDAResult

NUMERIC_FIELDS_FOR_DISTRIBUTION = ["quantity_sold", "unit_price", "unit_cost", "current_stock"]


def run_eda(df: pd.DataFrame) -> EDAResult:
    """Run dataset-wide EDA: available-field summary, stats, and charts."""
    result = EDAResult()

    # 1. which optional fields exist in this dataset
    present = [f for f in OPTIONAL_FIELDS if field_is_available(df, f)]
    absent = [f for f in OPTIONAL_FIELDS if not field_is_available(df, f)]
    result.available_fields = present
    result.unavailable_fields = absent

    label_col = "product_name" if field_is_available(df, "product_name") else "product_id"

    # 2. basic stats -- always computable, since date/product_id/quantity_sold are required
    result.stats["total_units_sold"] = float(df["quantity_sold"].sum())
    result.stats["distinct_products"] = int(df["product_id"].nunique())
    result.stats["date_range_days"] = int((df["date"].max() - df["date"].min()).days)

    if field_is_available(df, "unit_price"):
        result.stats["total_revenue"] = float((df["quantity_sold"] * df["unit_price"]).sum())

    # 3. demand over time -- always possible (date + quantity_sold are required)
    result.figures["demand_over_time"] = charts.build_demand_over_time_chart(df)

    # 4. revenue by category -- needs category AND unit_price
    if field_is_available(df, "category") and field_is_available(df, "unit_price"):
        result.figures["revenue_by_category"] = charts.build_revenue_by_category_chart(df)

    # 5. top products by volume -- always possible
    result.figures["top_products"] = charts.build_top_products_chart(df, label_col=label_col)

    # 6. trendy products -- may legitimately not exist (handled inside the chart function)
    trendy_fig = charts.build_trendy_products_chart(df, label_col=label_col)
    if trendy_fig is not None:
        result.figures["trendy_products"] = trendy_fig

    # 7. distribution/outliers -- for every numeric field actually present
    for field in NUMERIC_FIELDS_FOR_DISTRIBUTION:
        if field_is_available(df, field):
            result.figures[f"distribution_{field}"] = charts.build_distribution_chart(df, field)

    return result