"""
Shared chart-building helpers for src.eda.

Each function takes a DataFrame (already filtered to whatever scope is
needed -- whole dataset or one product) and returns a Plotly Figure.
Used by both overview.py (dataset-wide) and product_analysis.py
(per-product), so the actual charting logic is written exactly once.
"""

from __future__ import annotations

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


def build_demand_over_time_chart(df: pd.DataFrame) -> go.Figure:
    """Line chart: total quantity_sold per day."""
    daily = df.groupby(df["date"].dt.date)["quantity_sold"].sum().reset_index()
    daily.columns = ["date", "quantity_sold"]
    fig = px.line(daily, x="date", y="quantity_sold", title="Demand over time")
    return fig


def build_revenue_by_category_chart(df: pd.DataFrame) -> go.Figure:
    """Pie chart: total revenue (quantity_sold * unit_price) by category.
    Caller is responsible for confirming 'category' and 'unit_price' exist
    before calling this."""
    df = df.copy()
    df["revenue"] = df["quantity_sold"] * df["unit_price"]
    by_category = df.groupby("category")["revenue"].sum().reset_index()
    fig = px.pie(by_category, names="category", values="revenue", title="Revenue by category")
    return fig


def build_top_products_chart(df: pd.DataFrame, label_col: str = "product_id", top_n: int = 10) -> go.Figure:
    """Horizontal bar chart: top N products by total quantity_sold.
    label_col lets the caller pass product_name instead of product_id
    when a friendlier label is available."""
    totals = df.groupby(label_col)["quantity_sold"].sum().nlargest(top_n).reset_index()
    fig = px.bar(totals, x="quantity_sold", y=label_col, orientation="h", title=f"Top {top_n} products by volume")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig


def build_distribution_chart(df: pd.DataFrame, field: str) -> go.Figure:
    """Box plot showing distribution and outliers for a numeric field."""
    fig = px.box(df, y=field, title=f"Distribution of {field}", points="outliers")
    return fig



def build_trendy_products_chart(df: pd.DataFrame, label_col: str = "product_id", top_n: int = 10) -> go.Figure | None:
    """
    Horizontal bar chart: top N products by GROWTH (% change in average
    daily demand, early third vs. recent third of the date range).
    Returns None if there isn't enough overlapping data to compute
    meaningful growth for any product.
    """
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])

    date_min, date_max = df["date"].min(), df["date"].max()
    third = (date_max - date_min) / 3

    early_period = df[df["date"] <= date_min + third]
    recent_period = df[df["date"] >= date_max - third]

    early_avg = early_period.groupby(label_col)["quantity_sold"].mean()
    recent_avg = recent_period.groupby(label_col)["quantity_sold"].mean()

    growth = ((recent_avg - early_avg) / early_avg * 100).dropna()

    if growth.empty:
        return None  # not enough products with data in BOTH periods

    top_growth = growth.nlargest(top_n).reset_index()
    top_growth.columns = [label_col, "growth_percent"]

    fig = px.bar(top_growth, x="growth_percent", y=label_col, orientation="h",
                 title=f"Top {top_n} trending products (% growth)")
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig