"""
Shared supporting visuals for src.analytics.

Most of analytics is numeric/textual (p-values, coefficients), but two
results are genuinely easier to interpret visually: a correlation
heatmap (many pairs at once) and a distribution histogram with a normal
curve overlay (to visually judge the Shapiro-Wilk result).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from scipy import stats as scipy_stats


def build_correlation_heatmap(df: pd.DataFrame, numeric_fields: list[str]) -> go.Figure | None:
    """Heatmap of Pearson correlation between all pairs of numeric_fields
    actually present in df. Returns None if fewer than 2 fields available."""
    available = [f for f in numeric_fields if f in df.columns]
    if len(available) < 2:
        return None

    corr_matrix = df[available].corr(method="pearson")
    fig = px.imshow(
        corr_matrix, text_auto=".2f", color_continuous_scale="RdBu_r",
        zmin=-1, zmax=1, title="Correlation heatmap",
    )
    return fig


def build_distribution_histogram(df: pd.DataFrame, field: str) -> go.Figure:
    """Histogram of `field` with a fitted normal curve overlaid, so the
    Shapiro-Wilk result can be judged visually alongside the number."""
    series = df[field].dropna()

    fig = px.histogram(series, x=field, nbins=40, histnorm="probability density",
                        title=f"Distribution of {field} (with normal curve overlay)")

    x_range = np.linspace(series.min(), series.max(), 200)
    normal_curve = scipy_stats.norm.pdf(x_range, series.mean(), series.std())

    fig.add_trace(go.Scatter(x=x_range, y=normal_curve, mode="lines",
                              name="Normal curve (fitted)", line=dict(color="red")))
    return fig