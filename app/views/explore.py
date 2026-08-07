"""Stockast -- EDA & Analytics page."""

from __future__ import annotations

import streamlit as st

from app.state import init_session_state, cached_eda, cached_analytics
from src.eda.product_analysis import get_product_display_options, run_product_eda
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))


init_session_state()

st.title("📊 Explore Your Data")

if st.session_state.get("cleaned_df") is None:
    st.warning("Complete validation & cleaning first.")
    st.stop()

df = st.session_state["cleaned_df"]

tab_overview, tab_product, tab_stats = st.tabs(["Overview", "By Product", "Statistical Analysis"])

# ---------- Tab 1: Dataset-wide overview ----------
with tab_overview:
    with st.spinner("Analyzing your data..."):
        eda_result = cached_eda(df)

    if eda_result.available_fields:
        st.caption(f"Available fields: {', '.join(eda_result.available_fields)}")
    if eda_result.unavailable_fields:
        st.caption(f"Not in this dataset: {', '.join(eda_result.unavailable_fields)}")

    stat_cols = st.columns(len(eda_result.stats))
    for col, (name, value) in zip(stat_cols, eda_result.stats.items()):
        label = name.replace("_", " ").title()
        col.metric(label, f"{value:,.0f}" if isinstance(value, (int, float)) else value)

    st.divider()

    chart_grid = st.columns(2)
    for i, (name, fig) in enumerate(eda_result.figures.items()):
        with chart_grid[i % 2]:
            st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 2: Per-product drill-down ----------
with tab_product:
    display_options = get_product_display_options(df)
    selected_label = st.selectbox("Choose a product", list(display_options.values()))
    selected_id = next(pid for pid, label in display_options.items() if label == selected_label)

    product_result = run_product_eda(df, selected_id)

    stat_cols = st.columns(len(product_result.stats))
    for col, (name, value) in zip(stat_cols, product_result.stats.items()):
        label = name.replace("_", " ").title()
        col.metric(label, f"{value:,.1f}" if isinstance(value, (int, float)) else value)

    for name, fig in product_result.figures.items():
        st.plotly_chart(fig, use_container_width=True)

# ---------- Tab 3: Statistical analysis ----------
with tab_stats:
    with st.spinner("Running statistical analysis..."):
        analytics_result = cached_analytics(df)

    if analytics_result.correlations:
        st.markdown("**Correlations**")
        for corr in analytics_result.correlations:
            st.markdown(f"- {corr['interpretation']}")

    if analytics_result.hypothesis_tests:
        st.markdown("**Group comparisons**")
        for test in analytics_result.hypothesis_tests:
            st.markdown(f"- {test['interpretation']}")

    if analytics_result.distribution_tests:
        st.markdown("**Distribution checks**")
        for dist in analytics_result.distribution_tests:
            st.markdown(f"- {dist['interpretation']}")

    if analytics_result.has_figure("correlation_heatmap"):
        st.plotly_chart(analytics_result.figures["correlation_heatmap"], use_container_width=True)