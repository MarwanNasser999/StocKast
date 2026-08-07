"""Stockast -- Price Elasticity What-If page."""

from __future__ import annotations

import streamlit as st

from app.state import init_session_state, cached_price_elasticity
from src.eda.product_analysis import get_product_display_options
from src.price_elasticity.what_if import project_price_change
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
init_session_state()

st.title("💰 Price What-If")
st.caption("Estimate the revenue impact of raising or lowering a product's price, based on its own historical demand response.")

if st.session_state.get("cleaned_df") is None:
    st.warning("Complete validation & cleaning first.")
    st.stop()

df = st.session_state["cleaned_df"]

with st.spinner("Estimating price sensitivity..."):
    elasticity_result = cached_price_elasticity(df)

if elasticity_result.unavailable_reason:
    st.info(elasticity_result.unavailable_reason)
    st.stop()

elasticity_table = elasticity_result.elasticity_table
display_options = get_product_display_options(df)

# only offer products that actually have an elasticity estimate
available_ids = set(elasticity_table["product_id"])
options = {pid: label for pid, label in display_options.items() if pid in available_ids}

if not options:
    st.info("No products in this dataset had enough price variation to estimate elasticity.")
    st.stop()

col1, col2 = st.columns([2, 1])

with col1:
    selected_label = st.selectbox("Choose a product", list(options.values()))
selected_id = next(pid for pid, label in options.items() if label == selected_label)

row = elasticity_table[elasticity_table["product_id"] == selected_id].iloc[0]

with col2:
    st.metric("Elasticity classification", row["classification"].replace("_", " ").title())

if row["classification"] == "suspect_extreme":
    st.warning(
        "This product's estimated elasticity is statistically implausible "
        "(likely due to limited or noisy price data) — treat this projection with caution."
    )

st.divider()

price_change_pct = st.slider("Hypothetical price change (%)", -50, 50, 10, 1)

if st.button("Project impact", type="primary"):
    with st.spinner("Running projection..."):
        projection = project_price_change(df, selected_id, price_change_pct)

    if "error" in projection:
        st.error(projection["error"])
    else:
        c1, c2, c3 = st.columns(3)
        c1.metric("New price", f"${projection['new_price']:.2f}",
                   delta=f"{price_change_pct:+d}%")
        c2.metric("Projected units (14 days)", f"{projection['projected_units']:.0f}",
                   delta=f"{projection['projected_demand_change_pct']:+.1f}%")
        c3.metric("Projected revenue", f"${projection['projected_revenue']:.2f}",
                   delta=f"{projection['revenue_change_pct']:+.1f}%")

        if projection["revenue_change_pct"] > 0:
            st.success(f"This change is estimated to **increase** revenue by {projection['revenue_change_pct']:.1f}%.")
        else:
            st.warning(f"This change is estimated to **decrease** revenue by {abs(projection['revenue_change_pct']):.1f}%.")

        st.caption(
            "This is a linear approximation based on historical price/demand data for this "
            "specific product — treat it as a directional estimate, not a guarantee."
        )