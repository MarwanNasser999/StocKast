"""Stockast -- AI-Generated Report page."""

from __future__ import annotations

import streamlit as st

from app.state import init_session_state, cached_inventory_ml, cached_recommendations
from src.ai_assistant.report_generator import generate_executive_summary, generate_product_report
from src.eda.product_analysis import get_product_display_options


init_session_state()
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

st.title("📝 AI-Generated Report")
st.caption("Turns the analysis above into a plain-English summary. The AI only narrates numbers already computed by Stockast — it never invents figures or makes its own judgment calls.")

if st.session_state.get("cleaned_df") is None:
    st.warning("Complete validation & cleaning first.")
    st.stop()

df = st.session_state["cleaned_df"]

with st.spinner("Gathering analysis..."):
    ml_result = cached_inventory_ml(df)
    recommendations = cached_recommendations(df)

risk_table = ml_result.get("risk_table") if ml_result["path"] == "ml_classifier" else None

tab_exec, tab_product = st.tabs(["Executive Summary", "Product Report"])

# ---------- Tab 1: Executive summary ----------
with tab_exec:
    if st.button("Generate executive summary", type="primary"):
        high_risk_count = int((risk_table["risk_label"] == "high").sum()) if risk_table is not None else 0
        dead_stock_count = sum(1 for r in recommendations if r["action"] == "discount")

        with st.spinner("Writing report..."):
            result = generate_executive_summary(
                recommendations,
                total_products=df["product_id"].nunique(),
                high_risk_count=high_risk_count,
                dead_stock_count=dead_stock_count,
            )

        if result["error"]:
            st.error(f"Couldn't generate the report: {result['error']}")
            st.caption("You can still review the full KPIs and recommendations on the other pages.")
        else:
            st.markdown(result["text"])

# ---------- Tab 2: Per-product report ----------
with tab_product:
    display_options = get_product_display_options(df)
    selected_label = st.selectbox("Choose a product", list(display_options.values()))
    selected_id = next(pid for pid, label in display_options.items() if label == selected_label)

    if st.button("Generate product report", type="primary"):
        risk_row = {"risk_label": "unknown", "risk_score": "unknown", "days_of_inventory": "unknown"}
        if risk_table is not None:
            rows = risk_table[risk_table["product_id"] == selected_id]
            if not rows.empty:
                risk_row = rows.iloc[0].to_dict()

        recommendation = next((r for r in recommendations if r["product_id"] == selected_id), None)

        with st.spinner("Writing report..."):
            result = generate_product_report(selected_label, risk_row, recommendation)

        if result["error"]:
            st.error(f"Couldn't generate the report: {result['error']}")
        else:
            st.markdown(result["text"])