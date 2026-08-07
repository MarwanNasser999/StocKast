"""Stockast -- Dashboard hub."""

from __future__ import annotations

import streamlit as st

st.title("🏠 Dashboard")
st.caption("Your data is ready. Choose where to go.")

cards = [
    ("views/explore.py", "📊 Explore Your Data", "Charts, trends, and per-product breakdowns."),
    ("views/kpis_forecast.py", "📈 KPIs & Forecast", "Turnover, reorder points, demand forecasts."),
    ("views/price_whatif.py", "💰 Price What-If", "See the projected impact of a price change."),
    ("views/risk_recommendations.py", "⚠️ Risk & Recommendations", "Stockout risk and suggested actions."),
    ("views/ai_report.py", "📝 AI Report", "A plain-English summary of everything above."),
]

cols = st.columns(2)
for i, (path, label, desc) in enumerate(cards):
    with cols[i % 2]:
        with st.container(border=True):
            st.markdown(f"### {label}")
            st.caption(desc)
            st.page_link(path, label="Open →")