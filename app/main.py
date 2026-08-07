"""Stockast -- entry point / router."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import streamlit as st

from app.state import init_session_state

st.set_page_config(page_title="Stockast", page_icon="📦", layout="wide")
init_session_state()

onboarding_page = st.Page("views/onboarding.py", title="Get Started", icon="🚀", default=True)
home_page = st.Page("views/dashboard_home.py", title="Dashboard", icon="🏠")
explore_page = st.Page("views/explore.py", title="Explore Your Data", icon="📊")
kpis_page = st.Page("views/kpis_forecast.py", title="KPIs & Forecast", icon="📈")
price_page = st.Page("views/price_whatif.py", title="Price What-If", icon="💰")
risk_page = st.Page("views/risk_recommendations.py", title="Risk & Recommendations", icon="⚠️")
report_page = st.Page("views/ai_report.py", title="AI Report", icon="📝")

if st.session_state.get("cleaned_df") is None:
    pg = st.navigation([onboarding_page])
else:
    pg = st.navigation({
        "Overview": [home_page],
        "Analysis": [explore_page, kpis_page, price_page, risk_page],
        "Reports": [report_page],
    })

pg.run()