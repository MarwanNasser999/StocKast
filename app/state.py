"""
Session state + caching helpers for the StockSense app.

Streamlit reruns the ENTIRE script on every interaction. session_state
persists values (like the uploaded/cleaned DataFrame) across those
reruns. @st.cache_data avoids re-running expensive pipeline steps
(forecasting, ML training) when the underlying data hasn't changed.
"""

from __future__ import annotations

import streamlit as st
import pandas as pd

from src.analytics.overview import run_analytics
from src.eda.overview import run_eda
from src.forecasting.overview import run_forecasting
from src.inventory_ml.overview import run_inventory_ml
from src.kpis.overview import run_kpis
from src.price_elasticity.overview import run_price_elasticity
from src.recommendation_engine.overview import run_recommendations

SESSION_KEYS = [
    "raw_load_result", "mapping_result", "canonical_df",
    "validation_report", "cleaning_report", "cleaned_df",
]


def init_session_state() -> None:
    for key in SESSION_KEYS:
        if key not in st.session_state:
            st.session_state[key] = None


def has_cleaned_data() -> bool:
    return st.session_state.get("cleaned_df") is not None


@st.cache_data(show_spinner=False)
def cached_eda(df: pd.DataFrame):
    return run_eda(df)


@st.cache_data(show_spinner=False)
def cached_analytics(df: pd.DataFrame):
    return run_analytics(df)


@st.cache_data(show_spinner=False)
def cached_kpis(df: pd.DataFrame, service_level_z: float, default_lead_time_days):
    return run_kpis(df, service_level_z, default_lead_time_days)


@st.cache_data(show_spinner=False)
def cached_forecasting(df: pd.DataFrame):
    return run_forecasting(df)


@st.cache_data(show_spinner=False)
def cached_inventory_ml(df: pd.DataFrame):
    return run_inventory_ml(df)


@st.cache_data(show_spinner=False)
def cached_price_elasticity(df: pd.DataFrame):
    return run_price_elasticity(df)


@st.cache_data(show_spinner=False)
def cached_recommendations(df: pd.DataFrame):
    return run_recommendations(df)

def show_table_or_message(table, message: str, **kwargs) -> None:
    """Shows a table, or an explanatory message if it's None or empty --
    never a silently blank table."""
    if table is None or (hasattr(table, "empty") and table.empty):
        st.info(message)
    else:
        st.dataframe(table, use_container_width=True, hide_index=True, **kwargs)