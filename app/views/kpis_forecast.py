"""Stockast -- KPIs & Forecast page."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
import plotly.express as px
import streamlit as st

from app.state import init_session_state, cached_kpis, show_table_or_message
from src.eda.product_analysis import get_product_display_options
from src.forecasting.overview import run_forecasting

st.title("📈 KPIs & Forecast")

if st.session_state.get("cleaned_df") is None:
    st.warning("Complete the onboarding steps first.")
    st.stop()

df = st.session_state["cleaned_df"]

with st.sidebar:
    st.markdown("**Safety stock settings**")
    service_level_pct = st.select_slider(
        "How confident do you want to be that you won't run out of stock?",
        options=["90%", "95%", "97.5%", "99%"], value="95%",
    )
    Z_LOOKUP = {"90%": 1.28, "95%": 1.65, "97.5%": 1.96, "99%": 2.33}
    service_level_z = Z_LOOKUP[service_level_pct]

    default_lead_time = st.number_input(
        "Default lead time (days)", min_value=1, value=7,
        help="Used if your data doesn't include a lead_time_days column.",
    )

with st.spinner("Computing KPIs..."):
    kpi_result = cached_kpis(df, service_level_z, float(default_lead_time))

tab_summary, tab_abc_xyz, tab_forecast = st.tabs(["Summary KPIs", "ABC / XYZ Analysis", "Demand Forecast"])

# ---------- Tab 1: Core KPI tables ----------
with tab_summary:
    st.markdown("**Inventory Turnover**")
    show_table_or_message(
        kpi_result.turnover,
        "This dataset doesn't include stock-level data, so turnover can't be calculated.",
    )

    if kpi_result.days_of_inventory is not None and not kpi_result.days_of_inventory.empty:
        st.markdown("**Days of Inventory**")
        st.caption("10 products with the least inventory remaining:")
        low_doi = kpi_result.days_of_inventory.sort_values("days_of_inventory").head(10)
        show_table_or_message(low_doi, "No days-of-inventory data available.")
    else:
        st.markdown("**Days of Inventory**")
        show_table_or_message(
            kpi_result.days_of_inventory,
            "This dataset doesn't include stock-level data, so days of inventory can't be calculated.",
        )

    st.markdown("**Dead / Slow-Moving Stock**")
    if kpi_result.dead_and_slow_stock is not None and not kpi_result.dead_and_slow_stock.empty:
        flagged = kpi_result.dead_and_slow_stock[
            kpi_result.dead_and_slow_stock["is_dead_stock"] | kpi_result.dead_and_slow_stock["is_slow_mover"]
        ]
        st.caption(f"{len(flagged)} product(s) flagged. Gross sales, returns, and net are shown separately.")
        show_table_or_message(
            flagged[["product_id", "gross_quantity_sold", "returns", "net_quantity_sold",
                     "days_since_last_sale", "is_dead_stock", "is_slow_mover"]],
            "No dead or slow-moving stock detected.",
        )
    else:
        show_table_or_message(kpi_result.dead_and_slow_stock, "No sales data available to assess.")

    st.markdown("**Reorder Point & Safety Stock**")
    show_table_or_message(
        kpi_result.reorder_point,
        "Couldn't calculate reorder points — this dataset may be missing stock-level or lead-time data.",
    )

    if kpi_result.seasonality:
        st.markdown("**Seasonality**")
        for period_label, result in kpi_result.seasonality.items():
            if "error" not in result:
                st.markdown(f"- {result['interpretation']}")
            else:
                st.caption(f"- {period_label.title()}: {result['error']}")

# ---------- Tab 2: ABC / XYZ ----------
with tab_abc_xyz:
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**ABC Analysis**")
        if kpi_result.abc_analysis is not None and not kpi_result.abc_analysis.empty:
            st.caption(f"Basis: {kpi_result.abc_analysis['basis'].iloc[0]}")
            show_table_or_message(kpi_result.abc_analysis, "No ABC data available.")
            tier_counts = kpi_result.abc_analysis["tier"].value_counts().reset_index()
            st.plotly_chart(px.pie(tier_counts, names="tier", values="count", title="Products by ABC tier"),
                             use_container_width=True)
        else:
            st.info("Not enough data to compute ABC analysis.")

    with col2:
        st.markdown("**XYZ Analysis**")
        if kpi_result.xyz_analysis is not None and not kpi_result.xyz_analysis.empty:
            show_table_or_message(kpi_result.xyz_analysis, "No XYZ data available.")
            tier_counts = kpi_result.xyz_analysis["xyz_tier"].value_counts().reset_index()
            st.plotly_chart(px.pie(tier_counts, names="xyz_tier", values="count", title="Products by XYZ tier"),
                             use_container_width=True)
        else:
            st.info("Not enough data to compute XYZ analysis.")

# ---------- Tab 3: Forecast ----------
with tab_forecast:
    display_options = get_product_display_options(df)

    if "forecast_results" not in st.session_state:
        status_placeholder = st.empty()

        def update_progress(completed, total):
            status_placeholder.text(f"Forecasting products... {completed}/{total}")

        with st.spinner("Preparing forecasts... this can take a while for large datasets"):
            st.session_state["forecast_results"] = run_forecasting(df, on_progress=update_progress)

        status_placeholder.empty()

    forecast_results = st.session_state["forecast_results"]

    selected_label = st.selectbox("Choose a product to forecast", list(display_options.values()))
    selected_id = next(pid for pid, label in display_options.items() if label == selected_label)

    result = forecast_results.get(selected_id, {})

    if "error" in result:
        st.warning(f"Couldn't forecast this product: {result['error']}")
    else:
        st.caption(f"Technique used: **{result['technique_used']}** "
                    f"(MAE={result['test_mae']:.2f} on held-back test data)")

        forecast_df = result["forecast"].reset_index()
        forecast_df.columns = ["date", "forecasted_quantity"]

        st.plotly_chart(
            px.line(forecast_df, x="date", y="forecasted_quantity", title=f"14-day forecast: {selected_label}"),
            use_container_width=True,
        )

        with st.expander("See how each technique scored"):
            scores_df = pd.DataFrame(result["all_scores"].items(), columns=["Technique", "MAE"])
            st.dataframe(scores_df, use_container_width=True, hide_index=True)