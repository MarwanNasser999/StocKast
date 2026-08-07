"""Stockast -- Risk & Recommendations page."""

from __future__ import annotations

import streamlit as st

from app.state import init_session_state, cached_inventory_ml, cached_recommendations
from src.eda.product_analysis import get_product_display_options
from src.inventory_ml.explainability import explain_product_risk

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

init_session_state()

st.title("⚠️ Risk & Recommendations")

if st.session_state.get("cleaned_df") is None:
    st.warning("Complete validation & cleaning first.")
    st.stop()

df = st.session_state["cleaned_df"]

tab_risk, tab_recs = st.tabs(["Stockout Risk", "Recommendations"])

# ---------- Tab 1: Risk assessment ----------
with tab_risk:
    with st.spinner("Assessing stockout risk..."):
        ml_result = cached_inventory_ml(df)

    if ml_result["path"] == "unavailable":
        st.info(
            f"Machine-learning risk prediction isn't available for this dataset: {ml_result['reason']} "
            f"A simpler rule-based estimate is used instead on the Recommendations tab."
        )
    else:
        st.success(
            f"Trained on this dataset's own stockout history "
            f"(accuracy: {ml_result['metrics']['accuracy']:.0%}, recall: {ml_result['metrics']['recall']:.0%})."
        )

        accuracy = ml_result["metrics"]["accuracy"]

        if accuracy < 0.65:
            st.caption(
                "⚠️ This model's accuracy is on the lower side — usually because there isn't yet enough "
                "stockout history in this dataset for it to learn reliable patterns. Treat its predictions "
                "as a rough signal, not a certainty, and lean more on the KPI numbers above."
            )
        else:
            st.caption(
                "✅ This level of accuracy suggests the model has learned useful patterns from this dataset's "
                "historical stockout behavior. While no model is perfect, its predictions can generally be "
                "treated as more reliable when interpreted alongside the KPI metrics."
            )

        display_options = get_product_display_options(df)
        risk_table = ml_result["risk_table"]
        available_ids = set(risk_table["product_id"])
        options = {
            pid: label
            for pid, label in display_options.items()
            if pid in available_ids
        }

        selected_label = st.selectbox(
            "Inspect a product's risk",
            list(options.values()),
        )
        selected_id = next(
            pid for pid, label in options.items()
            if label == selected_label
        )

        risk_row = risk_table[risk_table["product_id"] == selected_id].iloc[0]

        col1, col2 = st.columns(2)
        col1.metric("Risk level", risk_row["risk_label"].title())
        col2.metric("Risk score", f"{risk_row['risk_score']:.0%}")

        feature_table = ml_result["feature_table"]
        product_rows = feature_table[
            feature_table["product_id"] == selected_id
        ]

        if not product_rows.empty:
            latest_snapshot = product_rows.sort_values(
                "snapshot_date"
            ).iloc[-1]

            explanation = explain_product_risk(
                ml_result["explainer"],
                latest_snapshot,
            )

            st.markdown(f"**Why:** {explanation['summary']}")

            with st.expander("Full breakdown"):
                for feature, contribution in explanation["contributions"].items():
                    st.markdown(
                        f"- {feature.replace('_', ' ').title()}: `{contribution:+.3f}`"
                    )

        st.dataframe(
            risk_table.sort_values("risk_score", ascending=False),
            use_container_width=True,
            hide_index=True,
        )

# ---------- Tab 2: Recommendations ----------
with tab_recs:
    with st.spinner("Generating recommendations..."):
        recommendations = cached_recommendations(df)

    if not recommendations:
        st.info(
            "No specific actions are recommended right now — inventory looks healthy across the board."
        )
    else:
        priority_filter = st.multiselect(
            "Filter by priority",
            ["high", "medium", "low"],
            default=["high", "medium", "low"],
        )

        action_labels = {
            "increase_inventory": "📦 Increase inventory",
            "reduce_inventory": "📉 Reduce inventory",
            "discount": "🏷️ Discount",
            "increase_safety_stock": "🛡️ Increase safety stock",
            "consider_price_increase": "⬆️ Consider price increase",
            "consider_price_decrease": "⬇️ Consider price decrease",
        }

        priority_colors = {
            "high": "🔴",
            "medium": "🟡",
            "low": "🟢",
        }

        filtered = [
            r for r in recommendations
            if r["priority"] in priority_filter
        ]

        st.caption(f"{len(filtered)} recommendation(s)")

        for rec in filtered:
            with st.container(border=True):
                st.markdown(
                    f"{priority_colors[rec['priority']]} **{rec['display_name']}** — "
                    f"{action_labels.get(rec['action'], rec['action'])}"
                )
                st.caption(rec["reasoning"])