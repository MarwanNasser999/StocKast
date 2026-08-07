"""Stockast -- guided onboarding: upload, map, validate, clean."""

from __future__ import annotations

import os
import tempfile

import pandas as pd
import streamlit as st

from src.common.canonical_schema import ALL_FIELDS, REQUIRED_FIELDS
from src.data_cleaning.cleaner import clean
from src.data_cleaning.exceptions import InsufficientDataError
from src.data_loading.exceptions import DataLoadingError
from src.data_loading.loaders import load_file
from src.data_validation.validator import validate
from src.schema_mapping.applier import apply_mapping
from src.schema_mapping.exceptions import SchemaMappingError
from src.schema_mapping.matcher import suggest_mapping
from src.schema_mapping.result import MappingResult

st.title("🚀 Get Started")

# ---------- Step 1: Upload ----------

st.subheader("1. Upload your data")

uploaded_file = st.file_uploader(
    "CSV, Excel, or JSON",
    type=["csv", "xlsx", "xls", "json"],
    label_visibility="collapsed",
)

if uploaded_file is not None:
    suffix = os.path.splitext(uploaded_file.name)[1]

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    try:
        result = load_file(tmp_path)
        st.session_state["raw_load_result"] = result

        st.success(
            f"Loaded **{result.filename}** — "
            f"{result.row_count} rows, {result.column_count} columns."
        )

    except DataLoadingError as exc:
        st.error(f"Couldn't load this file: {exc}")
        st.session_state["raw_load_result"] = None


# ---------- Step 2: Mapping ----------

if st.session_state.get("raw_load_result") is not None:
    st.divider()
    st.subheader("2. Confirm column mapping")

    raw_df = st.session_state["raw_load_result"].dataframe
    suggestion = suggest_mapping(list(raw_df.columns))

    confirmed_mapping: dict[str, str] = {}

    cols = st.columns(3)

    for i, field in enumerate(ALL_FIELDS):
        with cols[i % 3]:
            options = ["-- not mapped --"] + list(raw_df.columns)
            default = suggestion["mapping"].get(field, "-- not mapped --")
            default_index = options.index(default) if default in options else 0

            label = (
                f"**{field}**"
                + (" *(required)*" if field in REQUIRED_FIELDS else "")
            )

            choice = st.selectbox(
                label,
                options,
                index=default_index,
                key=f"map_{field}",
            )

            if choice != "-- not mapped --":
                confirmed_mapping[field] = choice

    missing_required = [
        field for field in REQUIRED_FIELDS
        if field not in confirmed_mapping
    ]

    if missing_required:
        st.error(
            f"Please map the required field(s): {', '.join(missing_required)}"
        )

    elif st.button("✅ Confirm mapping", type="primary"):
        mapping_result = MappingResult(
            mapping=confirmed_mapping,
            is_confirmed=True,
        )

        try:
            st.session_state["canonical_df"] = apply_mapping(
                raw_df,
                mapping_result,
            )

        except SchemaMappingError as exc:
            st.error(str(exc))


# ---------- Step 3: Validation + Cleaning ----------

if st.session_state.get("canonical_df") is not None:
    st.divider()
    st.subheader("3. Validation & cleaning")

    df = st.session_state["canonical_df"]
    report = validate(df)

    col1, col2, col3 = st.columns(3)

    col1.metric("Status", "Passed" if report.is_valid else "Failed")
    col2.metric("Errors", len(report.errors))
    col3.metric("Warnings", len(report.warnings))

    if report.errors:
        for issue in report.errors:
            st.error(f"**{issue.check_name}**: {issue.message}")

    if report.warnings:
        with st.expander(
            f"⚠️ {len(report.warnings)} warning(s) — data can still be used"
        ):
            for issue in report.warnings:
                st.markdown(
                    f"- **{issue.check_name}**: {issue.message}"
                )

    # Always allow cleaning; clean() decides what is fixable.
    if st.button("🧹 Clean data", type="primary"):
        try:
            cleaned_df, cleaning_report = clean(df)

            st.session_state["cleaned_df"] = cleaned_df
            st.session_state["cleaning_report"] = cleaning_report

        except InsufficientDataError as exc:
            st.error(
                "Dataset too small/short even after cleaning:\n\n"
                + "\n".join(f"- {reason}" for reason in exc.reasons)
            )

    if st.session_state.get("cleaning_report") is not None:
        cr = st.session_state["cleaning_report"]

        st.metric("Rows", f"{cr.rows_before} → {cr.rows_after}")

        for action in cr.actions:
            st.caption(f"- {action.message}")

        st.success("Data is ready.")

        if st.button("Go to Dashboard →", type="primary"):
            st.switch_page("views/dashboard_home.py")