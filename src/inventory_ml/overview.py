"""
Orchestrator for src.inventory_ml.

Only attempts Path A (real trained classifier, walk-forward point-in-time
features). If the dataset can't support it (checked via
labeling.can_build_ml_classifier BEFORE any feature building or training
is attempted), reports honestly that ML risk prediction is unavailable
for this dataset -- does NOT fall back to a rule-based score here. A
simpler DOI/XYZ/trend-based risk convenience feature belongs in
recommendation_engine (Phase 11), explicitly labeled as a formula, not
presented as this module's output.
"""

from __future__ import annotations

import pandas as pd

from src.inventory_ml.explainability import build_explainer
from src.inventory_ml.labeling import can_build_ml_classifier
from src.inventory_ml.risk_model import build_feature_table, train_risk_model
from src.inventory_ml.risk_model import build_feature_table, train_risk_model, build_risk_table

def run_inventory_ml(df: pd.DataFrame) -> dict:
    can_build, reason = can_build_ml_classifier(df)
    if not can_build:
        return {"path": "unavailable", "reason": reason}

    feature_table = build_feature_table(df)
    if feature_table.empty:
        return {"path": "unavailable", "reason": "Could not build any valid walk-forward training examples from this dataset."}

    model, metrics = train_risk_model(feature_table)
    if model is None:
        return {"path": "unavailable", "reason": metrics.get("error", "Model training failed.")}

    explainer = build_explainer(model)
    risk_table = build_risk_table(model, feature_table)

    return {
        "path": "ml_classifier",
        "model": model,
        "explainer": explainer,
        "metrics": metrics,
        "feature_table": feature_table,
        "risk_table": risk_table,
    }


def run_inventory_ml(df: pd.DataFrame) -> dict:
    """
    Attempts to train a real, walk-forward stockout risk classifier.

    Returns either:
    {"path": "unavailable", "reason": "..."}
    or
    {"path": "ml_classifier", "model": ..., "explainer": ..., "metrics": {...}, "feature_table": ...}
    """
    can_build, reason = can_build_ml_classifier(df)
    if not can_build:
        return {"path": "unavailable", "reason": reason}

    feature_table = build_feature_table(df)

    if feature_table.empty:
        return {"path": "unavailable", "reason": "Could not build any valid walk-forward training examples from this dataset."}

    model, metrics = train_risk_model(feature_table)

    if model is None:
        return {"path": "unavailable", "reason": metrics.get("error", "Model training failed.")}

    explainer = build_explainer(model)

    return {
        "path": "ml_classifier",
        "model": model,
        "explainer": explainer,
        "metrics": metrics,
        "feature_table": feature_table,
    }