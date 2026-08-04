"""
SHAP explainability for src.inventory_ml.

Explains individual risk predictions: for one specific product, how
much did each feature push the predicted risk up or down, relative to
the average prediction across all products (the "base value").

Handles both SHAP API shapes: older versions return a list of arrays
(one per class), newer versions return a single array shaped
(n_samples, n_features, n_classes) -- verified directly against the
installed version rather than assumed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap

FEATURE_COLS = ["days_of_inventory", "demand_volatility", "trailing_demand_slope", "trailing_total_demand"]


def build_explainer(model):
    """Builds a SHAP TreeExplainer for the trained Random Forest --
    fast and exact for tree-based models, unlike general-purpose SHAP
    methods needed for other model types."""
    return shap.TreeExplainer(model)


def explain_product_risk(explainer, feature_row: pd.Series) -> dict:
    """
    Explains one specific product's prediction.

    feature_row: a single row (as a Series) from the feature table,
    containing at least FEATURE_COLS values for one product's snapshot.

    Returns the base value (average predicted risk across all
    products), each feature's individual contribution, and a
    plain-language summary naming the top contributing factor.
    """
    X = feature_row[FEATURE_COLS].to_frame().T
    raw = explainer.shap_values(X)

    if isinstance(raw, list):
        # older SHAP API: list of arrays, one per class
        class_1_values = np.asarray(raw[1])[0]
        base_value = explainer.expected_value[1]
    else:
        raw = np.asarray(raw)
        if raw.ndim == 3:
            # newer SHAP API: shape (n_samples, n_features, n_classes)
            class_1_values = raw[0, :, 1]
        else:
            # binary case collapsed to a single output: shape (n_samples, n_features)
            class_1_values = raw[0]
        base_value = (
            explainer.expected_value[1]
            if hasattr(explainer.expected_value, "__len__")
            else explainer.expected_value
        )

    contributions = {
        feature: float(value) for feature, value in zip(FEATURE_COLS, class_1_values)
    }

    top_feature = max(contributions, key=lambda f: abs(contributions[f]))
    top_direction = "increased" if contributions[top_feature] > 0 else "decreased"

    return {
        "base_value": float(base_value),
        "contributions": contributions,
        "top_factor": top_feature,
        "summary": (
            f"Risk was most strongly {top_direction} by '{top_feature}' "
            f"(contribution: {contributions[top_feature]:+.3f})."
        ),
    }