"""Unit tests for src.analytics.distributions.tests."""

import numpy as np
import pandas as pd

from src.analytics.distributions.tests import test_normality as run_normality_test
from src.analytics.distributions.tests import test_all_distributions as run_all_distributions_test


def test_normal_data_is_detected_as_normal():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"quantity_sold": rng.normal(loc=50, scale=5, size=500)})

    result = run_normality_test(df, "quantity_sold")

    assert result["is_normal"] is True


def test_skewed_data_is_detected_as_not_normal():
    rng = np.random.default_rng(42)
    df = pd.DataFrame({"quantity_sold": rng.exponential(scale=5, size=500)})

    result = run_normality_test(df, "quantity_sold")

    assert result["is_normal"] is False


def test_insufficient_data_returns_error():
    df = pd.DataFrame({"quantity_sold": [10, 20]})
    result = run_normality_test(df, "quantity_sold")

    assert "error" in result


def test_run_all_distributions_covers_available_fields():
    rng = np.random.default_rng(1)
    df = pd.DataFrame({
        "quantity_sold": rng.normal(50, 5, size=200),
        "unit_price": rng.normal(20, 2, size=200),
    })
    results = run_all_distributions_test(df)

    fields_tested = {r["field"] for r in results}
    assert fields_tested == {"quantity_sold", "unit_price"}