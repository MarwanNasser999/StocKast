"""Unit tests for src.analytics.hypothesis.tests."""

import numpy as np
import pandas as pd

from src.analytics.hypothesis.tests import compare_two_groups, compare_top_two_groups


def make_df():
    return pd.DataFrame({
        "category": ["A"] * 20 + ["B"] * 20,
        "quantity_sold": [50] * 20 + [10] * 20,  # clearly different
    })


def test_compare_two_groups_detects_real_difference():
    df = make_df()
    result = compare_two_groups(df, "quantity_sold", "category", "A", "B")

    assert result["is_significant"] is True
    assert result["p_value"] < 0.05


def test_compare_two_groups_no_difference():
    rng = np.random.default_rng(7)
    df = pd.DataFrame({
        "category": ["A"] * 50 + ["B"] * 50,
        "quantity_sold": list(rng.normal(loc=10, scale=3, size=50)) +
                         list(rng.normal(loc=10, scale=3, size=50)),
    })
    result = compare_two_groups(df, "quantity_sold", "category", "A", "B")

    assert result["is_significant"] is False


def test_compare_two_groups_insufficient_data():
    df = pd.DataFrame({"category": ["A"], "quantity_sold": [10]})
    result = compare_two_groups(df, "quantity_sold", "category", "A", "B")

    assert "error" in result


def test_compare_top_two_groups_picks_largest():
    df = pd.DataFrame({
        "category": ["A"] * 20 + ["B"] * 15 + ["C"] * 2,
        "quantity_sold": [50] * 20 + [10] * 15 + [999] * 2,
    })
    result = compare_top_two_groups(df, "quantity_sold", "category")

    assert {result["group_a"], result["group_b"]} == {"A", "B"}