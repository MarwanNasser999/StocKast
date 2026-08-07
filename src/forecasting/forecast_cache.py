"""
Forecasting result caching for src.forecasting.

Mirrors src.inventory_ml.model_cache: avoids re-running Naive/ETS/ARIMA
across every product when the dataset hasn't changed. Cache key is a
hash of the dataset's actual content, so it invalidates automatically
whenever the underlying data changes.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("storage/forecast_cache")


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    content_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(content_bytes).hexdigest()


def load_cached_forecast(df: pd.DataFrame) -> dict | None:
    fingerprint = _dataset_fingerprint(df)
    cache_file = CACHE_DIR / f"{fingerprint}.pkl"
    if not cache_file.exists():
        return None
    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None  # corrupted cache -- fail safe, just recompute


def save_forecast_to_cache(df: pd.DataFrame, results: dict) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = _dataset_fingerprint(df)
    cache_file = CACHE_DIR / f"{fingerprint}.pkl"
    with open(cache_file, "wb") as f:
        pickle.dump(results, f)