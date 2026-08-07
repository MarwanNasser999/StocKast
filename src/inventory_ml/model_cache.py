"""
Model caching for src.inventory_ml.

Avoids retraining the classifier on every run when the dataset hasn't
changed. Uses a hash of the dataset's actual content as the cache key
-- if the hash matches a previously saved model, that model is reused
instead of retraining from scratch.
"""

from __future__ import annotations

import hashlib
import pickle
from pathlib import Path

import pandas as pd

CACHE_DIR = Path("storage/model_cache")


def _dataset_fingerprint(df: pd.DataFrame) -> str:
    """A stable hash of the dataset's actual content -- changes if and
    only if the underlying data genuinely changes."""
    content_bytes = pd.util.hash_pandas_object(df, index=True).values.tobytes()
    return hashlib.sha256(content_bytes).hexdigest()


def load_cached_model(df: pd.DataFrame):
    """Returns the cached (model, metrics, feature_table) tuple if the
    dataset's fingerprint matches a saved cache entry, otherwise None."""
    fingerprint = _dataset_fingerprint(df)
    cache_file = CACHE_DIR / f"{fingerprint}.pkl"

    if not cache_file.exists():
        return None

    try:
        with open(cache_file, "rb") as f:
            return pickle.load(f)
    except Exception:
        return None  # corrupted cache file -- fail safe, just retrain


def save_model_to_cache(df: pd.DataFrame, model, metrics: dict, feature_table: pd.DataFrame) -> None:
    """Saves the trained model, keyed by the dataset's fingerprint."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fingerprint = _dataset_fingerprint(df)
    cache_file = CACHE_DIR / f"{fingerprint}.pkl"

    with open(cache_file, "wb") as f:
        pickle.dump((model, metrics, feature_table), f)