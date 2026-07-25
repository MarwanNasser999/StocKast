"""
Loaders for src.data_loading.

Each function here has exactly one job (single responsibility), so each
can be unit-tested in isolation:
- detect_file_format: extension -> format string
- detect_encoding: raw bytes -> best-guess text encoding
- load_csv / load_excel / load_json: format -> raw DataFrame
- load_file: the public entry point that orchestrates all of the above
  and returns a LoadResult

IMPORTANT: nothing in this file knows about the canonical schema, column
names, or business meaning. It only turns bytes on disk into a raw
DataFrame. That boundary is intentional (see module design discussion).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
from charset_normalizer import from_path

from src.data_loading.exceptions import FileLoadError, UnsupportedFileFormatError
from src.data_loading.result import LoadResult

SUPPORTED_FORMATS = ["csv", "excel", "json"]

_EXTENSION_MAP = {
    ".csv": "csv",
    ".xlsx": "excel",
    ".xls": "excel",
    ".json": "json",
}


def detect_file_format(filepath: Path) -> str:
    """Determine file format purely from the file extension.

    We deliberately don't sniff file contents for v1 — extension-based
    detection is simple, predictable, and matches how users actually name
    their exports. Content-sniffing can be a future improvement if we ever
    see mislabeled files in practice.
    """
    suffix = filepath.suffix.lower()
    if suffix not in _EXTENSION_MAP:
        raise UnsupportedFileFormatError(filepath.name, SUPPORTED_FORMATS)
    return _EXTENSION_MAP[suffix]


def detect_encoding(filepath: Path) -> str:
    """Best-guess text encoding for a CSV/JSON file using charset-normalizer.

    Only meaningful for text-based formats. Excel files are binary
    (handled by openpyxl/pandas directly) and never call this function.
    """
    try:
        result = from_path(str(filepath)).best()
    except Exception as exc:  # pragma: no cover - defensive, library-level failure
        raise FileLoadError(
            filepath.name,
            f"encoding detection failed unexpectedly ({exc})."
        ) from exc

    if result is None:
        raise FileLoadError(
            filepath.name,
            "could not determine the file's text encoding. "
            "The file may be corrupted or not a text file."
        )
    return result.encoding


def load_csv(filepath: Path, encoding: str) -> pd.DataFrame:
    try:
        return pd.read_csv(filepath, encoding=encoding)
    except Exception as exc:
        raise FileLoadError(
            filepath.name,
            f"failed to parse as CSV using encoding '{encoding}' ({exc})."
        ) from exc


def load_excel(filepath: Path) -> pd.DataFrame:
    try:
        return pd.read_excel(filepath)
    except Exception as exc:
        raise FileLoadError(filepath.name, f"failed to parse as Excel ({exc}).") from exc


def load_json(filepath: Path, encoding: str) -> pd.DataFrame:
    try:
        return pd.read_json(filepath, encoding=encoding)
    except Exception as exc:
        raise FileLoadError(
            filepath.name,
            f"failed to parse as JSON using encoding '{encoding}' ({exc})."
        ) from exc


def load_file(filepath: str | Path) -> LoadResult:
    """Public entry point: load any supported file into a LoadResult.

    Raises:
        UnsupportedFileFormatError: extension isn't csv/excel/json.
        FileLoadError: file matched a supported extension but couldn't
            actually be parsed (bad encoding, corrupted content, etc.).
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileLoadError(filepath.name, "file does not exist at the given path.")

    file_format = detect_file_format(filepath)

    detected_encoding = None
    if file_format == "csv":
        detected_encoding = detect_encoding(filepath)
        dataframe = load_csv(filepath, detected_encoding)
    elif file_format == "excel":
        dataframe = load_excel(filepath)
    elif file_format == "json":
        detected_encoding = detect_encoding(filepath)
        dataframe = load_json(filepath, detected_encoding)
    else:  # pragma: no cover - guarded by detect_file_format already
        raise UnsupportedFileFormatError(filepath.name, SUPPORTED_FORMATS)

    return LoadResult(
        dataframe=dataframe,
        filename=filepath.name,
        file_format=file_format,
        detected_encoding=detected_encoding,
        row_count=len(dataframe),
        column_count=len(dataframe.columns),
        loaded_at=datetime.now(),
    )