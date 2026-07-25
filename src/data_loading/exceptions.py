"""
Exceptions for src.data_loading.

Kept narrow and specific on purpose: a caller (e.g. the Streamlit UI in a
later phase) should be able to catch UnsupportedFileFormatError vs
FileLoadError and show the user a precise, actionable message instead of
a raw traceback.
"""


class DataLoadingError(Exception):
    """Base class for all data_loading errors. Callers can catch this if
    they just want to know "loading failed" without caring which subtype."""


class UnsupportedFileFormatError(DataLoadingError):
    """Raised when the uploaded file's extension is not one we support."""

    def __init__(self, filename: str, supported_formats: list[str]):
        self.filename = filename
        self.supported_formats = supported_formats
        message = (
            f"'{filename}' is not a supported file format. "
            f"StockSense currently supports: {', '.join(supported_formats)}."
        )
        super().__init__(message)


class FileLoadError(DataLoadingError):
    """Raised when a file has a supported extension but still fails to
    load (corrupted file, undetectable encoding, unreadable Excel sheet,
    malformed JSON, etc.). Wraps the underlying error with a plain-English
    explanation."""

    def __init__(self, filename: str, reason: str):
        self.filename = filename
        self.reason = reason
        message = f"Could not load '{filename}': {reason}"
        super().__init__(message)