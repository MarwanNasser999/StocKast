"""Exceptions for src.data_cleaning."""


class DataCleaningError(Exception):
    """Base class for all data_cleaning errors."""


class InsufficientDataError(DataCleaningError):
    """Raised when minimum_row_count or minimum_date_span errors are still
    present -- these cannot be fixed by cleaning, only reported. Cleaning
    a few bad rows doesn't manufacture more data out of nothing."""

    def __init__(self, reasons: list[str]):
        self.reasons = reasons
        super().__init__(
            "This dataset is too small or too short to proceed, even after "
            f"cleaning: {'; '.join(reasons)}."
        )