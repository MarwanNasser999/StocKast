"""
Exceptions for src.schema_mapping.

Same pattern as src.data_loading.exceptions: narrow, specific error types
carrying structured context, so callers can catch precisely what went
wrong and build a clear message instead of a raw traceback.
"""


class SchemaMappingError(Exception):
    """Base class for all schema_mapping errors."""


class NoColumnsToMapError(SchemaMappingError):
    """Raised when suggest_mapping() is called with an empty column list --
    there is nothing to match against, which usually means the uploaded
    file loaded with zero columns (a data_loading issue upstream)."""

    def __init__(self):
        super().__init__(
            "No raw columns were provided to map. The uploaded file may "
            "have loaded with zero columns."
        )


class MappingNotConfirmedError(SchemaMappingError):
    """Raised by apply_mapping() (built next) if it's ever called on a
    MappingResult where is_confirmed is still False -- this is the
    code-level enforcement of 'never rename real columns based on an
    unreviewed suggestion.'"""

    def __init__(self):
        super().__init__(
            "This mapping has not been confirmed by the user yet. "
            "Refusing to apply it to the DataFrame."
        )


class MissingRequiredFieldsError(SchemaMappingError):
    """Raised when a mapping (even after user confirmation) is still
    missing one or more required canonical fields -- the pipeline cannot
    proceed without them."""

    def __init__(self, missing_fields: list[str]):
        self.missing_fields = missing_fields
        super().__init__(
            f"The mapping is missing required field(s): {', '.join(missing_fields)}. "
            f"Please map these before continuing."
        )