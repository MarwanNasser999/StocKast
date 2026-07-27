"""
ValidationReport -- the structured result of running all data_validation
checks against a canonical DataFrame.

Design: validation only REPORTS on the data, never modifies it. Every
check appends either an error or a warning here; the DataFrame passed in
comes back completely untouched. Deciding what to DO about any issue
(drop rows, cap outliers, impute values) is data_cleaning's job, not this
module's.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class Severity(str, Enum):
    ERROR = "error"      # blocks the pipeline from proceeding
    WARNING = "warning"  # informational, pipeline can proceed


class ValidationIssue(BaseModel):
    check_name: str        # e.g. "date_parseability", "quantity_outliers"
    severity: Severity
    message: str            # human-readable description
    affected_field: str | None = None   # which canonical field this concerns, if any
    affected_row_count: int | None = None  # how many rows this issue touches, if applicable


class ValidationReport(BaseModel):
    issues: list[ValidationIssue] = Field(default_factory=list)

    def add_error(self, check_name: str, message: str, affected_field: str | None = None,
                  affected_row_count: int | None = None) -> None:
        self.issues.append(ValidationIssue(
            check_name=check_name, severity=Severity.ERROR, message=message,
            affected_field=affected_field, affected_row_count=affected_row_count,
        ))

    def add_warning(self, check_name: str, message: str, affected_field: str | None = None,
                     affected_row_count: int | None = None) -> None:
        self.issues.append(ValidationIssue(
            check_name=check_name, severity=Severity.WARNING, message=message,
            affected_field=affected_field, affected_row_count=affected_row_count,
        ))

    @property
    def errors(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.ERROR]

    @property
    def warnings(self) -> list[ValidationIssue]:
        return [i for i in self.issues if i.severity == Severity.WARNING]

    @property
    def is_valid(self) -> bool:
        """True if there are no blocking errors. Warnings don't affect this --
        the pipeline can proceed with warnings, just not with errors."""
        return len(self.errors) == 0

    def summary(self) -> str:
        return (
            f"Validation: {'PASSED' if self.is_valid else 'FAILED'} "
            f"({len(self.errors)} error(s), {len(self.warnings)} warning(s))"
        )