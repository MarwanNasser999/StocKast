"""
CleaningReport -- the "receipt" of every action data_cleaning took,
following the same pattern as ValidationReport: structured, never
silent, so the user can see exactly what changed and why.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class CleaningAction(BaseModel):
    check_name: str          # which validation error this action resolves
    action: str                # e.g. "dropped_rows", "coerced_to_null"
    affected_field: str | None = None
    affected_row_count: int = 0
    message: str


class CleaningReport(BaseModel):
    actions: list[CleaningAction] = Field(default_factory=list)
    rows_before: int = 0
    rows_after: int = 0

    def add_action(self, check_name: str, action: str, message: str,
                   affected_field: str | None = None, affected_row_count: int = 0) -> None:
        self.actions.append(CleaningAction(
            check_name=check_name, action=action, message=message,
            affected_field=affected_field, affected_row_count=affected_row_count,
        ))

    def summary(self) -> str:
        return (
            f"Cleaning: {self.rows_before} rows -> {self.rows_after} rows "
            f"({len(self.actions)} action(s) taken)"
        )