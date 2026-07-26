"""
MappingResult -- the object src.schema_mapping returns after suggesting
(and later, after the user confirms) a column mapping.

Two distinct stages live in this one object's lifecycle:
1. SUGGESTED: produced by matcher.suggest_mapping(), before any human review.
2. CONFIRMED: produced after the user reviews/corrects it (Streamlit step,
   later in this same phase) -- at that point `mapping` is considered final
   and safe for apply_mapping() to actually rename DataFrame columns.

Keeping both stages in the same model (with an `is_confirmed` flag) means
downstream code can check that flag before trusting the mapping, rather
than us needing two separate classes that drift out of sync.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class MappingResult(BaseModel):
    mapping: dict[str, str] = Field(default_factory=dict)
    # canonical_field -> raw_column, only for confident/confirmed matches

    confidence_scores: dict[str, float] = Field(default_factory=dict)
    # canonical_field -> similarity score (0-100), matches keys in `mapping`

    ambiguous: dict[str, list[tuple[str, float]]] = Field(default_factory=dict)
    # canonical_field -> [(candidate_raw_column, score), ...], unresolved

    unmapped_columns: list[str] = Field(default_factory=list)
    # raw columns that matched nothing, preserved (not dropped) for later use

    is_confirmed: bool = False
    # False = this is still just a suggestion; True = user has reviewed/approved it

    def missing_required_fields(self, required_fields: list[str]) -> list[str]:
        """Which required canonical fields have no mapping at all yet --
        used to block progress until the user resolves them."""
        return [field for field in required_fields if field not in self.mapping]

    def has_unresolved_ambiguity(self) -> bool:
        return len(self.ambiguous) > 0

    def summary(self) -> str:
        return (
            f"Mapped {len(self.mapping)} field(s), "
            f"{len(self.ambiguous)} ambiguous, "
            f"{len(self.unmapped_columns)} unmapped column(s), "
            f"confirmed={self.is_confirmed}"
        )