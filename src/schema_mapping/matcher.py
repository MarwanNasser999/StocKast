"""
Matcher for schema_mapping.

Takes the raw column names from an uploaded file and suggests which
canonical field each one most likely represents, using fuzzy string
matching (rapidfuzz) against the alias lists defined in aliases.py.

This produces a SUGGESTION only -- nothing here renames columns or
touches the actual DataFrame. That happens later, only after the user
confirms the mapping (Streamlit step, later in Phase 2).
"""

from __future__ import annotations

import re

from rapidfuzz import fuzz

from src.common.canonical_schema import ALL_FIELDS
from src.schema_mapping.aliases import get_aliases

# Two raw columns competing for the same canonical field are considered
# "ambiguous" if their scores are this close to each other.
AMBIGUITY_MARGIN = 5.0


def normalize_column_name(name: str) -> str:
    """Normalize a raw column name so 'Invoice Date', 'invoice_date', and
    'InvoiceDate' all compare fairly against our lowercase, underscored
    alias lists.
    """
    name = name.strip().lower()
    name = re.sub(r"[\s\-]+", "_", name)   # spaces/dashes -> underscore
    name = re.sub(r"[^\w]", "", name)      # drop anything not alphanumeric/underscore
    return name


def score_column_against_field(raw_column: str, canonical_field: str) -> float:
    """Best similarity score (0-100) between one raw column and one
    canonical field, checked against every known alias for that field.
    """
    normalized_raw = normalize_column_name(raw_column)
    aliases = get_aliases(canonical_field)
    scores = [fuzz.ratio(normalized_raw, alias) for alias in aliases]
    return max(scores)


def build_score_matrix(raw_columns: list[str]) -> dict[tuple[str, str], float]:
    """Score every (canonical_field, raw_column) pair. Small enough
    (9 fields x typically <20 columns) that brute-force is fine -- no
    need for anything cleverer at this scale.
    """
    matrix: dict[tuple[str, str], float] = {}
    for field in ALL_FIELDS:
        for column in raw_columns:
            matrix[(field, column)] = score_column_against_field(column, field)
    return matrix


def suggest_mapping(raw_columns: list[str], threshold: float = 75.0) -> dict:
    """
    Suggest a canonical mapping for a raw DataFrame's columns.

    Returns a dict with:
    - "mapping": {canonical_field: raw_column} for confident, unambiguous matches
    - "confidence_scores": {canonical_field: score} for each mapped field
    - "ambiguous": {canonical_field: [(raw_column, score), ...]} where two+
      raw columns were close competitors and no automatic choice was made
    - "unmapped_columns": raw columns not used in any mapping

    Greedy strategy: repeatedly pick the single best-scoring (field, column)
    pair across the whole matrix, assign it if it clears the threshold, then
    remove both the field and the column from further consideration. This
    prevents one raw column from being claimed by two canonical fields.
    """
    matrix = build_score_matrix(raw_columns)

    remaining_fields = set(ALL_FIELDS)
    remaining_columns = set(raw_columns)

    mapping: dict[str, str] = {}
    confidence_scores: dict[str, float] = {}
    ambiguous: dict[str, list[tuple[str, float]]] = {}

    while remaining_fields and remaining_columns:
        # find the best-scoring pair still available
        candidate_pairs = [
            (field, column, score)
            for (field, column), score in matrix.items()
            if field in remaining_fields and column in remaining_columns
        ]
        if not candidate_pairs:
            break

        best_field, best_column, best_score = max(candidate_pairs, key=lambda p: p[2])

        if best_score < threshold:
            break  # nothing left is confident enough to auto-suggest

        # check for ambiguity: other columns scoring close to the winner,
        # for this same field
        competitors = [
            (column, score) for (field, column, score) in candidate_pairs
            if field == best_field and column != best_column
            and (best_score - score) <= AMBIGUITY_MARGIN
        ]

        if competitors:
            ambiguous[best_field] = [(best_column, best_score)] + competitors
        else:
            mapping[best_field] = best_column
            confidence_scores[best_field] = best_score

        remaining_fields.discard(best_field)
        remaining_columns.discard(best_column)

    unmapped_columns = list(remaining_columns)

    return {
        "mapping": mapping,
        "confidence_scores": confidence_scores,
        "ambiguous": ambiguous,
        "unmapped_columns": unmapped_columns,
    }