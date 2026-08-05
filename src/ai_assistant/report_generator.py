"""
Report generation for src.ai_assistant.

Fills in the prompt templates with real structured data from
recommendation_engine's output, then calls the Gemini client. If the
API call fails, returns the underlying structured facts alongside a
clear error, so the caller can still show something useful.
"""

from __future__ import annotations

from src.ai_assistant.client import generate_text, AIAssistantError
from src.ai_assistant.prompts import PRODUCT_REPORT_PROMPT, EXECUTIVE_SUMMARY_PROMPT


def _describe_trend(risk_row: dict) -> str:
    # risk_row doesn't carry raw trend directly -- this is a simple,
    # honest placeholder description; a fuller version could pull the
    # actual forecast slope if passed in separately.
    return "see forecast for details"


def generate_product_report(display_name: str, risk_row: dict, recommendation: dict | None) -> dict:
    """
    Generates a per-product narrative report.
    Returns {"text": ..., "error": None} on success, or
    {"text": None, "error": "..."} on failure -- caller can still
    display the raw risk_row/recommendation even if narration fails.
    """
    prompt = PRODUCT_REPORT_PROMPT.format(
        display_name=display_name,
        days_of_inventory=risk_row.get("days_of_inventory", "unknown"),
        risk_label=risk_row.get("risk_label", "unknown"),
        risk_score=risk_row.get("risk_score", "unknown"),
        trend_description=_describe_trend(risk_row),
        action=recommendation["action"] if recommendation else "none",
        reasoning=recommendation["reasoning"] if recommendation else "No specific action recommended at this time.",
    )

    try:
        text = generate_text(prompt)
        return {"text": text, "error": None}
    except AIAssistantError as exc:
        return {"text": None, "error": str(exc)}


def generate_executive_summary(recommendations: list[dict], total_products: int,
                                high_risk_count: int, dead_stock_count: int) -> dict:
    """Generates a whole-dataset executive summary report."""
    top_three = recommendations[:3]
    formatted_top = "\n".join(
        f"{i+1}. {r['display_name']}: {r['action']} ({r['priority']} priority) -- {r['reasoning']}"
        for i, r in enumerate(top_three)
    ) or "No high-priority recommendations at this time."

    prompt = EXECUTIVE_SUMMARY_PROMPT.format(
        total_products=total_products,
        high_risk_count=high_risk_count,
        dead_stock_count=dead_stock_count,
        total_recommendations=len(recommendations),
        top_recommendations=formatted_top,
    )

    try:
        text = generate_text(prompt)
        return {"text": text, "error": None}
    except AIAssistantError as exc:
        return {"text": None, "error": str(exc)}