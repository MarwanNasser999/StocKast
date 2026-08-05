"""
Unit tests for src.ai_assistant.

Uses unittest.mock to replace the real Gemini API call with a fake,
predictable response -- so tests run fast, don't need a real API key,
and don't burn free-tier quota on every test run.
"""

from unittest.mock import patch

import pytest

from src.ai_assistant.client import generate_text, AIAssistantError
from src.ai_assistant.report_generator import generate_product_report, generate_executive_summary


def test_generate_text_raises_clear_error_without_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(AIAssistantError, match="GEMINI_API_KEY"):
        generate_text("test prompt")


@patch("src.ai_assistant.client._get_client")
def test_generate_text_returns_response_text(mock_get_client):
    mock_response = type("MockResponse", (), {"text": "  Generated report text.  "})()
    mock_get_client.return_value.models.generate_content.return_value = mock_response

    result = generate_text("test prompt")

    assert result == "Generated report text."  # confirms .strip() is applied


@patch("src.ai_assistant.report_generator.generate_text")
def test_generate_product_report_fills_template_correctly(mock_generate_text):
    mock_generate_text.return_value = "Mocked product summary."

    risk_row = {"days_of_inventory": 3.2, "risk_label": "high", "risk_score": 0.82}
    recommendation = {"action": "increase_inventory", "reasoning": "Stock below reorder point."}

    result = generate_product_report("Wireless Mouse", risk_row, recommendation)

    assert result["error"] is None
    assert result["text"] == "Mocked product summary."

    sent_prompt = mock_generate_text.call_args[0][0]
    assert "Wireless Mouse" in sent_prompt
    assert "high" in sent_prompt
    assert "3.2" in sent_prompt


@patch("src.ai_assistant.report_generator.generate_text")
def test_generate_product_report_handles_api_failure_gracefully(mock_generate_text):
    mock_generate_text.side_effect = AIAssistantError("API is down")

    risk_row = {"days_of_inventory": 3.2, "risk_label": "high", "risk_score": 0.82}
    result = generate_product_report("Wireless Mouse", risk_row, None)

    assert result["text"] is None
    assert "API is down" in result["error"]


@patch("src.ai_assistant.report_generator.generate_text")
def test_generate_executive_summary_includes_top_recommendations(mock_generate_text):
    mock_generate_text.return_value = "Mocked executive summary."

    recommendations = [
        {"display_name": "Wireless Mouse", "action": "increase_inventory", "priority": "high",
         "reasoning": "Stock below reorder point."},
        {"display_name": "USB Cable", "action": "discount", "priority": "low",
         "reasoning": "No sales in 60 days."},
    ]

    result = generate_executive_summary(recommendations, total_products=50,
                                          high_risk_count=5, dead_stock_count=8)

    assert result["error"] is None
    sent_prompt = mock_generate_text.call_args[0][0]
    assert "Wireless Mouse" in sent_prompt
    assert "USB Cable" in sent_prompt
    assert "50" in sent_prompt


@patch("src.ai_assistant.report_generator.generate_text")
def test_generate_executive_summary_handles_no_recommendations(mock_generate_text):
    mock_generate_text.return_value = "All clear."

    result = generate_executive_summary([], total_products=10, high_risk_count=0, dead_stock_count=0)

    assert result["error"] is None
    sent_prompt = mock_generate_text.call_args[0][0]
    assert "No high-priority recommendations" in sent_prompt