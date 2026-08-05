"""
Gemini API client wrapper for src.ai_assistant.

Uses google-genai (the current SDK) -- google-generativeai was
deprecated. Handles the actual API call, using an API key from an
environment variable (never hardcoded). Fails gracefully -- if the API
call fails, callers get a clear error, and the underlying structured
data (the actually trustworthy part) remains fully usable regardless.
"""

from __future__ import annotations

import os

from google import genai

MODEL_NAME = "gemini-3.5-flash-lite"


class AIAssistantError(Exception):
    """Raised when the Gemini API call fails for any reason."""


def _get_client() -> genai.Client:
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise AIAssistantError(
            "GEMINI_API_KEY environment variable is not set. "
            "Get a free key at https://aistudio.google.com/apikey"
        )
    return genai.Client(api_key=api_key)


def generate_text(prompt: str) -> str:
    """Sends a prompt to Gemini and returns the generated text.
    Raises AIAssistantError on any failure, with a clear message."""
    try:
        client = _get_client()
        response = client.models.generate_content(model=MODEL_NAME, contents=prompt)
        return response.text.strip()
    except AIAssistantError:
        raise
    except Exception as exc:
        raise AIAssistantError(f"Gemini API call failed: {exc}") from exc