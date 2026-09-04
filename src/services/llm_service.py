"""Reusable Gemini LLM service layer for HireLens AI.

This module is the only place in the application that imports and communicates
with the Google Gemini SDK. Higher-level modules use GeminiLLMService instead
of talking directly to the Gemini SDK.

SECURITY NOTE:
The Gemini API key is read from the GEMINI_API_KEY environment variable and is
never printed or logged.
"""

from __future__ import annotations

import os

from dotenv import load_dotenv
from google import genai
from google.genai import errors as genai_errors
from google.genai import types

# Load variables from the local .env file.
load_dotenv()

# Default model used when GEMINI_MODEL is not configured.
DEFAULT_GEMINI_MODEL = "gemini-3.6-flash"


class LLMServiceError(Exception):
    """Raised when the LLM service cannot complete a request."""


class LLMQuotaExceededError(LLMServiceError):
    """Raised when Gemini reports a quota or rate-limit error.

    Covers HTTP 429 responses and Gemini's ``RESOURCE_EXHAUSTED`` status,
    e.g. free-tier quota exhaustion. Callers can catch this specifically to
    show a friendly "try again later" message instead of a generic failure.
    """


def _is_quota_or_rate_limit_error(exc: Exception) -> bool:
    """Return True if `exc` looks like a Gemini quota/rate-limit error.

    Checks the structured fields ``google.genai.errors.APIError`` exposes
    (``code`` / ``status``) first, then falls back to a substring check on
    the stringified error so unexpected shapes are still detected.
    """
    if getattr(exc, "code", None) == 429:
        return True

    status = getattr(exc, "status", None) or ""
    if isinstance(status, str) and "RESOURCE_EXHAUSTED" in status.upper():
        return True

    return "RESOURCE_EXHAUSTED" in str(exc).upper()


class GeminiLLMService:
    """Reusable service for generating text with Google Gemini."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the Gemini service."""

        resolved_api_key = api_key or os.getenv("GEMINI_API_KEY")

        if not resolved_api_key:
            raise LLMServiceError(
                "GEMINI_API_KEY is not set. Add it to your environment "
                "or local .env file."
            )

        self._model = (
            model
            or os.getenv("GEMINI_MODEL")
            or DEFAULT_GEMINI_MODEL
        )

        self._client = genai.Client(api_key=resolved_api_key)

    @property
    def model(self) -> str:
        """Return the configured Gemini model name."""
        return self._model

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate text using the configured Gemini model.

        Raises:
            LLMServiceError: If Gemini cannot complete the request.
        """

        config = (
            types.GenerateContentConfig(
                system_instruction=system_instruction
            )
            if system_instruction is not None
            else None
        )

        try:
            response = self._client.models.generate_content(
                model=self._model,
                contents=prompt,
                config=config,
            )

        except genai_errors.APIError as exc:
            if _is_quota_or_rate_limit_error(exc):
                # A distinct, catchable error type so callers can show a
                # friendly "try again later" message instead of a generic
                # failure. The original exception is preserved as the
                # cause for developers who need the full detail.
                raise LLMQuotaExceededError(
                    f"Gemini API quota or rate limit reached "
                    f"(model='{self._model}'): {exc}"
                ) from exc

            # Include the real API error so we can diagnose model,
            # configuration, or other request problems.
            raise LLMServiceError(
                f"Gemini API request failed "
                f"(model='{self._model}'): {exc}"
            ) from exc

        except Exception as exc:
            # Include unexpected errors during development/debugging.
            raise LLMServiceError(
                f"Unexpected error while calling Gemini "
                f"(model='{self._model}'): {exc}"
            ) from exc

        text = getattr(response, "text", None)

        if not text or not text.strip():
            raise LLMServiceError(
                f"Gemini returned no usable text "
                f"(model='{self._model}')."
            )

        return text.strip()