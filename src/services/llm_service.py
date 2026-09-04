"""Reusable Groq LLM service layer for HireLens AI.

This module is the only place in the application that imports and communicates
with the Groq SDK. Higher-level modules use GroqLLMService instead of talking
directly to the Groq SDK.

SECURITY NOTE:
The Groq API key is read from the GROQ_API_KEY environment variable and is
never printed or logged.
"""

from __future__ import annotations

import os

import groq
from dotenv import load_dotenv

# Load variables from the local .env file.
load_dotenv()

# Default model used when GROQ_MODEL is not configured.
DEFAULT_GROQ_MODEL = "llama-3.3-70b-versatile"


class LLMServiceError(Exception):
    """Raised when the LLM service cannot complete a request."""


class LLMQuotaExceededError(LLMServiceError):
    """Raised when the LLM provider reports a quota or rate-limit error.

    Covers HTTP 429 responses, e.g. free-tier quota exhaustion. Callers can
    catch this specifically to show a friendly "try again later" message
    instead of a generic failure.
    """


class LLMAuthenticationError(LLMServiceError):
    """Raised when the LLM provider rejects the configured API key.

    Covers HTTP 401 responses (missing, invalid, or revoked API key).
    """


class LLMInvalidModelError(LLMServiceError):
    """Raised when the configured model name is rejected by the provider.

    Covers HTTP 404 ("model not found") responses and HTTP 400 responses
    whose message indicates the model name itself is the problem.
    """


class LLMConnectionError(LLMServiceError):
    """Raised when the provider's API could not be reached at all.

    Covers network/connectivity failures and request timeouts, as opposed
    to an error response returned by the API itself.
    """


def _is_invalid_model_error(exc: groq.APIStatusError) -> bool:
    """Return True if a 400 Bad Request looks like an invalid-model error.

    Groq (and OpenAI-compatible APIs generally) report an unrecognized
    model name as a 400 whose message names the model, rather than a
    dedicated status code. NotFoundError (404) is handled separately by
    the caller and doesn't need this heuristic.
    """
    message = str(exc).lower()
    return "model" in message and (
        "not found" in message or "does not exist" in message or "invalid" in message
    )


class GroqLLMService:
    """Reusable service for generating text with Groq."""

    def __init__(
        self,
        api_key: str | None = None,
        model: str | None = None,
    ) -> None:
        """Initialize the Groq service."""

        resolved_api_key = api_key or os.getenv("GROQ_API_KEY")

        if not resolved_api_key:
            raise LLMServiceError(
                "GROQ_API_KEY is not set. Add it to your environment "
                "or local .env file."
            )

        self._model = (
            model
            or os.getenv("GROQ_MODEL")
            or DEFAULT_GROQ_MODEL
        )

        self._client = groq.Groq(api_key=resolved_api_key)

    @property
    def model(self) -> str:
        """Return the configured Groq model name."""
        return self._model

    def generate_text(
        self,
        prompt: str,
        system_instruction: str | None = None,
    ) -> str:
        """Generate text using the configured Groq model.

        Raises:
            LLMServiceError: If Groq cannot complete the request.
        """

        messages = []

        if system_instruction:
            messages.append({
                "role": "system",
                "content": system_instruction,
            })

        messages.append({
            "role": "user",
            "content": prompt,
        })

        try:
            response = self._client.chat.completions.create(
                model=self._model,
                messages=messages,
            )

        except groq.APIStatusError as exc:
            # An error response (4xx/5xx) from the Groq API itself.
            if isinstance(exc, groq.AuthenticationError):
                raise LLMAuthenticationError(
                    f"Groq API authentication failed (model='{self._model}'). "
                    f"Check that GROQ_API_KEY is set correctly: {exc}"
                ) from exc

            if isinstance(exc, groq.RateLimitError):
                raise LLMQuotaExceededError(
                    f"Groq API quota or rate limit reached "
                    f"(model='{self._model}'): {exc}"
                ) from exc

            if isinstance(exc, groq.NotFoundError) or _is_invalid_model_error(exc):
                raise LLMInvalidModelError(
                    f"Groq model '{self._model}' was rejected by the API. "
                    f"Check GROQ_MODEL: {exc}"
                ) from exc

            # Other 4xx/5xx errors (permission denied, conflict, malformed
            # request, server error, ...) — include the real API error so
            # we can diagnose configuration or other request problems.
            raise LLMServiceError(
                f"Groq API request failed "
                f"(model='{self._model}'): {exc}"
            ) from exc

        except groq.APIConnectionError as exc:
            # The request never reached Groq, or no response came back
            # (network failure, DNS failure, timeout).
            raise LLMConnectionError(
                f"Could not reach the Groq API "
                f"(model='{self._model}'): {exc}"
            ) from exc

        except groq.GroqError as exc:
            # Any other error raised by the Groq SDK itself.
            raise LLMServiceError(
                f"Groq API request failed "
                f"(model='{self._model}'): {exc}"
            ) from exc

        except Exception as exc:
            # Include unexpected errors during development/debugging.
            raise LLMServiceError(
                f"Unexpected error while calling Groq "
                f"(model='{self._model}'): {exc}"
            ) from exc

        text = self._extract_text(response)

        if not text or not text.strip():
            raise LLMServiceError(
                f"Groq returned no usable text "
                f"(model='{self._model}')."
            )

        return text.strip()

    @staticmethod
    def _extract_text(response: object) -> str | None:
        """Safely pull the generated text out of a Groq chat completion.

        Defensive against an empty/malformed ``choices`` list rather than
        assuming index 0 always exists.
        """
        choices = getattr(response, "choices", None)
        if not choices:
            return None

        message = getattr(choices[0], "message", None)
        if message is None:
            return None

        return getattr(message, "content", None)
