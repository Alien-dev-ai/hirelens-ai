"""Tests for src/services/llm_service.py.

These tests never make real Groq API calls: the ``groq.Groq`` client
constructor is monkeypatched to return a stub client, so no network access
or real API key is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import groq
import httpx
import pytest

from src.services.llm_service import (
    DEFAULT_GROQ_MODEL,
    GroqLLMService,
    LLMAuthenticationError,
    LLMConnectionError,
    LLMInvalidModelError,
    LLMQuotaExceededError,
    LLMServiceError,
)


@pytest.fixture(autouse=True)
def _isolated_from_ambient_env(monkeypatch):
    """Isolate every test from whatever GROQ_MODEL a real local .env sets.

    ``load_dotenv()`` runs at import time and populates the process
    environment from the developer's actual .env file. Without this, a
    test that doesn't explicitly pin GROQ_MODEL could silently pass or
    fail depending on what's in that file rather than the code under
    test. Tests that want a specific GROQ_MODEL set it themselves via
    monkeypatch.setenv.
    """
    monkeypatch.delenv("GROQ_MODEL", raising=False)


def _make_status_error(error_cls, status_code: int, message: str) -> groq.APIStatusError:
    """Build a real Groq SDK status error for a given HTTP status code.

    Mirrors what the Groq SDK itself constructs from an HTTP response, so
    the code under test exercises the exact exception types/attributes it
    would see in production.
    """
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    response = httpx.Response(status_code, request=request, json={"error": {"message": message}})
    return error_cls(message, response=response, body=None)


class _FakeCompletions:
    """Stub for ``client.chat.completions`` that records calls and returns a canned response."""

    def __init__(self, response_text: str | None = "Generated response text.", error: Exception | None = None):
        self.response_text = response_text
        self.error = error
        self.last_call: dict | None = None

    def create(self, *, model: str, messages: list[dict]):
        self.last_call = {"model": model, "messages": messages}
        if self.error is not None:
            raise self.error
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content=self.response_text))]
        )


class _FakeChat:
    def __init__(self, completions: _FakeCompletions):
        self.completions = completions


class _FakeClient:
    """Stub for ``groq.Groq`` that exposes a ``.chat.completions`` attribute."""

    def __init__(self, completions: _FakeCompletions):
        self.chat = _FakeChat(completions)


def test_generate_text_success(monkeypatch):
    """generate_text should return the model's text on a successful call."""
    fake_completions = _FakeCompletions(response_text="HireLens Groq connection successful")
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")
    result = service.generate_text("Say hello", system_instruction="Be concise.")

    assert result == "HireLens Groq connection successful"
    assert fake_completions.last_call["model"] == DEFAULT_GROQ_MODEL
    assert fake_completions.last_call["messages"] == [
        {"role": "system", "content": "Be concise."},
        {"role": "user", "content": "Say hello"},
    ]


def test_generate_text_without_system_instruction(monkeypatch):
    """system_instruction is optional; omitting it should send only a user message."""
    fake_completions = _FakeCompletions(response_text="OK")
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")
    result = service.generate_text("Say hello")

    assert result == "OK"
    assert fake_completions.last_call["messages"] == [{"role": "user", "content": "Say hello"}]


def test_missing_api_key_raises_llm_service_error(monkeypatch):
    """Missing GROQ_API_KEY (and no explicit api_key) should raise a clear config error."""
    monkeypatch.delenv("GROQ_API_KEY", raising=False)

    with pytest.raises(LLMServiceError, match="GROQ_API_KEY"):
        GroqLLMService()


def test_rate_limit_error_raises_quota_exceeded(monkeypatch):
    """A 429 from Groq should be wrapped in LLMQuotaExceededError, not leaked."""
    api_error = _make_status_error(groq.RateLimitError, 429, "Rate limit reached")
    fake_completions = _FakeCompletions(error=api_error)
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")

    with pytest.raises(LLMQuotaExceededError) as exc_info:
        service.generate_text("Say hello")

    # The application-level exception should not simply be the raw SDK error.
    assert exc_info.value is not api_error
    assert exc_info.value.__cause__ is api_error


def test_authentication_error_raises_llm_authentication_error(monkeypatch):
    """A 401 from Groq should be wrapped in LLMAuthenticationError, not leaked."""
    api_error = _make_status_error(groq.AuthenticationError, 401, "Invalid API Key")
    fake_completions = _FakeCompletions(error=api_error)
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")

    with pytest.raises(LLMAuthenticationError) as exc_info:
        service.generate_text("Say hello")

    assert exc_info.value.__cause__ is api_error
    # The API key itself must never appear in the raised message.
    assert "fake-test-key" not in str(exc_info.value)


def test_not_found_error_raises_invalid_model_error(monkeypatch):
    """A 404 from Groq should be wrapped in LLMInvalidModelError, not leaked."""
    api_error = _make_status_error(groq.NotFoundError, 404, "Model not found")
    fake_completions = _FakeCompletions(error=api_error)
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")

    with pytest.raises(LLMInvalidModelError) as exc_info:
        service.generate_text("Say hello")

    assert exc_info.value.__cause__ is api_error


def test_bad_request_naming_model_raises_invalid_model_error(monkeypatch):
    """A 400 whose message names the model should also be treated as an invalid-model error."""
    api_error = _make_status_error(
        groq.BadRequestError, 400, "The model `bogus-model` does not exist"
    )
    fake_completions = _FakeCompletions(error=api_error)
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key", model="bogus-model")

    with pytest.raises(LLMInvalidModelError) as exc_info:
        service.generate_text("Say hello")

    assert exc_info.value.__cause__ is api_error


def test_other_bad_request_raises_generic_llm_service_error(monkeypatch):
    """A 400 unrelated to the model name should fall back to the generic error type."""
    api_error = _make_status_error(groq.BadRequestError, 400, "Malformed request body")
    fake_completions = _FakeCompletions(error=api_error)
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")

    with pytest.raises(LLMServiceError) as exc_info:
        service.generate_text("Say hello")

    assert not isinstance(exc_info.value, LLMInvalidModelError)
    assert exc_info.value.__cause__ is api_error


def test_connection_error_raises_llm_connection_error(monkeypatch):
    """A network/connection failure should be wrapped in LLMConnectionError, not leaked."""
    request = httpx.Request("POST", "https://api.groq.com/openai/v1/chat/completions")
    api_error = groq.APIConnectionError(request=request)
    fake_completions = _FakeCompletions(error=api_error)
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")

    with pytest.raises(LLMConnectionError) as exc_info:
        service.generate_text("Say hello")

    assert exc_info.value.__cause__ is api_error


def test_unexpected_error_raises_llm_service_error(monkeypatch):
    """A non-SDK exception during the call should still be wrapped, not leaked."""
    fake_completions = _FakeCompletions(error=RuntimeError("boom"))
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")

    with pytest.raises(LLMServiceError):
        service.generate_text("Say hello")


def test_empty_response_raises_llm_service_error(monkeypatch):
    """A response with no usable text should be treated as a failure, not a silent success."""
    fake_completions = _FakeCompletions(response_text="   ")
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")

    with pytest.raises(LLMServiceError):
        service.generate_text("Say hello")


def test_missing_choices_raises_llm_service_error(monkeypatch):
    """A malformed response with no choices should be treated as a failure, not crash."""
    fake_completions = _FakeCompletions()
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )
    # Override create() to return a response with an empty choices list.
    fake_completions.create = lambda **kwargs: SimpleNamespace(choices=[])

    service = GroqLLMService(api_key="fake-test-key")

    with pytest.raises(LLMServiceError):
        service.generate_text("Say hello")


def test_groq_model_env_var_overrides_default(monkeypatch):
    """Setting GROQ_MODEL should override DEFAULT_GROQ_MODEL."""
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    fake_completions = _FakeCompletions(response_text="custom model response")
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")

    assert service.model == "llama-3.1-8b-instant"

    service.generate_text("Say hello")
    assert fake_completions.last_call["model"] == "llama-3.1-8b-instant"


def test_explicit_model_argument_overrides_env_var(monkeypatch):
    """An explicit `model` constructor argument should take precedence over GROQ_MODEL."""
    monkeypatch.setenv("GROQ_MODEL", "llama-3.1-8b-instant")
    fake_completions = _FakeCompletions()
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key", model="gemma2-9b-it")

    assert service.model == "gemma2-9b-it"


def test_default_model_used_when_env_var_unset(monkeypatch):
    """With GROQ_MODEL unset, the service should fall back to DEFAULT_GROQ_MODEL."""
    monkeypatch.delenv("GROQ_MODEL", raising=False)
    fake_completions = _FakeCompletions()
    monkeypatch.setattr(
        "src.services.llm_service.groq.Groq",
        lambda api_key: _FakeClient(fake_completions),
    )

    service = GroqLLMService(api_key="fake-test-key")

    assert service.model == DEFAULT_GROQ_MODEL


def test_default_model_is_llama_3_3_70b_versatile():
    """The default model must be the project's chosen Groq model."""
    assert DEFAULT_GROQ_MODEL == "llama-3.3-70b-versatile"
