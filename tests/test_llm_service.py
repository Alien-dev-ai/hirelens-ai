"""Tests for src/services/llm_service.py.

These tests never make real Gemini API calls: the ``google.genai.Client``
constructor is monkeypatched to return a stub client, so no network access
or real API key is required.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from google.genai import errors as genai_errors

from src.services.llm_service import (
    DEFAULT_GEMINI_MODEL,
    GeminiLLMService,
    LLMServiceError,
)


class _FakeModels:
    """Stub for ``client.models`` that records calls and returns a canned response."""

    def __init__(self, response_text: str | None = "Generated response text.", error: Exception | None = None):
        self.response_text = response_text
        self.error = error
        self.last_call: dict | None = None

    def generate_content(self, *, model: str, contents: str, config=None):
        self.last_call = {"model": model, "contents": contents, "config": config}
        if self.error is not None:
            raise self.error
        return SimpleNamespace(text=self.response_text)


class _FakeClient:
    """Stub for ``genai.Client`` that exposes a ``.models`` attribute."""

    def __init__(self, models: _FakeModels):
        self.models = models


@pytest.fixture(autouse=True)
def _no_dotenv_side_effects(monkeypatch):
    """Prevent load_dotenv (called at import time) from pulling in a real key."""
    # The module already called load_dotenv() at import time; per-test we
    # explicitly control GEMINI_API_KEY / GEMINI_MODEL via monkeypatch below,
    # so no further action is needed here. This fixture exists for clarity
    # and as a hook if stricter isolation is ever needed.
    yield


def test_generate_text_success(monkeypatch):
    """generate_text should return the model's text on a successful call."""
    fake_models = _FakeModels(response_text="HireLens Gemini connection successful")
    monkeypatch.setattr(
        "src.services.llm_service.genai.Client",
        lambda api_key: _FakeClient(fake_models),
    )

    service = GeminiLLMService(api_key="fake-test-key")
    result = service.generate_text("Say hello", system_instruction="Be concise.")

    assert result == "HireLens Gemini connection successful"
    assert fake_models.last_call["model"] == DEFAULT_GEMINI_MODEL
    assert fake_models.last_call["contents"] == "Say hello"
    assert fake_models.last_call["config"] is not None


def test_generate_text_without_system_instruction(monkeypatch):
    """system_instruction is optional; omitting it should still work and pass config=None."""
    fake_models = _FakeModels(response_text="OK")
    monkeypatch.setattr(
        "src.services.llm_service.genai.Client",
        lambda api_key: _FakeClient(fake_models),
    )

    service = GeminiLLMService(api_key="fake-test-key")
    result = service.generate_text("Say hello")

    assert result == "OK"
    assert fake_models.last_call["config"] is None


def test_missing_api_key_raises_llm_service_error(monkeypatch):
    """Missing GEMINI_API_KEY (and no explicit api_key) should raise a clear config error."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(LLMServiceError, match="GEMINI_API_KEY"):
        GeminiLLMService()


def test_api_failure_raises_llm_service_error(monkeypatch):
    """An underlying Gemini API error should be wrapped in LLMServiceError, not leaked."""
    api_error = genai_errors.ClientError(429, {"error": {"message": "rate limited"}})
    fake_models = _FakeModels(error=api_error)
    monkeypatch.setattr(
        "src.services.llm_service.genai.Client",
        lambda api_key: _FakeClient(fake_models),
    )

    service = GeminiLLMService(api_key="fake-test-key")

    with pytest.raises(LLMServiceError) as exc_info:
        service.generate_text("Say hello")

    # The application-level exception should not simply be the raw SDK error.
    assert exc_info.value is not api_error
    assert exc_info.value.__cause__ is api_error


def test_empty_response_raises_llm_service_error(monkeypatch):
    """A response with no usable text should be treated as a failure, not a silent success."""
    fake_models = _FakeModels(response_text="   ")
    monkeypatch.setattr(
        "src.services.llm_service.genai.Client",
        lambda api_key: _FakeClient(fake_models),
    )

    service = GeminiLLMService(api_key="fake-test-key")

    with pytest.raises(LLMServiceError):
        service.generate_text("Say hello")


def test_gemini_model_env_var_overrides_default(monkeypatch):
    """Setting GEMINI_MODEL should override DEFAULT_GEMINI_MODEL."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-custom-test-model")
    fake_models = _FakeModels(response_text="custom model response")
    monkeypatch.setattr(
        "src.services.llm_service.genai.Client",
        lambda api_key: _FakeClient(fake_models),
    )

    service = GeminiLLMService(api_key="fake-test-key")

    assert service.model == "gemini-custom-test-model"

    service.generate_text("Say hello")
    assert fake_models.last_call["model"] == "gemini-custom-test-model"


def test_explicit_model_argument_overrides_env_var(monkeypatch):
    """An explicit `model` constructor argument should take precedence over GEMINI_MODEL."""
    monkeypatch.setenv("GEMINI_MODEL", "gemini-env-model")
    fake_models = _FakeModels()
    monkeypatch.setattr(
        "src.services.llm_service.genai.Client",
        lambda api_key: _FakeClient(fake_models),
    )

    service = GeminiLLMService(api_key="fake-test-key", model="gemini-explicit-model")

    assert service.model == "gemini-explicit-model"


def test_default_model_used_when_env_var_unset(monkeypatch):
    """With GEMINI_MODEL unset, the service should fall back to DEFAULT_GEMINI_MODEL."""
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    fake_models = _FakeModels()
    monkeypatch.setattr(
        "src.services.llm_service.genai.Client",
        lambda api_key: _FakeClient(fake_models),
    )

    service = GeminiLLMService(api_key="fake-test-key")

    assert service.model == DEFAULT_GEMINI_MODEL
