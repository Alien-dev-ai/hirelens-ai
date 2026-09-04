"""Tests for src/services/candidate_analyzer.py.

These tests never make real Groq API calls: a fake LLM service is
injected into ``CandidateAnalyzer`` in place of ``GroqLLMService``.
"""

from __future__ import annotations

import json

import pytest

from src.models.schemas import CandidateProfile
from src.services.candidate_analyzer import CandidateAnalyzer, CandidateAnalyzerError
from src.services.llm_service import LLMServiceError

VALID_PROFILE_JSON = {
    "full_name": "Jordan Rivera",
    "email": "jordan.rivera@example.com",
    "phone": "+1-555-0100",
    "location": "Austin, TX",
    "summary": "Backend engineer.",
    "skills": ["Python", "SQL"],
    "experiences": [
        {
            "role": "Backend Engineer",
            "organization": "Acme Corp",
            "start_date": "2020-01",
            "end_date": "2023-06",
            "description": "Built internal APIs.",
        }
    ],
    "education": ["B.S. Computer Science, University of Texas"],
    "projects": [
        {
            "name": "Internal API Gateway",
            "description": "A gateway service.",
            "technologies": ["Python", "FastAPI"],
            "url": None,
        }
    ],
    "certifications": ["AWS Certified Developer"],
    "links": ["https://github.com/jrivera"],
}

SAMPLE_CV_TEXT = (
    "Jordan Rivera\n"
    "jordan.rivera@example.com | Austin, TX\n\n"
    "Backend Engineer at Acme Corp (2020-01 to 2023-06)\n"
    "Built internal APIs.\n\n"
    "Skills: Python, SQL\n"
    "Education: B.S. Computer Science, University of Texas\n"
    "Certifications: AWS Certified Developer\n"
    "Links: https://github.com/jrivera\n"
)


class _FakeLLMService:
    """Stub for GroqLLMService that records the last call and returns a canned response."""

    def __init__(self, response_text: str | None = None, error: Exception | None = None):
        self.response_text = response_text
        self.error = error
        self.last_prompt: str | None = None
        self.last_system_instruction: str | None = None

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        self.last_prompt = prompt
        self.last_system_instruction = system_instruction
        if self.error is not None:
            raise self.error
        return self.response_text


def test_valid_cv_text_produces_valid_candidate_profile():
    """A well-formed JSON response should be validated into a CandidateProfile."""
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_PROFILE_JSON))
    analyzer = CandidateAnalyzer(fake_llm)

    profile = analyzer.analyze(SAMPLE_CV_TEXT)

    assert isinstance(profile, CandidateProfile)
    assert profile.full_name == "Jordan Rivera"
    assert profile.email == "jordan.rivera@example.com"
    assert profile.skills == ["Python", "SQL"]
    assert len(profile.experiences) == 1
    assert profile.experiences[0].organization == "Acme Corp"
    assert len(profile.projects) == 1
    assert profile.certifications == ["AWS Certified Developer"]


@pytest.mark.parametrize("empty_text", ["", "   ", "\n\n\t  "])
def test_empty_cv_text_raises_meaningful_error(empty_text: str):
    """Empty or whitespace-only CV text should raise CandidateAnalyzerError, not call the LLM."""
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_PROFILE_JSON))
    analyzer = CandidateAnalyzer(fake_llm)

    with pytest.raises(CandidateAnalyzerError, match="empty"):
        analyzer.analyze(empty_text)

    assert fake_llm.last_prompt is None  # LLM should never have been called


def test_markdown_fenced_json_is_handled():
    """A response wrapped in a ```json ... ``` code fence should still parse correctly."""
    fenced_response = "```json\n" + json.dumps(VALID_PROFILE_JSON) + "\n```"
    fake_llm = _FakeLLMService(response_text=fenced_response)
    analyzer = CandidateAnalyzer(fake_llm)

    profile = analyzer.analyze(SAMPLE_CV_TEXT)

    assert profile.full_name == "Jordan Rivera"


def test_generic_code_fence_without_json_label_is_handled():
    """A plain ``` ... ``` fence (no 'json' label) should also be stripped correctly."""
    fenced_response = "```\n" + json.dumps(VALID_PROFILE_JSON) + "\n```"
    fake_llm = _FakeLLMService(response_text=fenced_response)
    analyzer = CandidateAnalyzer(fake_llm)

    profile = analyzer.analyze(SAMPLE_CV_TEXT)

    assert profile.email == "jordan.rivera@example.com"


def test_invalid_json_raises_meaningful_error():
    """A non-JSON response should raise a clear CandidateAnalyzerError."""
    fake_llm = _FakeLLMService(response_text="this is not JSON at all {{{")
    analyzer = CandidateAnalyzer(fake_llm)

    with pytest.raises(CandidateAnalyzerError, match="JSON"):
        analyzer.analyze(SAMPLE_CV_TEXT)


def test_empty_llm_response_raises_meaningful_error():
    """An empty string response from the LLM should raise a clear CandidateAnalyzerError."""
    fake_llm = _FakeLLMService(response_text="   ")
    analyzer = CandidateAnalyzer(fake_llm)

    with pytest.raises(CandidateAnalyzerError, match="empty"):
        analyzer.analyze(SAMPLE_CV_TEXT)


def test_llm_service_failure_is_handled_cleanly():
    """An LLMServiceError from the underlying service should be wrapped, not leaked."""
    original_error = LLMServiceError("Groq API request failed.")
    fake_llm = _FakeLLMService(error=original_error)
    analyzer = CandidateAnalyzer(fake_llm)

    with pytest.raises(CandidateAnalyzerError) as exc_info:
        analyzer.analyze(SAMPLE_CV_TEXT)

    assert exc_info.value.__cause__ is original_error


def test_invalid_candidate_profile_structure_raises_meaningful_error():
    """JSON that doesn't match CandidateProfile's schema should raise CandidateAnalyzerError."""
    # `skills` should be a list of strings, not a single string.
    bad_payload = {**VALID_PROFILE_JSON, "skills": "Python, SQL"}
    fake_llm = _FakeLLMService(response_text=json.dumps(bad_payload))
    analyzer = CandidateAnalyzer(fake_llm)

    with pytest.raises(CandidateAnalyzerError, match="structure"):
        analyzer.analyze(SAMPLE_CV_TEXT)


def test_non_object_json_raises_meaningful_error():
    """A JSON response that isn't an object (e.g. a JSON array) should raise a clear error."""
    fake_llm = _FakeLLMService(response_text=json.dumps(["not", "an", "object"]))
    analyzer = CandidateAnalyzer(fake_llm)

    with pytest.raises(CandidateAnalyzerError):
        analyzer.analyze(SAMPLE_CV_TEXT)


def test_prompt_instructs_llm_not_to_invent_candidate_information():
    """The prompt/system instruction sent to the LLM must contain explicit grounding rules."""
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_PROFILE_JSON))
    analyzer = CandidateAnalyzer(fake_llm)

    analyzer.analyze(SAMPLE_CV_TEXT)

    combined_instructions = (
        (fake_llm.last_system_instruction or "") + "\n" + (fake_llm.last_prompt or "")
    ).lower()

    assert "only explicitly supported" in combined_instructions
    assert "do not infer" in combined_instructions
    assert "do not fabricate" in combined_instructions
    # Confirm the CV text itself was actually included in what was sent.
    assert "jordan rivera" in combined_instructions


def test_valid_cv_text_with_minimal_profile_json():
    """A minimal JSON response (only required-ish info) should still validate successfully."""
    minimal_payload = {
        "full_name": "Sam Lee",
        "skills": ["Python"],
    }
    fake_llm = _FakeLLMService(response_text=json.dumps(minimal_payload))
    analyzer = CandidateAnalyzer(fake_llm)

    profile = analyzer.analyze("Sam Lee\nSkills: Python")

    assert profile.full_name == "Sam Lee"
    assert profile.skills == ["Python"]
    assert profile.email is None
    assert profile.experiences == []
