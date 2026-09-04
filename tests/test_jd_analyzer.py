"""Tests for src/services/jd_analyzer.py.

These tests never make real Gemini API calls: a fake LLM service is
injected into ``JobDescriptionAnalyzer`` in place of ``GeminiLLMService``.
"""

from __future__ import annotations

import json

import pytest

from src.models.schemas import JobRequirements
from src.services.jd_analyzer import JobDescriptionAnalyzer, JobDescriptionAnalyzerError
from src.services.llm_service import LLMServiceError

VALID_REQUIREMENTS_JSON = {
    "role_title": "Backend Engineer",
    "requirements": [
        {
            "id": "req-1",
            "requirement": "3+ years of experience with Python",
            "category": "experience",
            "importance": "required",
        },
        {
            "id": "req-2",
            "requirement": "Experience with Kubernetes",
            "category": "skill",
            "importance": "preferred",
        },
        {
            "id": "req-3",
            "requirement": "Familiarity with GraphQL",
            "category": "skill",
            "importance": "unclear",
        },
    ],
    "responsibilities": [
        "Design and maintain backend services.",
        "Collaborate with product managers on API design.",
    ],
    "warnings": ["Minimum education level was not specified."],
}

SAMPLE_JD_TEXT = (
    "Backend Engineer\n\n"
    "Requirements:\n"
    "- 3+ years of experience with Python (required)\n"
    "- Experience with Kubernetes (preferred)\n"
    "- Familiarity with GraphQL\n\n"
    "Responsibilities:\n"
    "- Design and maintain backend services.\n"
    "- Collaborate with product managers on API design.\n"
)


class _FakeLLMService:
    """Stub for GeminiLLMService that records the last call and returns a canned response."""

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


def test_valid_jd_text_produces_valid_job_requirements():
    """A well-formed JSON response should be validated into a JobRequirements."""
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_REQUIREMENTS_JSON))
    analyzer = JobDescriptionAnalyzer(fake_llm)

    requirements = analyzer.analyze(SAMPLE_JD_TEXT)

    assert isinstance(requirements, JobRequirements)
    assert requirements.role_title == "Backend Engineer"
    assert len(requirements.requirements) == 3
    assert requirements.requirements[0].importance.value == "required"
    assert requirements.requirements[1].importance.value == "preferred"
    assert requirements.requirements[2].importance.value == "unclear"
    assert len(requirements.responsibilities) == 2
    assert requirements.warnings == ["Minimum education level was not specified."]


@pytest.mark.parametrize("empty_text", ["", "   ", "\n\n\t  "])
def test_empty_jd_text_raises_meaningful_error_and_skips_llm(empty_text: str):
    """Empty or whitespace-only JD text should raise an error without calling the LLM."""
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_REQUIREMENTS_JSON))
    analyzer = JobDescriptionAnalyzer(fake_llm)

    with pytest.raises(JobDescriptionAnalyzerError, match="empty"):
        analyzer.analyze(empty_text)

    assert fake_llm.last_prompt is None  # LLM should never have been called


def test_markdown_fenced_json_is_handled():
    """A response wrapped in a ```json ... ``` code fence should still parse correctly."""
    fenced_response = "```json\n" + json.dumps(VALID_REQUIREMENTS_JSON) + "\n```"
    fake_llm = _FakeLLMService(response_text=fenced_response)
    analyzer = JobDescriptionAnalyzer(fake_llm)

    requirements = analyzer.analyze(SAMPLE_JD_TEXT)

    assert requirements.role_title == "Backend Engineer"


def test_generic_code_fence_without_json_label_is_handled():
    """A plain ``` ... ``` fence (no 'json' label) should also be stripped correctly."""
    fenced_response = "```\n" + json.dumps(VALID_REQUIREMENTS_JSON) + "\n```"
    fake_llm = _FakeLLMService(response_text=fenced_response)
    analyzer = JobDescriptionAnalyzer(fake_llm)

    requirements = analyzer.analyze(SAMPLE_JD_TEXT)

    assert len(requirements.requirements) == 3


def test_invalid_json_raises_meaningful_error():
    """A non-JSON response should raise a clear JobDescriptionAnalyzerError."""
    fake_llm = _FakeLLMService(response_text="this is not JSON at all {{{")
    analyzer = JobDescriptionAnalyzer(fake_llm)

    with pytest.raises(JobDescriptionAnalyzerError, match="JSON"):
        analyzer.analyze(SAMPLE_JD_TEXT)


def test_empty_llm_response_raises_meaningful_error():
    """An empty string response from the LLM should raise a clear JobDescriptionAnalyzerError."""
    fake_llm = _FakeLLMService(response_text="   ")
    analyzer = JobDescriptionAnalyzer(fake_llm)

    with pytest.raises(JobDescriptionAnalyzerError, match="empty"):
        analyzer.analyze(SAMPLE_JD_TEXT)


def test_invalid_schema_structure_raises_meaningful_error():
    """JSON that doesn't match JobRequirements' schema should raise JobDescriptionAnalyzerError."""
    # "importance" must be one of required/preferred/unclear, not an arbitrary string.
    bad_payload = json.loads(json.dumps(VALID_REQUIREMENTS_JSON))
    bad_payload["requirements"][0]["importance"] = "mandatory"
    fake_llm = _FakeLLMService(response_text=json.dumps(bad_payload))
    analyzer = JobDescriptionAnalyzer(fake_llm)

    with pytest.raises(JobDescriptionAnalyzerError, match="structure"):
        analyzer.analyze(SAMPLE_JD_TEXT)


def test_missing_required_field_raises_meaningful_error():
    """A requirement missing a required field (e.g. 'id') should raise a schema error."""
    bad_payload = json.loads(json.dumps(VALID_REQUIREMENTS_JSON))
    del bad_payload["requirements"][0]["id"]
    fake_llm = _FakeLLMService(response_text=json.dumps(bad_payload))
    analyzer = JobDescriptionAnalyzer(fake_llm)

    with pytest.raises(JobDescriptionAnalyzerError, match="structure"):
        analyzer.analyze(SAMPLE_JD_TEXT)


def test_non_object_json_raises_meaningful_error():
    """A JSON response that isn't an object (e.g. a JSON array) should raise a clear error."""
    fake_llm = _FakeLLMService(response_text=json.dumps(["not", "an", "object"]))
    analyzer = JobDescriptionAnalyzer(fake_llm)

    with pytest.raises(JobDescriptionAnalyzerError):
        analyzer.analyze(SAMPLE_JD_TEXT)


def test_llm_service_failure_is_handled_cleanly():
    """An LLMServiceError from the underlying service should be wrapped, not leaked."""
    original_error = LLMServiceError("Gemini API request failed.")
    fake_llm = _FakeLLMService(error=original_error)
    analyzer = JobDescriptionAnalyzer(fake_llm)

    with pytest.raises(JobDescriptionAnalyzerError) as exc_info:
        analyzer.analyze(SAMPLE_JD_TEXT)

    assert exc_info.value.__cause__ is original_error


def test_prompt_instructs_llm_not_to_invent_requirements():
    """The prompt/system instruction sent to the LLM must contain explicit grounding rules."""
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_REQUIREMENTS_JSON))
    analyzer = JobDescriptionAnalyzer(fake_llm)

    analyzer.analyze(SAMPLE_JD_TEXT)

    combined_instructions = (
        (fake_llm.last_system_instruction or "") + "\n" + (fake_llm.last_prompt or "")
    ).lower()

    assert "only explicitly stated" in combined_instructions
    assert "do not infer" in combined_instructions
    assert "do not fabricate" in combined_instructions
    # Confirm the JD text itself was actually included in what was sent.
    assert "backend engineer" in combined_instructions


def test_prompt_does_not_ask_llm_to_evaluate_candidates():
    """The prompt/system instruction must not ask the model to rank, score, or evaluate candidates."""
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_REQUIREMENTS_JSON))
    analyzer = JobDescriptionAnalyzer(fake_llm)

    analyzer.analyze(SAMPLE_JD_TEXT)

    combined_instructions = (
        (fake_llm.last_system_instruction or "") + "\n" + (fake_llm.last_prompt or "")
    ).lower()

    # These verbs, if present as instructions to *act on candidates*, would
    # cross the product boundary. They should never appear as directives.
    for forbidden_phrase in ("rank candidates", "score candidates", "evaluate candidates", "candidate suitability"):
        assert forbidden_phrase not in combined_instructions

    # And the instructions should explicitly state this is normalization,
    # not evaluation.
    assert "not candidate evaluation" in combined_instructions


def test_valid_jd_text_with_minimal_requirements_json():
    """A minimal JSON response should still validate successfully via schema defaults."""
    minimal_payload = {
        "requirements": [
            {
                "id": "req-1",
                "requirement": "Bachelor's degree in Computer Science",
                "category": "education",
                "importance": "unclear",
            }
        ],
    }
    fake_llm = _FakeLLMService(response_text=json.dumps(minimal_payload))
    analyzer = JobDescriptionAnalyzer(fake_llm)

    requirements = analyzer.analyze("Bachelor's degree in Computer Science preferred.")

    assert requirements.role_title is None
    assert len(requirements.requirements) == 1
    assert requirements.responsibilities == []
    assert requirements.warnings == []
