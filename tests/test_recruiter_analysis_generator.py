"""Tests for src/services/recruiter_analysis_generator.py.

These tests never make real Groq API calls: a fake LLM service is injected
into ``RecruiterAnalysisGenerator`` in place of ``GroqLLMService``.
"""

from __future__ import annotations

import json

import pytest

from src.models.schemas import (
    AlignmentScore,
    CandidateProfile,
    EvidenceMatch,
    EvidenceStatus,
    JobRequirement,
    JobRequirements,
    RequirementCoverage,
    RequirementImportance,
)
from src.services.llm_service import LLMServiceError
from src.services.recruiter_analysis_generator import (
    RecruiterAnalysisGenerator,
    RecruiterAnalysisGeneratorError,
)


class _FakeLLMService:
    """Stub for GroqLLMService that records calls and returns a canned response."""

    def __init__(self, response_text: str | None = None, error: Exception | None = None):
        self.response_text = response_text
        self.error = error
        self.calls: list[dict] = []

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        self.calls.append({"prompt": prompt, "system_instruction": system_instruction})
        if self.error is not None:
            raise self.error
        return self.response_text

    @property
    def last_prompt(self) -> str | None:
        return self.calls[-1]["prompt"] if self.calls else None

    @property
    def last_system_instruction(self) -> str | None:
        return self.calls[-1]["system_instruction"] if self.calls else None


CANDIDATE_PROFILE = CandidateProfile(full_name="Jordan Rivera", skills=["Python"])

JOB_REQUIREMENTS = JobRequirements(
    role_title="Backend Engineer",
    requirements=[
        JobRequirement(id="r1", requirement="Python", category="skill", importance=RequirementImportance.required),
        JobRequirement(id="r2", requirement="AWS", category="skill", importance=RequirementImportance.preferred),
    ],
)

EVIDENCE_MATCHES = [
    EvidenceMatch(
        requirement_id="r1",
        requirement="Python",
        status=EvidenceStatus.evidence_found,
        evidence=["3 years of Python"],
        explanation="Found in skills.",
        confidence=0.9,
    ),
    EvidenceMatch(
        requirement_id="r2",
        requirement="AWS",
        status=EvidenceStatus.no_evidence_found,
        evidence=[],
        explanation="Not found.",
        confidence=0.6,
    ),
]

ALIGNMENT_SCORE = AlignmentScore(
    overall_score=70,
    alignment_label="Moderate Alignment",
    required_score=100.0,
    preferred_score=0.0,
    required_weight=0.7,
    preferred_weight=0.3,
    coverage=[
        RequirementCoverage(
            importance=RequirementImportance.required, total=1, evidence_found=1, no_evidence_found=0, needs_verification=0
        ),
        RequirementCoverage(
            importance=RequirementImportance.preferred, total=1, evidence_found=0, no_evidence_found=1, needs_verification=0
        ),
    ],
    methodology_note="Computed deterministically from evidence-match statuses.",
)

VALID_ANALYSIS_JSON = {
    "summary": "The candidate shows strong alignment on required Python experience.",
    "strengths": ["Explicit Python experience evidenced in the submitted documents."],
    "gaps": ["AWS experience was not evidenced in the submitted CV."],
    "validation_areas": ["AWS/cloud experience"],
}


# --- Deterministic fallback (no llm_service) --------------------------------


def test_no_llm_service_uses_deterministic_fallback():
    """With no llm_service, the analysis should still be generated, with no LLM call."""
    generator = RecruiterAnalysisGenerator(llm_service=None)

    result = generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert result.summary
    assert "70/100" in result.summary
    assert any("Python" in s for s in result.strengths)
    assert any("AWS" in g and "not evidenced" in g for g in result.gaps)


def test_deterministic_fallback_never_claims_confirmed_absence():
    """Deterministic gaps must read as 'not evidenced', never as a confirmed lack of skill."""
    generator = RecruiterAnalysisGenerator(llm_service=None)

    result = generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    # Bare affirmative claims of absence must never appear. "lacks" itself
    # is expected to appear, but only inside the negated hedge ("does not
    # mean the candidate lacks these qualifications") — never as a
    # standalone claim.
    combined = (result.summary + " ".join(result.gaps)).lower()
    forbidden_bare_claims = ["does not have", "doesn't have", "candidate cannot", "unqualified", "unsuitable"]
    for phrase in forbidden_bare_claims:
        assert phrase not in combined
    assert "does not mean the candidate lacks" in combined
    assert all("not evidenced in the submitted cv" in gap.lower() for gap in result.gaps)


def test_deterministic_fallback_prioritizes_required_gaps_in_validation_areas():
    """Required-importance gaps should be prioritized ahead of preferred ones."""
    job_requirements = JobRequirements(
        requirements=[
            JobRequirement(id="r1", requirement="Docker", category="skill", importance=RequirementImportance.preferred),
            JobRequirement(id="r2", requirement="Kubernetes", category="skill", importance=RequirementImportance.required),
        ]
    )
    matches = [
        EvidenceMatch(
            requirement_id="r1", requirement="Docker", status=EvidenceStatus.no_evidence_found,
            evidence=[], explanation="not found", confidence=0.5,
        ),
        EvidenceMatch(
            requirement_id="r2", requirement="Kubernetes", status=EvidenceStatus.no_evidence_found,
            evidence=[], explanation="not found", confidence=0.5,
        ),
    ]
    generator = RecruiterAnalysisGenerator(llm_service=None)

    result = generator.generate(CANDIDATE_PROFILE, job_requirements, matches, ALIGNMENT_SCORE)

    assert result.validation_areas[0] == "Kubernetes"


def test_deterministic_fallback_never_calls_llm():
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_ANALYSIS_JSON))
    generator = RecruiterAnalysisGenerator(llm_service=None)

    generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert fake_llm.calls == []


# --- LLM-based synthesis -----------------------------------------------------


def test_llm_success_returns_validated_analysis():
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_ANALYSIS_JSON))
    generator = RecruiterAnalysisGenerator(fake_llm)

    result = generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert result.summary == VALID_ANALYSIS_JSON["summary"]
    assert result.strengths == VALID_ANALYSIS_JSON["strengths"]
    assert result.gaps == VALID_ANALYSIS_JSON["gaps"]
    assert result.validation_areas == VALID_ANALYSIS_JSON["validation_areas"]


def test_llm_response_wrapped_in_code_fence_is_parsed():
    fenced = f"```json\n{json.dumps(VALID_ANALYSIS_JSON)}\n```"
    fake_llm = _FakeLLMService(response_text=fenced)
    generator = RecruiterAnalysisGenerator(fake_llm)

    result = generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert result.summary == VALID_ANALYSIS_JSON["summary"]


def test_llm_service_error_falls_back_to_deterministic_analysis():
    """An LLMServiceError from the underlying service must not crash generation."""
    fake_llm = _FakeLLMService(error=LLMServiceError("Groq API request failed."))
    generator = RecruiterAnalysisGenerator(fake_llm)

    result = generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert result.summary
    assert "70/100" in result.summary


def test_malformed_json_falls_back_to_deterministic_analysis():
    fake_llm = _FakeLLMService(response_text="not valid json")
    generator = RecruiterAnalysisGenerator(fake_llm)

    result = generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert result.summary


def test_schema_invalid_response_falls_back_to_deterministic_analysis():
    """A JSON response missing the required 'summary' field should fall back, not crash."""
    bad_payload = {**VALID_ANALYSIS_JSON}
    del bad_payload["summary"]
    fake_llm = _FakeLLMService(response_text=json.dumps(bad_payload))
    generator = RecruiterAnalysisGenerator(fake_llm)

    result = generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert result.summary


def test_empty_response_falls_back_to_deterministic_analysis():
    fake_llm = _FakeLLMService(response_text="   ")
    generator = RecruiterAnalysisGenerator(fake_llm)

    result = generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert result.summary


def test_prompt_never_includes_candidate_contact_details():
    """The prompt must never carry raw CandidateProfile contact details."""
    profile = CandidateProfile(
        full_name="Jordan Rivera",
        email="jordan@example.com",
        phone="+1-555-0100",
        skills=["Python"],
    )
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_ANALYSIS_JSON))
    generator = RecruiterAnalysisGenerator(fake_llm)

    generator.generate(profile, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert "jordan@example.com" not in fake_llm.last_prompt
    assert "+1-555-0100" not in fake_llm.last_prompt


def test_prompt_does_not_ask_llm_to_invent_the_score():
    """The alignment score passed to the LLM must be the pre-computed one, presented as fixed."""
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_ANALYSIS_JSON))
    generator = RecruiterAnalysisGenerator(fake_llm)

    generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    assert "70" in fake_llm.last_prompt
    assert "Moderate Alignment" in fake_llm.last_prompt


def test_system_instruction_forbids_protected_characteristics_and_certainty():
    fake_llm = _FakeLLMService(response_text=json.dumps(VALID_ANALYSIS_JSON))
    generator = RecruiterAnalysisGenerator(fake_llm)

    generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)

    instruction = fake_llm.last_system_instruction.lower()
    assert "protected" in instruction
    assert "hire" in instruction or "hiring" in instruction
    assert "not evidenced" in instruction


# --- Invalid usage -----------------------------------------------------------


def test_none_candidate_profile_raises_error():
    generator = RecruiterAnalysisGenerator(llm_service=None)
    with pytest.raises(RecruiterAnalysisGeneratorError):
        generator.generate(None, JOB_REQUIREMENTS, EVIDENCE_MATCHES, ALIGNMENT_SCORE)


def test_none_job_requirements_raises_error():
    generator = RecruiterAnalysisGenerator(llm_service=None)
    with pytest.raises(RecruiterAnalysisGeneratorError):
        generator.generate(CANDIDATE_PROFILE, None, EVIDENCE_MATCHES, ALIGNMENT_SCORE)


def test_none_evidence_matches_raises_error():
    generator = RecruiterAnalysisGenerator(llm_service=None)
    with pytest.raises(RecruiterAnalysisGeneratorError):
        generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, None, ALIGNMENT_SCORE)


def test_none_alignment_score_raises_error():
    generator = RecruiterAnalysisGenerator(llm_service=None)
    with pytest.raises(RecruiterAnalysisGeneratorError):
        generator.generate(CANDIDATE_PROFILE, JOB_REQUIREMENTS, EVIDENCE_MATCHES, None)
