"""Tests for src/services/interview_generator.py.

These tests never make real Groq API calls: a fake LLM service is
injected into ``InterviewQuestionGenerator`` in place of
``GroqLLMService``.
"""

from __future__ import annotations

import json

import pytest

from src.models.schemas import (
    CandidateProfile,
    EvidenceMatch,
    EvidenceStatus,
    InterviewQuestion,
    JobRequirement,
    JobRequirements,
    RequirementImportance,
)
from src.services.interview_generator import (
    InterviewQuestionGenerator,
    InterviewQuestionGeneratorError,
)
from src.services.llm_service import LLMServiceError


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


def _make_requirement(req_id: str, text: str, importance=RequirementImportance.required, category: str = "skill") -> JobRequirement:
    return JobRequirement(id=req_id, requirement=text, category=category, importance=importance)


def _make_match(req_id: str, requirement_text: str, status: EvidenceStatus, evidence: list[str] | None = None) -> EvidenceMatch:
    return EvidenceMatch(
        requirement_id=req_id,
        requirement=requirement_text,
        status=status,
        evidence=evidence or [],
        explanation="test explanation",
        confidence=0.8,
    )


CANDIDATE_PROFILE = CandidateProfile(full_name="Jordan Rivera", skills=["Python", "FastAPI"])


# --- Deterministic behavior (no LLM injected) --------------------------------


def test_evidence_found_generates_grounded_verification_question():
    """evidence_found should produce a question that references the specific evidence."""
    requirement = _make_requirement("req-1", "Experience building REST APIs using FastAPI")
    match = _make_match(
        "req-1",
        requirement.requirement,
        EvidenceStatus.evidence_found,
        evidence=["Developed REST APIs using Python and FastAPI."],
    )
    job_requirements = JobRequirements(requirements=[requirement])
    generator = InterviewQuestionGenerator()  # no LLM

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert len(questions) == 1
    question = questions[0]
    assert isinstance(question, InterviewQuestion)
    assert "Developed REST APIs using Python and FastAPI." in question.question
    assert question.focus_requirement == requirement.requirement
    assert question.priority in {"high", "medium", "low"}


def test_no_evidence_found_generates_neutral_clarification_question():
    """no_evidence_found should produce a neutral question, never implying the candidate lacks the skill."""
    requirement = _make_requirement("req-1", "Kubernetes or container orchestration")
    match = _make_match("req-1", requirement.requirement, EvidenceStatus.no_evidence_found)
    job_requirements = JobRequirements(requirements=[requirement])
    generator = InterviewQuestionGenerator()

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    question = questions[0]
    combined = (question.question + " " + question.reason).lower()
    # Note: the reason text is expected to *negate* these claims (e.g. "does
    # not mean the candidate is missing this qualification") — that
    # negation is exactly the desired, neutral phrasing. What must never
    # appear is an *affirmative* negative claim, so we check for that
    # directly rather than banning these words outright.
    assert "does not mean" in question.reason.lower()
    forbidden_affirmative_phrases = [
        "the candidate lacks",
        "candidate does not have",
        "candidate is unqualified",
        "candidate is unsuitable",
        "candidate is not qualified",
        "candidate is not suitable",
    ]
    for phrase in forbidden_affirmative_phrases:
        assert phrase not in combined
    assert "kubernetes" in question.question.lower()


def test_needs_verification_generates_neutral_clarifying_question():
    """needs_verification should produce a neutral clarifying question."""
    requirement = _make_requirement("req-1", "Experience with GraphQL")
    match = _make_match("req-1", requirement.requirement, EvidenceStatus.needs_verification, evidence=["Mentioned APIs."])
    job_requirements = JobRequirements(requirements=[requirement])
    generator = InterviewQuestionGenerator()

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    question = questions[0]
    assert "graphql" in question.question.lower() or "graphql" in question.reason.lower()
    assert question.focus_requirement == requirement.requirement


def test_empty_requirements_returns_empty_list():
    job_requirements = JobRequirements(requirements=[])
    generator = InterviewQuestionGenerator()

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [])

    assert questions == []


def test_empty_evidence_matches_still_generates_neutral_questions():
    """With requirements but zero evidence matches, every requirement still gets a question."""
    requirements = [
        _make_requirement("req-1", "Python"),
        _make_requirement("req-2", "SQL"),
    ]
    job_requirements = JobRequirements(requirements=requirements)
    generator = InterviewQuestionGenerator()

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [])

    assert len(questions) == 2
    assert questions[0].focus_requirement == "Python"
    assert questions[1].focus_requirement == "SQL"


def test_missing_match_for_one_requirement_does_not_crash_others():
    """A requirement with no corresponding EvidenceMatch should still get a question."""
    requirements = [
        _make_requirement("req-1", "Python"),
        _make_requirement("req-2", "Kubernetes"),
    ]
    # Only req-1 has a match; req-2 is missing.
    match = _make_match("req-1", "Python", EvidenceStatus.evidence_found, evidence=["Python"])
    job_requirements = JobRequirements(requirements=requirements)
    generator = InterviewQuestionGenerator()

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert len(questions) == 2
    assert questions[0].focus_requirement == "Python"
    assert questions[1].focus_requirement == "Kubernetes"  # generated despite missing match


def test_every_question_connected_to_correct_requirement():
    """Each generated question's focus_requirement must match its own requirement, in order."""
    requirements = [
        _make_requirement("req-1", "Python"),
        _make_requirement("req-2", "Kubernetes"),
        _make_requirement("req-3", "SQL"),
    ]
    matches = [
        _make_match("req-1", "Python", EvidenceStatus.evidence_found, evidence=["Python"]),
        _make_match("req-2", "Kubernetes", EvidenceStatus.no_evidence_found),
        _make_match("req-3", "SQL", EvidenceStatus.needs_verification),
    ]
    job_requirements = JobRequirements(requirements=requirements)
    generator = InterviewQuestionGenerator()

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, matches)

    assert [q.focus_requirement for q in questions] == ["Python", "Kubernetes", "SQL"]


def test_missing_arguments_raise_error():
    generator = InterviewQuestionGenerator()
    with pytest.raises(InterviewQuestionGeneratorError):
        generator.generate(None, JobRequirements(requirements=[]), [])
    with pytest.raises(InterviewQuestionGeneratorError):
        generator.generate(CANDIDATE_PROFILE, None, [])
    with pytest.raises(InterviewQuestionGeneratorError):
        generator.generate(CANDIDATE_PROFILE, JobRequirements(requirements=[]), None)


# --- LLM-based generation -----------------------------------------------------


def test_valid_llm_question_generation():
    """A well-formed LLM JSON response should be used to build the InterviewQuestion."""
    requirement = _make_requirement("req-1", "Experience building REST APIs using FastAPI")
    match = _make_match(
        "req-1", requirement.requirement, EvidenceStatus.evidence_found, evidence=["Developed REST APIs using Python and FastAPI."]
    )
    job_requirements = JobRequirements(requirements=[requirement])
    llm_response = json.dumps(
        {
            "question": "Can you walk me through the REST APIs you developed using FastAPI and explain the technical decisions you made?",
            "focus_requirement": requirement.requirement,
            "reason": "Evidence for this requirement was found; this explores it in more depth.",
            "priority": "medium",
        }
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert len(fake_llm.calls) == 1
    assert questions[0].question.startswith("Can you walk me through the REST APIs")
    assert questions[0].focus_requirement == requirement.requirement
    assert questions[0].priority == "medium"


def test_markdown_fenced_llm_response_is_handled():
    requirement = _make_requirement("req-1", "Python")
    match = _make_match("req-1", "Python", EvidenceStatus.evidence_found, evidence=["Python"])
    job_requirements = JobRequirements(requirements=[requirement])
    fenced_response = "```json\n" + json.dumps(
        {"question": "Tell me about your Python experience.", "focus_requirement": "Python", "reason": "Explore evidence.", "priority": "low"}
    ) + "\n```"
    fake_llm = _FakeLLMService(response_text=fenced_response)
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert questions[0].question == "Tell me about your Python experience."


def test_invalid_json_falls_back_to_deterministic_question():
    """An LLM response that isn't valid JSON should not crash — it should fall back deterministically."""
    requirement = _make_requirement("req-1", "Python")
    match = _make_match("req-1", "Python", EvidenceStatus.evidence_found, evidence=["Python"])
    job_requirements = JobRequirements(requirements=[requirement])
    fake_llm = _FakeLLMService(response_text="not valid json {{{")
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert len(questions) == 1
    assert questions[0].focus_requirement == "Python"
    assert isinstance(questions[0], InterviewQuestion)


def test_empty_llm_response_falls_back_to_deterministic_question():
    requirement = _make_requirement("req-1", "Python")
    match = _make_match("req-1", "Python", EvidenceStatus.no_evidence_found)
    job_requirements = JobRequirements(requirements=[requirement])
    fake_llm = _FakeLLMService(response_text="   ")
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert len(questions) == 1
    assert questions[0].focus_requirement == "Python"


def test_llm_service_failure_falls_back_to_deterministic_question():
    requirement = _make_requirement("req-1", "Python")
    match = _make_match("req-1", "Python", EvidenceStatus.evidence_found, evidence=["Python"])
    job_requirements = JobRequirements(requirements=[requirement])
    fake_llm = _FakeLLMService(error=LLMServiceError("Groq API request failed."))
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert len(questions) == 1
    assert questions[0].focus_requirement == "Python"


def test_llm_schema_validation_failure_falls_back_to_deterministic_question():
    """An LLM JSON response missing a required InterviewQuestion field should fall back safely."""
    requirement = _make_requirement("req-1", "Python")
    match = _make_match("req-1", "Python", EvidenceStatus.evidence_found, evidence=["Python"])
    job_requirements = JobRequirements(requirements=[requirement])
    # Missing "reason", which InterviewQuestion requires.
    bad_response = json.dumps({"question": "Tell me about Python.", "focus_requirement": "Python", "priority": "medium"})
    fake_llm = _FakeLLMService(response_text=bad_response)
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert len(questions) == 1
    assert questions[0].reason  # deterministic fallback always fills reason
    assert questions[0].focus_requirement == "Python"


def test_llm_focus_requirement_is_always_overridden_with_ground_truth():
    """Even if the LLM returns a wrong/garbled focus_requirement, the real requirement text wins."""
    requirement = _make_requirement("req-1", "Python")
    match = _make_match("req-1", "Python", EvidenceStatus.evidence_found, evidence=["Python"])
    job_requirements = JobRequirements(requirements=[requirement])
    llm_response = json.dumps(
        {"question": "Tell me about Python.", "focus_requirement": "SOMETHING ELSE ENTIRELY", "reason": "x", "priority": "medium"}
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert questions[0].focus_requirement == "Python"


# --- Requirement-echo guard ---------------------------------------------------


def test_llm_question_identical_to_requirement_falls_back_to_deterministic():
    """An LLM 'question' that's just the requirement text restated must be rejected and replaced."""
    requirement = _make_requirement("req-1", "Experience with Adobe Premiere Pro")
    match = _make_match(
        "req-1", requirement.requirement, EvidenceStatus.evidence_found, evidence=["Edited videos using Adobe Premiere Pro."]
    )
    job_requirements = JobRequirements(requirements=[requirement])
    llm_response = json.dumps(
        {
            "question": "Experience with Adobe Premiere Pro",
            "focus_requirement": requirement.requirement,
            "reason": "Checking this requirement.",
            "priority": "medium",
        }
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert len(fake_llm.calls) == 1  # the LLM was still called once; only its output was rejected
    assert questions[0].question != requirement.requirement
    # The deterministic template for evidence_found always starts this way —
    # proves the fallback path actually engaged, not just any different text.
    assert questions[0].question.startswith("You mentioned the following in your submitted documents")
    assert questions[0].focus_requirement == requirement.requirement


def test_llm_question_matching_requirement_after_normalization_falls_back_to_deterministic():
    """Whitespace/case differences alone must not let an echoed requirement pass as a real question."""
    requirement = _make_requirement("req-1", "Experience with Adobe Premiere Pro")
    match = _make_match(
        "req-1", requirement.requirement, EvidenceStatus.evidence_found, evidence=["Edited videos using Adobe Premiere Pro."]
    )
    job_requirements = JobRequirements(requirements=[requirement])
    llm_response = json.dumps(
        {
            "question": "  EXPERIENCE with adobe premiere pro  ",
            "focus_requirement": requirement.requirement,
            "reason": "Checking this requirement.",
            "priority": "medium",
        }
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert questions[0].question.startswith("You mentioned the following in your submitted documents")
    assert questions[0].focus_requirement == requirement.requirement


def test_llm_question_sharing_wording_with_requirement_is_accepted():
    """A real question that reuses requirement wording (but isn't identical to it) must NOT be rejected."""
    requirement = _make_requirement("req-1", "Experience with Adobe Premiere Pro")
    match = _make_match(
        "req-1", requirement.requirement, EvidenceStatus.evidence_found, evidence=["Edited videos using Adobe Premiere Pro."]
    )
    job_requirements = JobRequirements(requirements=[requirement])
    question_text = "Can you walk me through a project where your experience with Adobe Premiere Pro made a difference?"
    llm_response = json.dumps(
        {
            "question": question_text,
            "focus_requirement": requirement.requirement,
            "reason": "Evidence for this requirement was found; this explores it in more depth.",
            "priority": "medium",
        }
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert questions[0].question == question_text


def test_valid_llm_question_for_no_evidence_found_remains_accepted():
    """A genuine, well-formed LLM question with no wording overlap must still pass the guard."""
    requirement = _make_requirement("req-1", "Experience with Kubernetes")
    match = _make_match("req-1", requirement.requirement, EvidenceStatus.no_evidence_found)
    job_requirements = JobRequirements(requirements=[requirement])
    question_text = "Have you worked with container orchestration tools, and if so, could you describe that experience?"
    llm_response = json.dumps(
        {
            "question": question_text,
            "focus_requirement": requirement.requirement,
            "reason": "No explicit evidence was found for this requirement in the submitted documents.",
            "priority": "high",
        }
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    generator = InterviewQuestionGenerator(fake_llm)

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    assert questions[0].question == question_text
    assert questions[0].focus_requirement == requirement.requirement


# --- Prompt grounding verification --------------------------------------------


def test_prompt_grounds_question_in_requirement_and_evidence():
    requirement = _make_requirement("req-1", "Experience with FastAPI")
    match = _make_match("req-1", requirement.requirement, EvidenceStatus.evidence_found, evidence=["Built FastAPI services."])
    job_requirements = JobRequirements(requirements=[requirement])
    llm_response = json.dumps(
        {"question": "Tell me more.", "focus_requirement": requirement.requirement, "reason": "x", "priority": "medium"}
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    generator = InterviewQuestionGenerator(fake_llm)

    generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    combined = ((fake_llm.last_system_instruction or "") + "\n" + (fake_llm.last_prompt or "")).lower()
    assert "experience with fastapi" in combined
    assert "built fastapi services." in combined
    assert "do not invent" in combined or "never invent" in combined


def test_prompt_does_not_ask_for_scoring_ranking_or_hiring_decisions():
    requirement = _make_requirement("req-1", "Python")
    match = _make_match("req-1", "Python", EvidenceStatus.evidence_found, evidence=["Python"])
    job_requirements = JobRequirements(requirements=[requirement])
    llm_response = json.dumps(
        {"question": "Tell me more.", "focus_requirement": "Python", "reason": "x", "priority": "medium"}
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    generator = InterviewQuestionGenerator(fake_llm)

    generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    combined = ((fake_llm.last_system_instruction or "") + "\n" + (fake_llm.last_prompt or "")).lower()
    forbidden_directive_phrases = [
        "rank candidates",
        "rank the candidate",
        "score candidates",
        "score the candidate",
        "evaluate candidates",
        "candidate suitability",
        "recommend hiring",
        "recommend acceptance",
        "recommend rejection",
        "suitability score",
        "should we hire",
        "accept or reject",
    ]
    for phrase in forbidden_directive_phrases:
        assert phrase not in combined


def test_prompt_for_no_evidence_found_explicitly_forbids_negative_claims():
    """The prompt for a no_evidence_found requirement must instruct neutrality explicitly."""
    requirement = _make_requirement("req-1", "Kubernetes")
    match = _make_match("req-1", "Kubernetes", EvidenceStatus.no_evidence_found)
    job_requirements = JobRequirements(requirements=[requirement])
    llm_response = json.dumps(
        {"question": "Tell me about Kubernetes.", "focus_requirement": "Kubernetes", "reason": "x", "priority": "medium"}
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    generator = InterviewQuestionGenerator(fake_llm)

    generator.generate(CANDIDATE_PROFILE, job_requirements, [match])

    combined = ((fake_llm.last_system_instruction or "") + "\n" + (fake_llm.last_prompt or "")).lower()
    assert "no_evidence_found" in combined
    assert "never imply" in combined or "must never" in combined


def test_deterministic_used_when_no_llm_injected_avoids_any_llm_call():
    """No LLM service means no LLM call at all, for every requirement."""
    requirements = [_make_requirement("req-1", "Python"), _make_requirement("req-2", "SQL")]
    job_requirements = JobRequirements(requirements=requirements)
    generator = InterviewQuestionGenerator()  # no LLM

    questions = generator.generate(CANDIDATE_PROFILE, job_requirements, [])

    assert len(questions) == 2
    for question in questions:
        assert isinstance(question, InterviewQuestion)
