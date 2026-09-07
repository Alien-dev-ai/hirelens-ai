"""Tests for src/services/evidence_matcher.py.

These tests never make real Groq API calls: a fake LLM service is
injected into ``EvidenceMatcher`` in place of ``GroqLLMService``.
"""

from __future__ import annotations

import json

import pytest

from src.models.schemas import (
    CandidateExperience,
    CandidateProfile,
    CandidateProject,
    EvidenceStatus,
    JobRequirement,
    JobRequirements,
    RequirementImportance,
)
from src.services.evidence_matcher import EvidenceMatcher, EvidenceMatcherError
from src.services.llm_service import LLMServiceError


class _FakeLLMService:
    """Stub for GroqLLMService that records the last call and returns a canned response."""

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


def _make_requirement(req_id: str, text: str, category: str = "skill", importance=RequirementImportance.required) -> JobRequirement:
    return JobRequirement(id=req_id, requirement=text, category=category, importance=importance)


# --- Deterministic matching -------------------------------------------------


def test_exact_skill_match():
    """A requirement whose term appears directly in the candidate's skills list is matched."""
    profile = CandidateProfile(skills=["Python", "SQL"])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "3+ years of experience with Python")]
    )
    matcher = EvidenceMatcher()  # no LLM needed for a deterministic match

    matches = matcher.match(profile, job_requirements)

    assert len(matches) == 1
    match = matches[0]
    assert match.requirement_id == "req-1"
    assert match.status == EvidenceStatus.evidence_found
    assert "Python" in match.evidence
    assert 0.0 <= match.confidence <= 1.0


def test_match_found_inside_experience_description():
    """A requirement term found only in an experience description is matched."""
    profile = CandidateProfile(
        experiences=[
            CandidateExperience(
                role="Platform Engineer",
                organization="Acme Corp",
                description="Deployed and managed production workloads using Kubernetes on AWS.",
            )
        ]
    )
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Experience with Kubernetes")]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.evidence_found
    assert any("Kubernetes" in e for e in matches[0].evidence)
    # Evidence must be traceable to the actual candidate profile text.
    assert matches[0].evidence[0] == profile.experiences[0].description


def test_match_found_inside_project_technologies_and_description():
    """A requirement term found in project technologies or description is matched."""
    profile = CandidateProfile(
        projects=[
            CandidateProject(
                name="Internal API Gateway",
                description="Built a GraphQL gateway service for internal microservices.",
                technologies=["Python", "GraphQL", "Docker"],
            )
        ]
    )
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement("req-1", "Familiarity with GraphQL"),
            _make_requirement("req-2", "Experience with Docker"),
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.evidence_found
    assert "GraphQL" in matches[0].evidence
    assert matches[1].status == EvidenceStatus.evidence_found
    assert "Docker" in matches[1].evidence


# --- Category-aware credential source scoping (education/certification/license) ---


def test_education_requirement_not_satisfied_by_overlapping_skill():
    """Regression test for the confirmed false positive: an education
    requirement must not be satisfiable by an unrelated skill entry that
    merely shares a word with it (e.g. 'media' in 'Social Media Content
    Creation' vs. a Film/Media/Communications degree requirement)."""
    profile = CandidateProfile(
        skills=["Social Media Content Creation", "Microsoft Excel"],
        education=["BA in Business Administration — State University"],
    )
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement(
                "req-1",
                "Bachelor's degree in Film, Media, Communications, or a related field",
                category="education",
            )
        ]
    )
    matcher = EvidenceMatcher()  # deterministic only — proves the fix at the deterministic layer

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.no_evidence_found
    assert matches[0].evidence == []
    assert matches[0].confidence == 0.6  # deterministic no-evidence confidence, not the prior 0.95 skill-match bug


def test_education_requirement_not_satisfied_by_overlapping_experience_text():
    """An education requirement must not be satisfiable by work-experience
    text that happens to share a word with it, either."""
    profile = CandidateProfile(
        education=["BA in Business Administration — State University"],
        experiences=[
            CandidateExperience(
                role="Marketing Coordinator",
                organization="Acme Co",
                description="Created social media content and managed the brand Instagram account.",
            )
        ],
    )
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement(
                "req-1",
                "Bachelor's degree in Film, Media, Communications, or a related field",
                category="education",
            )
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.no_evidence_found
    assert matches[0].evidence == []


def test_education_requirement_matches_correct_education_entry():
    """A genuinely matching education entry is still found via the education source."""
    profile = CandidateProfile(education=["BSc Computer Science — State University"])
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement(
                "req-1",
                "Bachelor's degree in Computer Science or a related field",
                category="education",
            )
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.evidence_found
    assert matches[0].evidence == ["BSc Computer Science — State University"]


def test_education_requirement_not_satisfied_by_different_explicit_degree():
    """An explicit, non-matching degree on file must not be misread as satisfying
    an unrelated education requirement."""
    profile = CandidateProfile(education=["BA in Business Administration — State University"])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Bachelor's degree in Computer Science", category="education")]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.no_evidence_found
    assert matches[0].evidence == []


def test_education_requirement_falls_back_to_llm_when_deterministic_is_inconclusive():
    """Scoping to the education source must not replace the LLM fallback — a
    genuinely ambiguous/related-but-not-literal education match should still
    reach the LLM."""
    profile = CandidateProfile(education=["B.A. in Communication Studies - Central University"])
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement(
                "req-1",
                "Bachelor's degree in Film, Media Studies, or a related field",
                category="education",
            )
        ]
    )
    llm_response = json.dumps(
        {
            "status": "evidence_found",
            "evidence": ["B.A. in Communication Studies - Central University"],
            "explanation": "The candidate's education lists a Communication Studies degree, a closely related field.",
        }
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    matcher = EvidenceMatcher(fake_llm)

    matches = matcher.match(profile, job_requirements)

    assert len(fake_llm.calls) == 1  # deterministic scoped matching was inconclusive; LLM was consulted
    assert matches[0].status == EvidenceStatus.evidence_found


def test_certification_requirement_not_satisfied_by_overlapping_skill():
    """A certification requirement must not be satisfied merely because a
    related skill (not an actual certification) shares words with it."""
    profile = CandidateProfile(skills=["Project Management"], certifications=[])
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement(
                "req-1", "Certification in Project Management or equivalent", category="certification"
            )
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.no_evidence_found
    assert matches[0].evidence == []


def test_certification_requirement_matches_certification_entry():
    profile = CandidateProfile(certifications=["PMP - Project Management Professional"])
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement(
                "req-1", "Certification in Project Management or equivalent", category="certification"
            )
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.evidence_found
    assert matches[0].evidence == ["PMP - Project Management Professional"]


def test_license_requirement_not_satisfied_by_overlapping_skill():
    """License requirements have no dedicated source field, so they are scoped
    to 'certification' (the existing field licenses are recorded under); an
    unrelated skill mention must not satisfy one."""
    profile = CandidateProfile(skills=["Registered Nurse duties"], certifications=[])
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement(
                "req-1",
                "Registered Nurse, Licensed Practical Nurse, or equivalent license",
                category="license",
            )
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.no_evidence_found
    assert matches[0].evidence == []


def test_license_requirement_matches_certification_entry():
    profile = CandidateProfile(certifications=["Registered Nurse (RN) License - State Board of Nursing"])
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement(
                "req-1",
                "Registered Nurse, Licensed Practical Nurse, or equivalent license",
                category="license",
            )
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.evidence_found
    assert matches[0].evidence == ["Registered Nurse (RN) License - State Board of Nursing"]


def test_skill_category_requirement_still_searches_all_sources():
    """Category scoping is specific to credential categories (education/
    certification/license) — ordinary skill requirements must keep
    searching all candidate sources, unchanged from prior behavior."""
    profile = CandidateProfile(education=["BSc Computer Science, minor in Python programming"])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Python", category="skill")]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.evidence_found
    assert matches[0].evidence == ["BSc Computer Science, minor in Python programming"]


def test_no_evidence_found_status_when_nothing_matches():
    """A requirement with no supporting evidence anywhere gets no_evidence_found."""
    profile = CandidateProfile(skills=["Python", "SQL"])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Experience with Rust")]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.no_evidence_found
    assert matches[0].evidence == []


def test_no_evidence_found_is_not_a_negative_claim_about_candidate():
    """The explanation for no_evidence_found must never claim the candidate lacks the skill."""
    profile = CandidateProfile(skills=["Python"])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Experience with Rust")]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)
    explanation_lower = matches[0].explanation.lower()

    forbidden_phrases = ["lacks", "does not have", "unqualified", "unsuitable", "not qualified", "not suitable"]
    for phrase in forbidden_phrases:
        assert phrase not in explanation_lower
    # It should positively clarify the distinction.
    assert "does not mean" in explanation_lower


def test_every_requirement_receives_a_corresponding_match():
    """Every JobRequirement must produce exactly one EvidenceMatch, in order."""
    profile = CandidateProfile(skills=["Python"])
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement("req-1", "Python"),
            _make_requirement("req-2", "Rust"),
            _make_requirement("req-3", "SQL"),
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert len(matches) == 3
    assert [m.requirement_id for m in matches] == ["req-1", "req-2", "req-3"]


def test_evidence_is_traceable_to_candidate_profile_data():
    """Every evidence string returned must literally exist somewhere in the candidate profile."""
    profile = CandidateProfile(
        skills=["Python", "Docker"],
        experiences=[
            CandidateExperience(role="Engineer", organization="Acme", description="Built APIs with Python.")
        ],
        certifications=["AWS Certified Developer"],
    )
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement("req-1", "Python"),
            _make_requirement("req-2", "AWS Certified Developer"),
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    all_candidate_strings = (
        profile.skills
        + profile.certifications
        + [e.description for e in profile.experiences if e.description]
    )
    for match in matches:
        for evidence_item in match.evidence:
            assert evidence_item in all_candidate_strings


def test_empty_job_requirements_returns_empty_list():
    """An empty requirements list should return an empty match list, not error."""
    profile = CandidateProfile(skills=["Python"])
    job_requirements = JobRequirements(requirements=[])
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert matches == []


def test_empty_candidate_profile_yields_no_evidence_for_all_requirements():
    """An entirely empty candidate profile should not crash and should yield no_evidence_found."""
    profile = CandidateProfile()
    job_requirements = JobRequirements(
        requirements=[
            _make_requirement("req-1", "Python"),
            _make_requirement("req-2", "SQL"),
        ]
    )
    matcher = EvidenceMatcher()

    matches = matcher.match(profile, job_requirements)

    assert len(matches) == 2
    assert all(m.status == EvidenceStatus.no_evidence_found for m in matches)


def test_missing_candidate_profile_raises_error():
    matcher = EvidenceMatcher()
    with pytest.raises(EvidenceMatcherError):
        matcher.match(None, JobRequirements(requirements=[]))


def test_missing_job_requirements_raises_error():
    matcher = EvidenceMatcher()
    with pytest.raises(EvidenceMatcherError):
        matcher.match(CandidateProfile(), None)


# --- LLM-based semantic matching --------------------------------------------


def test_semantic_matching_via_llm_when_deterministic_is_inconclusive():
    """When no direct keyword match exists, the LLM is consulted for semantic evidence."""
    profile = CandidateProfile(
        experiences=[
            CandidateExperience(
                role="Backend Engineer",
                organization="Acme Corp",
                description="Worked extensively with distributed message queues and Kafka for event-driven microservices.",
            )
        ]
    )
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Experience designing distributed systems", category="skill")]
    )
    llm_response = json.dumps(
        {
            "status": "evidence_found",
            "evidence": [
                "Worked extensively with distributed message queues and Kafka for event-driven microservices."
            ],
            "explanation": "The experience description explicitly mentions building distributed, event-driven systems.",
        }
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    matcher = EvidenceMatcher(fake_llm)

    matches = matcher.match(profile, job_requirements)

    assert len(fake_llm.calls) == 1  # LLM was actually consulted
    assert matches[0].status == EvidenceStatus.evidence_found
    assert matches[0].evidence == [
        "Worked extensively with distributed message queues and Kafka for event-driven microservices."
    ]


def test_llm_evidence_not_traceable_is_downgraded_to_needs_verification():
    """LLM-claimed evidence that isn't actually present in the candidate data must not be trusted."""
    profile = CandidateProfile(experiences=[CandidateExperience(role="Engineer", description="Wrote unit tests.")])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Experience with distributed systems")]
    )
    llm_response = json.dumps(
        {
            "status": "evidence_found",
            "evidence": ["Led a team of 10 distributed systems engineers."],  # fabricated, not in profile
            "explanation": "Candidate led a distributed systems team.",
        }
    )
    fake_llm = _FakeLLMService(response_text=llm_response)
    matcher = EvidenceMatcher(fake_llm)

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.needs_verification
    assert matches[0].evidence == []


def test_llm_service_failure_falls_back_to_needs_verification():
    """An LLMServiceError from the underlying service must not crash matching."""
    profile = CandidateProfile(experiences=[CandidateExperience(role="Engineer", description="Wrote unit tests.")])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Experience with distributed systems")]
    )
    fake_llm = _FakeLLMService(error=LLMServiceError("Groq API request failed."))
    matcher = EvidenceMatcher(fake_llm)

    matches = matcher.match(profile, job_requirements)

    assert len(matches) == 1
    assert matches[0].status == EvidenceStatus.needs_verification
    assert matches[0].requirement_id == "req-1"


def test_malformed_llm_json_falls_back_to_needs_verification():
    """A non-JSON LLM response must not crash matching."""
    profile = CandidateProfile(experiences=[CandidateExperience(role="Engineer", description="Wrote unit tests.")])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Experience with distributed systems")]
    )
    fake_llm = _FakeLLMService(response_text="not valid json {{{")
    matcher = EvidenceMatcher(fake_llm)

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.needs_verification


def test_llm_invalid_status_value_falls_back_to_needs_verification():
    """An LLM response with a status outside the EvidenceStatus enum must not crash matching."""
    profile = CandidateProfile(experiences=[CandidateExperience(role="Engineer", description="Wrote unit tests.")])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Experience with distributed systems")]
    )
    fake_llm = _FakeLLMService(
        response_text=json.dumps({"status": "strong_match", "evidence": [], "explanation": "Good fit."})
    )
    matcher = EvidenceMatcher(fake_llm)

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.needs_verification


def test_deterministic_match_avoids_unnecessary_llm_call():
    """When deterministic matching already finds evidence, the LLM must not be called."""
    profile = CandidateProfile(skills=["Python"])
    job_requirements = JobRequirements(requirements=[_make_requirement("req-1", "Python")])
    fake_llm = _FakeLLMService(response_text=json.dumps({"status": "evidence_found", "evidence": [], "explanation": "x"}))
    matcher = EvidenceMatcher(fake_llm)

    matches = matcher.match(profile, job_requirements)

    assert matches[0].status == EvidenceStatus.evidence_found
    assert fake_llm.calls == []  # LLM should never have been invoked


def test_prompt_contains_grounding_and_no_evaluation_language():
    """The LLM prompt/system instruction must forbid fabrication and never ask for scoring/ranking/hiring decisions."""
    profile = CandidateProfile(experiences=[CandidateExperience(role="Engineer", description="Wrote unit tests.")])
    job_requirements = JobRequirements(
        requirements=[_make_requirement("req-1", "Experience with distributed systems")]
    )
    llm_response = json.dumps({"status": "no_evidence_found", "evidence": [], "explanation": "No mention found."})
    fake_llm = _FakeLLMService(response_text=llm_response)
    matcher = EvidenceMatcher(fake_llm)

    matcher.match(profile, job_requirements)

    combined = ((fake_llm.last_system_instruction or "") + "\n" + (fake_llm.last_prompt or "")).lower()

    assert "never invent" in combined or "do not invent" in combined or "fabricate" in combined
    assert "no_evidence_found" in combined
    assert "not a claim that the candidate lacks" in combined or "not a claim" in combined

    # These are directive phrases that would only appear if the prompt were
    # asking the model to *perform* scoring/ranking/hiring actions on the
    # candidate. Note: the prompt legitimately *prohibits* such actions
    # (e.g. "never recommend ... any hiring decision"), so this checks for
    # directive phrasing specifically, not the mere words "score"/"rank"/
    # "hiring" in isolation.
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
    ]
    for phrase in forbidden_directive_phrases:
        assert phrase not in combined
