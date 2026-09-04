"""Tests for the core Pydantic data models in src/models/schemas.py."""

from datetime import datetime

import pytest
from pydantic import ValidationError

from src.models.schemas import (
    CandidateDossier,
    CandidateExperience,
    CandidateProfile,
    EvidenceMatch,
    EvidenceStatus,
    JobRequirement,
    JobRequirements,
    RequirementImportance,
)


def test_valid_candidate_profile_creation():
    """A CandidateProfile should build correctly from typical parsed data."""
    profile = CandidateProfile(
        full_name="Jordan Rivera",
        email="jordan.rivera@example.com",
        phone="+1-555-0100",
        location="Austin, TX",
        summary="Backend engineer with 5 years of experience.",
        skills=["Python", "SQL", "Docker"],
        experiences=[
            CandidateExperience(
                role="Backend Engineer",
                organization="Acme Corp",
                start_date="2020-01",
                end_date="2023-06",
                description="Built and maintained internal APIs.",
            )
        ],
        education=["B.S. Computer Science, University of Texas"],
        certifications=["AWS Certified Developer"],
        links=["https://github.com/jrivera"],
    )

    assert profile.full_name == "Jordan Rivera"
    assert profile.skills == ["Python", "SQL", "Docker"]
    assert len(profile.experiences) == 1
    assert profile.experiences[0].organization == "Acme Corp"
    # Fields not provided should default sensibly rather than error.
    assert profile.projects == []


def test_candidate_profile_allows_all_optional_fields_missing():
    """CandidateProfile should be creatable with no data at all (all fields optional/defaulted)."""
    profile = CandidateProfile()
    assert profile.full_name is None
    assert profile.skills == []
    assert profile.experiences == []


def test_evidence_status_values():
    """EvidenceStatus should expose exactly the three defined statuses."""
    assert EvidenceStatus.evidence_found.value == "evidence_found"
    assert EvidenceStatus.no_evidence_found.value == "no_evidence_found"
    assert EvidenceStatus.needs_verification.value == "needs_verification"
    assert {s.value for s in EvidenceStatus} == {
        "evidence_found",
        "no_evidence_found",
        "needs_verification",
    }


def test_requirement_importance_values():
    """RequirementImportance should support required/preferred/unclear."""
    assert {i.value for i in RequirementImportance} == {"required", "preferred", "unclear"}


def _make_evidence_match(confidence: float) -> EvidenceMatch:
    return EvidenceMatch(
        requirement_id="req-1",
        requirement="3+ years of Python experience",
        status=EvidenceStatus.evidence_found,
        evidence=["Worked as a Backend Engineer using Python for 3 years."],
        explanation="Candidate's resume explicitly lists 3 years of Python usage.",
        confidence=confidence,
    )


@pytest.mark.parametrize("confidence", [0.0, 0.5, 1.0])
def test_confidence_accepts_boundary_and_mid_values(confidence):
    """confidence must accept the full inclusive range [0, 1]."""
    match = _make_evidence_match(confidence)
    assert match.confidence == confidence


@pytest.mark.parametrize("bad_confidence", [-0.01, -1, 1.01, 2])
def test_confidence_validation_rejects_out_of_range_values(bad_confidence):
    """confidence must reject values below 0 or above 1."""
    with pytest.raises(ValidationError):
        _make_evidence_match(bad_confidence)


def test_no_evidence_found_status_is_representable_without_disproving_skill():
    """A 'no_evidence_found' match should be a normal, valid state (not an error)."""
    match = EvidenceMatch(
        requirement_id="req-2",
        requirement="Experience with Kubernetes",
        status=EvidenceStatus.no_evidence_found,
        evidence=[],
        explanation="No explicit mention of Kubernetes was found in the submitted documents.",
        confidence=0.9,
    )
    assert match.status == EvidenceStatus.no_evidence_found
    assert match.evidence == []


def test_valid_candidate_dossier_creation():
    """A CandidateDossier should assemble profile, requirements, matches, and questions."""
    profile = CandidateProfile(full_name="Sam Lee", skills=["Python"])
    job_requirements = JobRequirements(
        role_title="Backend Engineer",
        requirements=[
            JobRequirement(
                id="req-1",
                requirement="Proficiency in Python",
                category="skill",
                importance=RequirementImportance.required,
            )
        ],
        responsibilities=["Design and maintain backend services."],
    )
    evidence_matches = [_make_evidence_match(0.85)]

    dossier = CandidateDossier(
        candidate_profile=profile,
        job_requirements=job_requirements,
        evidence_matches=evidence_matches,
        interview_questions=[],
        warnings=[],
    )

    assert dossier.candidate_profile.full_name == "Sam Lee"
    assert dossier.job_requirements.role_title == "Backend Engineer"
    assert len(dossier.evidence_matches) == 1
    assert isinstance(dossier.generated_at, datetime)


def test_candidate_dossier_requires_profile_and_requirements():
    """CandidateDossier should fail validation when required fields are missing."""
    with pytest.raises(ValidationError):
        CandidateDossier()
