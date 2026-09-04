"""Tests for src/services/alignment_scorer.py.

AlignmentScorer is pure application logic — no LLM, no I/O — so every test
here is a plain deterministic computation check.
"""

from __future__ import annotations

import pytest

from src.models.schemas import (
    EvidenceMatch,
    EvidenceStatus,
    JobRequirement,
    JobRequirements,
    RequirementImportance,
)
from src.services.alignment_scorer import AlignmentScorer, AlignmentScorerError


def _requirement(req_id: str, importance: RequirementImportance, text: str = "Some requirement") -> JobRequirement:
    return JobRequirement(id=req_id, requirement=text, category="skill", importance=importance)


def _match(req_id: str, status: EvidenceStatus, text: str = "Some requirement") -> EvidenceMatch:
    return EvidenceMatch(
        requirement_id=req_id,
        requirement=text,
        status=status,
        evidence=["some evidence"] if status == EvidenceStatus.evidence_found else [],
        explanation="explanation",
        confidence=0.8,
    )


@pytest.fixture
def scorer() -> AlignmentScorer:
    return AlignmentScorer()


# --- 1. All required requirements satisfied -------------------------------


def test_all_required_requirements_satisfied_scores_100(scorer):
    """With only required requirements, all evidence_found, the score should be 100."""
    job_requirements = JobRequirements(
        requirements=[
            _requirement("r1", RequirementImportance.required, "Python"),
            _requirement("r2", RequirementImportance.required, "SQL"),
        ]
    )
    matches = [
        _match("r1", EvidenceStatus.evidence_found, "Python"),
        _match("r2", EvidenceStatus.evidence_found, "SQL"),
    ]

    result = scorer.score(job_requirements, matches)

    assert result.overall_score == 100
    assert result.alignment_label == "Strong Alignment"
    assert result.required_score == 100.0


# --- 2. Several required requirements missing ------------------------------


def test_missing_several_required_requirements_lowers_score(scorer):
    """Required requirements with no evidence should pull the score down substantially."""
    job_requirements = JobRequirements(
        requirements=[
            _requirement("r1", RequirementImportance.required, "Python"),
            _requirement("r2", RequirementImportance.required, "AWS"),
            _requirement("r3", RequirementImportance.required, "Kubernetes"),
            _requirement("r4", RequirementImportance.required, "Docker"),
        ]
    )
    matches = [
        _match("r1", EvidenceStatus.evidence_found, "Python"),
        _match("r2", EvidenceStatus.no_evidence_found, "AWS"),
        _match("r3", EvidenceStatus.no_evidence_found, "Kubernetes"),
        _match("r4", EvidenceStatus.no_evidence_found, "Docker"),
    ]

    result = scorer.score(job_requirements, matches)

    # 1 of 4 required found -> 25.0 required_score -> overall == 25 (no preferred present).
    assert result.required_score == 25.0
    assert result.overall_score == 25
    assert result.overall_score < 50


# --- 3. Required satisfied but preferred missing ---------------------------


def test_required_satisfied_but_preferred_missing_caps_score_at_required_weight(scorer):
    """All required found, all preferred missing -> overall should equal the required weight's share."""
    job_requirements = JobRequirements(
        requirements=[
            _requirement("r1", RequirementImportance.required, "Python"),
            _requirement("r2", RequirementImportance.preferred, "AWS"),
            _requirement("r3", RequirementImportance.preferred, "Kubernetes"),
        ]
    )
    matches = [
        _match("r1", EvidenceStatus.evidence_found, "Python"),
        _match("r2", EvidenceStatus.no_evidence_found, "AWS"),
        _match("r3", EvidenceStatus.no_evidence_found, "Kubernetes"),
    ]

    result = scorer.score(job_requirements, matches)

    assert result.required_score == 100.0
    assert result.preferred_score == 0.0
    # 0.7 * 100 + 0.3 * 0 == 70
    assert result.overall_score == 70
    assert result.required_weight == pytest.approx(0.7)
    assert result.preferred_weight == pytest.approx(0.3)


# --- 4. No evidence for a requirement gives zero credit, not a penalty ----


def test_no_evidence_gives_zero_credit_not_negative(scorer):
    """A single no_evidence_found required requirement should score exactly 0, never negative."""
    job_requirements = JobRequirements(
        requirements=[_requirement("r1", RequirementImportance.required, "AWS")]
    )
    matches = [_match("r1", EvidenceStatus.no_evidence_found, "AWS")]

    result = scorer.score(job_requirements, matches)

    assert result.required_score == 0.0
    assert result.overall_score == 0
    assert result.overall_score >= 0


# --- 5. Partial evidence gives half credit ---------------------------------


def test_partial_evidence_gives_half_credit(scorer):
    """A needs_verification requirement should earn 0.5 credit, distinct from found/missing."""
    job_requirements = JobRequirements(
        requirements=[_requirement("r1", RequirementImportance.required, "Docker")]
    )
    matches = [_match("r1", EvidenceStatus.needs_verification, "Docker")]

    result = scorer.score(job_requirements, matches)

    assert result.required_score == 50.0
    assert result.overall_score == 50


# --- 6. Score always between 0 and 100 -------------------------------------


@pytest.mark.parametrize(
    "statuses_by_importance",
    [
        {},  # no requirements at all
        {RequirementImportance.required: [EvidenceStatus.evidence_found] * 3},
        {RequirementImportance.required: [EvidenceStatus.no_evidence_found] * 3},
        {RequirementImportance.preferred: [EvidenceStatus.needs_verification] * 2},
        {
            RequirementImportance.required: [EvidenceStatus.evidence_found, EvidenceStatus.no_evidence_found],
            RequirementImportance.preferred: [EvidenceStatus.needs_verification],
            RequirementImportance.unclear: [EvidenceStatus.no_evidence_found],
        },
    ],
)
def test_score_always_between_0_and_100(scorer, statuses_by_importance):
    requirements = []
    matches = []
    counter = 0
    for importance, statuses in statuses_by_importance.items():
        for status in statuses:
            counter += 1
            req_id = f"r{counter}"
            requirements.append(_requirement(req_id, importance))
            matches.append(_match(req_id, status))

    result = scorer.score(JobRequirements(requirements=requirements), matches)

    assert 0 <= result.overall_score <= 100


def test_empty_requirements_scores_zero_with_no_coverage(scorer):
    result = scorer.score(JobRequirements(requirements=[]), [])

    assert result.overall_score == 0
    assert result.coverage == []


# --- 7. Missing evidence must never be interpreted as confirmed absence ---


def test_no_evidence_found_is_never_phrased_as_confirmed_absence(scorer):
    """The methodology note must never claim the candidate lacks/doesn't have a skill."""
    job_requirements = JobRequirements(
        requirements=[_requirement("r1", RequirementImportance.required, "AWS")]
    )
    matches = [_match("r1", EvidenceStatus.no_evidence_found, "AWS")]

    result = scorer.score(job_requirements, matches)

    # Bare affirmative claims of absence must never appear. "lacks" itself
    # is expected to appear, but only inside the negated hedge ("not a
    # claim the candidate lacks the skill") — never as a standalone claim.
    note_lower = result.methodology_note.lower()
    forbidden_bare_claims = ["does not have", "doesn't have", "candidate cannot", "unqualified", "unsuitable"]
    for phrase in forbidden_bare_claims:
        assert phrase not in note_lower
    assert "not a claim the candidate lacks" in note_lower


def test_no_evidence_and_needs_verification_are_scored_differently(scorer):
    """Unknown (needs_verification) must not collapse into the same credit as confirmed-missing."""
    job_requirements = JobRequirements(
        requirements=[
            _requirement("r1", RequirementImportance.required, "AWS"),
            _requirement("r2", RequirementImportance.required, "Docker"),
        ]
    )
    no_evidence_matches = [
        _match("r1", EvidenceStatus.no_evidence_found, "AWS"),
        _match("r2", EvidenceStatus.evidence_found, "Docker"),
    ]
    needs_verification_matches = [
        _match("r1", EvidenceStatus.needs_verification, "AWS"),
        _match("r2", EvidenceStatus.evidence_found, "Docker"),
    ]

    no_evidence_result = scorer.score(job_requirements, no_evidence_matches)
    needs_verification_result = scorer.score(job_requirements, needs_verification_matches)

    assert no_evidence_result.overall_score < needs_verification_result.overall_score


# --- Coverage breakdown -----------------------------------------------------


def test_coverage_breaks_down_by_importance_and_status(scorer):
    job_requirements = JobRequirements(
        requirements=[
            _requirement("r1", RequirementImportance.required, "Python"),
            _requirement("r2", RequirementImportance.required, "AWS"),
            _requirement("r3", RequirementImportance.preferred, "Kubernetes"),
            _requirement("r4", RequirementImportance.unclear, "Design skills"),
        ]
    )
    matches = [
        _match("r1", EvidenceStatus.evidence_found, "Python"),
        _match("r2", EvidenceStatus.no_evidence_found, "AWS"),
        _match("r3", EvidenceStatus.needs_verification, "Kubernetes"),
        _match("r4", EvidenceStatus.evidence_found, "Design skills"),
    ]

    result = scorer.score(job_requirements, matches)
    by_importance = {row.importance: row for row in result.coverage}

    assert by_importance[RequirementImportance.required].total == 2
    assert by_importance[RequirementImportance.required].evidence_found == 1
    assert by_importance[RequirementImportance.required].no_evidence_found == 1
    assert by_importance[RequirementImportance.preferred].needs_verification == 1
    assert by_importance[RequirementImportance.unclear].evidence_found == 1
    # unclear folds into the preferred bucket for scoring, but preferred_score
    # should reflect BOTH preferred and unclear requirements (1 found, 1 needs_verification):
    # mean credit = (1.0 + 0.5) / 2 = 0.75 -> 75.0
    assert result.preferred_score == 75.0


# --- Invalid usage -----------------------------------------------------------


def test_none_job_requirements_raises_error(scorer):
    with pytest.raises(AlignmentScorerError):
        scorer.score(None, [])


def test_none_evidence_matches_raises_error(scorer):
    with pytest.raises(AlignmentScorerError):
        scorer.score(JobRequirements(requirements=[]), None)
