"""Deterministic candidate-to-role alignment scoring for HireLens AI.

Computes a 0-100 "alignment score" purely from application logic — never
from an LLM — reflecting how strongly the evidence already gathered by the
Evidence Matcher covers a job's stated requirements.

PRODUCT DIRECTION:
HireLens has intentionally evolved from a pure evidence-extraction tool
into a recruiter-facing decision-support system, and this module is part
of that: it IS allowed to compute a score. What it must never do:

- Claim the score is an objective probability of hiring success, or any
  kind of certainty. It is decision *support* for a human recruiter, never
  a hiring decision.
- Treat a requirement with no evidence as proof the candidate lacks that
  skill. Missing evidence simply cannot earn credit — that is different
  from confirmed absence — and every user-facing description of a gap
  must say "not evidenced in the submitted CV", never "does not have" or
  "lacks".
- Consider, or be influenced by, any protected characteristic (race,
  ethnicity, religion, gender, age, disability, marital status,
  nationality, appearance, health, or similar). Nothing in this module's
  inputs (``JobRequirements``, ``EvidenceMatch``) carries such information
  in the first place, so there is nothing here for it to act on.

SCORING METHODOLOGY:
Each requirement contributes a 0.0-1.0 credit based on its ``EvidenceMatch``
status: 1.0 for ``evidence_found``, 0.5 for ``needs_verification`` (partial
credit for partially-supported evidence), 0.0 for ``no_evidence_found`` (no
credit — never a penalty beyond that, and never a claim of confirmed
absence). Requirements are grouped into two buckets:

- "required": requirements the job description marked as required.
- "preferred": requirements marked as preferred *or* unclear. The current
  schema (``CandidateProfile`` / ``JobRequirements``) has no reliable,
  non-invented signal — e.g. verified years of experience or a validated
  role-title match — to justify a separate "relevant experience / role
  alignment" component without guessing. Inventing a heuristic for it
  would itself violate the "deterministic, explainable, no unsupported
  assumptions" rule this module operates under, so 'unclear'-importance
  requirements are folded into the lower-weight bucket instead of being
  used to fabricate a third dimension.

Each bucket's sub-score is the mean requirement credit in that bucket,
scaled to 0-100. The overall score is a weighted average of the two
sub-scores — required 70%, preferred/unclear 30% — reflecting that a job's
stated *required* qualifications should dominate the score while
*preferred* ones matter less. If a bucket has no requirements at all, its
weight is redistributed entirely to the other bucket; if there are no
requirements whatsoever, the score is 0.

SECURITY NOTE: This module only ever reads structured requirement/evidence
metadata (ids, statuses, requirement text) — it never touches raw CV or
job description text, and makes no LLM or network calls.
"""

from __future__ import annotations

from src.models.schemas import (
    AlignmentScore,
    EvidenceMatch,
    EvidenceStatus,
    JobRequirements,
    RequirementCoverage,
    RequirementImportance,
)

# Weights applied to the two sub-scores when both buckets have at least one
# requirement. Required qualifications dominate; preferred/unclear ones
# matter, but less. See the module docstring for why there is no third
# "role alignment" bucket.
REQUIRED_WEIGHT = 0.7
PREFERRED_WEIGHT = 0.3

# Per-requirement credit by evidence status. No status is ever penalized
# below 0.0 — "no_evidence_found" simply earns no credit.
_CREDIT_BY_STATUS = {
    EvidenceStatus.evidence_found: 1.0,
    EvidenceStatus.needs_verification: 0.5,
    EvidenceStatus.no_evidence_found: 0.0,
}

# Deterministic score -> label bands. Thresholds are inclusive lower bounds,
# checked in descending order.
_ALIGNMENT_BANDS: list[tuple[int, str]] = [
    (85, "Strong Alignment"),
    (60, "Moderate Alignment"),
    (35, "Limited Alignment"),
    (0, "Minimal Alignment"),
]


class AlignmentScorerError(Exception):
    """Raised for invalid usage of :class:`AlignmentScorer`.

    Reserved for programming errors (e.g. missing required arguments), not
    for ordinary scoring outcomes such as a requirement having no evidence.
    """


class AlignmentScorer:
    """Computes a deterministic candidate-to-role :class:`AlignmentScore`.

    Pure application logic — makes no LLM calls and has no external
    dependencies, so it never fails except on invalid input.
    """

    def score(
        self,
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
    ) -> AlignmentScore:
        """Compute the alignment score from a job's requirements and their evidence matches.

        Args:
            job_requirements: The job's structured requirements.
            evidence_matches: The Evidence Matcher's results, matched to
                requirements by ``requirement_id``.

        Returns:
            A fully-populated ``AlignmentScore``.

        Raises:
            AlignmentScorerError: If ``job_requirements`` or
                ``evidence_matches`` is not provided.
        """
        if job_requirements is None:
            raise AlignmentScorerError("job_requirements is required.")
        if evidence_matches is None:
            raise AlignmentScorerError("evidence_matches is required.")

        matches_by_requirement_id = {match.requirement_id: match for match in evidence_matches}

        required_credits: list[float] = []
        preferred_credits: list[float] = []
        coverage_counts = {
            level: {"total": 0, "evidence_found": 0, "no_evidence_found": 0, "needs_verification": 0}
            for level in RequirementImportance
        }

        for requirement in job_requirements.requirements:
            match = matches_by_requirement_id.get(requirement.id)
            # A requirement with no match at all (shouldn't normally happen,
            # since EvidenceMatcher produces one match per requirement) is
            # treated the same as "needs verification" — unknown, not absent.
            status = match.status if match is not None else EvidenceStatus.needs_verification
            credit = _CREDIT_BY_STATUS.get(status, 0.5)

            counts = coverage_counts[requirement.importance]
            counts["total"] += 1
            counts[status] = counts.get(status, 0) + 1

            if requirement.importance == RequirementImportance.required:
                required_credits.append(credit)
            else:
                # Preferred and unclear-importance requirements share the
                # lower-weight bucket — see the module docstring.
                preferred_credits.append(credit)

        required_score = self._mean_as_percentage(required_credits)
        preferred_score = self._mean_as_percentage(preferred_credits)

        required_weight, preferred_weight = self._effective_weights(
            has_required=bool(required_credits),
            has_preferred=bool(preferred_credits),
        )

        overall = required_weight * (required_score or 0.0) + preferred_weight * (preferred_score or 0.0)
        overall_score = round(max(0.0, min(100.0, overall)))

        coverage = [
            RequirementCoverage(importance=level, **coverage_counts[level])
            for level in RequirementImportance
            if coverage_counts[level]["total"] > 0
        ]

        return AlignmentScore(
            overall_score=overall_score,
            alignment_label=self._alignment_label(overall_score),
            required_score=required_score or 0.0,
            preferred_score=preferred_score or 0.0,
            required_weight=required_weight,
            preferred_weight=preferred_weight,
            coverage=coverage,
            methodology_note=self._methodology_note(required_weight, preferred_weight),
        )

    @staticmethod
    def _mean_as_percentage(credits: list[float]) -> float | None:
        """Mean of per-requirement credits, scaled to 0-100. None if empty."""
        if not credits:
            return None
        return round((sum(credits) / len(credits)) * 100, 1)

    @staticmethod
    def _effective_weights(*, has_required: bool, has_preferred: bool) -> tuple[float, float]:
        """Redistribute weight to whichever bucket(s) actually have requirements."""
        if has_required and has_preferred:
            return REQUIRED_WEIGHT, PREFERRED_WEIGHT
        if has_required:
            return 1.0, 0.0
        if has_preferred:
            return 0.0, 1.0
        return 0.0, 0.0

    @staticmethod
    def _alignment_label(overall_score: int) -> str:
        for threshold, label in _ALIGNMENT_BANDS:
            if overall_score >= threshold:
                return label
        return _ALIGNMENT_BANDS[-1][1]

    @staticmethod
    def _methodology_note(required_weight: float, preferred_weight: float) -> str:
        return (
            "Computed deterministically from evidence-match statuses, not by "
            "the LLM: each requirement earns 1.0 credit for 'evidence_found', "
            "0.5 for 'needs_verification', and 0.0 for 'no_evidence_found' "
            "(no credit, not a claim the candidate lacks the skill). "
            f"Required requirements are weighted {required_weight:.0%} of the "
            f"overall score; preferred/unclear-importance requirements are "
            f"weighted {preferred_weight:.0%}."
        )
