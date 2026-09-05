"""Core data models for HireLens AI.

These Pydantic models define the shared data contracts used across the
document extraction, evidence-matching, alignment-scoring, and
dossier-generation stages of the pipeline.

PRODUCT DIRECTION:
HireLens is a recruiter-facing candidate analysis and *decision-support*
system: it computes a deterministic alignment score and a grounded,
LLM-synthesized recruiter analysis (see ``AlignmentScore`` and
``RecruiterAnalysis`` below), in addition to the underlying evidence data.

IMPORTANT BOUNDARIES that still apply, unchanged, to everything in this
module:
- Nothing here is, or claims to be, an objective or certain hiring
  decision. ``AlignmentScore`` and ``RecruiterAnalysis`` support a human
  recruiter's own judgment; they never replace it, and no field here
  represents an accept/reject/hire outcome.
- ``EvidenceStatus.no_evidence_found`` means no explicit supporting
  evidence was located in the submitted document(s) — it must never be
  interpreted, anywhere downstream, as proof that a candidate lacks a
  given skill. Every gap-facing field must be phrased as "not evidenced",
  never as a confirmed absence.
- Scoring and analysis must never use or infer protected characteristics
  (race, ethnicity, religion, gender, age, disability, marital status,
  nationality, appearance, health, or similar) — nothing in these models
  carries such information in the first place.
"""

from __future__ import annotations

from datetime import datetime
from enum import Enum

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Document extraction
# ---------------------------------------------------------------------------


class ExtractedDocument(BaseModel):
    """Raw text extracted from a single uploaded candidate document."""

    filename: str = Field(..., description="Original filename of the uploaded document.")
    file_type: str = Field(..., description="File type/extension of the document, e.g. 'pdf', 'docx'.")
    text: str = Field(..., description="Raw text extracted from the document.")
    extraction_success: bool = Field(
        ..., description="Whether text extraction completed successfully."
    )
    error_message: str | None = Field(
        default=None, description="Details of the extraction error, if extraction failed."
    )


# ---------------------------------------------------------------------------
# Candidate profile
# ---------------------------------------------------------------------------


class CandidateExperience(BaseModel):
    """A single work experience entry parsed from a candidate document."""

    role: str | None = Field(
        default=None, description="Job title or role held by the candidate."
    )
    organization: str | None = Field(
        default=None, description="Name of the employer or organization."
    )
    start_date: str | None = Field(
        default=None, description="Start date of the role, as stated in the source document."
    )
    end_date: str | None = Field(
        default=None, description="End date of the role, as stated in the source document."
    )
    description: str | None = Field(
        default=None, description="Free-text description of responsibilities or achievements."
    )


class CandidateProject(BaseModel):
    """A project listed by the candidate."""

    name: str = Field(..., description="Name or title of the project.")
    description: str | None = Field(default=None, description="Free-text description of the project.")
    technologies: list[str] = Field(
        default_factory=list, description="Technologies, tools, or languages used in the project."
    )
    url: str | None = Field(default=None, description="Link to the project, if provided.")


class CandidateProfile(BaseModel):
    """Structured profile information parsed from a candidate's documents."""

    full_name: str | None = Field(default=None, description="Candidate's full name.")
    email: str | None = Field(default=None, description="Candidate's email address.")
    phone: str | None = Field(default=None, description="Candidate's phone number.")
    location: str | None = Field(default=None, description="Candidate's stated location.")
    summary: str | None = Field(
        default=None, description="Candidate's professional summary or objective statement."
    )
    skills: list[str] = Field(default_factory=list, description="Skills listed or extracted from the candidate.")
    experiences: list[CandidateExperience] = Field(
        default_factory=list, description="Work experience entries."
    )
    education: list[str] = Field(default_factory=list, description="Education entries, as free-text strings.")
    projects: list[CandidateProject] = Field(default_factory=list, description="Projects listed by the candidate.")
    certifications: list[str] = Field(default_factory=list, description="Certifications held by the candidate.")
    links: list[str] = Field(
        default_factory=list, description="External links, e.g. portfolio, GitHub, LinkedIn."
    )


# ---------------------------------------------------------------------------
# Job requirements
# ---------------------------------------------------------------------------


class RequirementImportance(str, Enum):
    """How strongly a job requirement is stated in the job description."""

    required = "required"
    preferred = "preferred"
    unclear = "unclear"


class JobRequirement(BaseModel):
    """A single requirement extracted from a job description."""

    id: str = Field(..., description="Stable identifier for this requirement.")
    requirement: str = Field(..., description="Text describing the requirement.")
    category: str = Field(..., description="Category of the requirement, e.g. 'skill', 'experience', 'education'.")
    importance: RequirementImportance = Field(
        ..., description="Whether the requirement is required, preferred, or unclear from the job description."
    )


class JobRequirements(BaseModel):
    """The full set of requirements extracted from a job description."""

    role_title: str | None = Field(default=None, description="Title of the role, if stated.")
    requirements: list[JobRequirement] = Field(
        default_factory=list, description="Individual requirements extracted from the job description."
    )
    responsibilities: list[str] = Field(
        default_factory=list, description="Responsibilities listed in the job description."
    )
    warnings: list[str] = Field(
        default_factory=list,
        description="Notes about ambiguities, gaps, or issues encountered while parsing the job description.",
    )


# ---------------------------------------------------------------------------
# Evidence matching
# ---------------------------------------------------------------------------


class EvidenceStatus(str, Enum):
    """Status of evidence found for a given job requirement.

    IMPORTANT: ``no_evidence_found`` means no explicit supporting evidence was
    identified in the submitted document(s). It must never be interpreted as
    proof that the candidate lacks that skill or qualification.
    """

    evidence_found = "evidence_found"
    no_evidence_found = "no_evidence_found"
    needs_verification = "needs_verification"


class EvidenceMatch(BaseModel):
    """The result of matching a single job requirement against candidate evidence."""

    requirement_id: str = Field(..., description="ID of the JobRequirement this match refers to.")
    requirement: str = Field(..., description="Text of the requirement being matched.")
    status: EvidenceStatus = Field(..., description="Evidence status for this requirement.")
    evidence: list[str] = Field(
        default_factory=list,
        description="Excerpts from candidate documents supporting this match, if any.",
    )
    explanation: str = Field(
        ..., description="Human-readable explanation of why this status was assigned."
    )
    confidence: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Confidence score between 0 and 1 for this evidence match.",
    )


# ---------------------------------------------------------------------------
# Alignment scoring
# ---------------------------------------------------------------------------


class RequirementCoverage(BaseModel):
    """Deterministic evidence-coverage counts for one requirement-importance level.

    A plain tally of the ``EvidenceMatch`` statuses already produced by the
    Evidence Matcher, grouped by whether the job description marked the
    requirement as required, preferred, or unclear. Carries no judgment of
    its own about the candidate — see ``AlignmentScorer``.
    """

    importance: RequirementImportance = Field(
        ..., description="The requirement-importance level this coverage row summarizes."
    )
    total: int = Field(..., ge=0, description="Total requirements at this importance level.")
    evidence_found: int = Field(..., ge=0, description="Requirements with status='evidence_found'.")
    no_evidence_found: int = Field(..., ge=0, description="Requirements with status='no_evidence_found'.")
    needs_verification: int = Field(..., ge=0, description="Requirements with status='needs_verification'.")


class AlignmentScore(BaseModel):
    """A deterministic, code-computed candidate-to-role alignment score.

    IMPORTANT: This score is decision *support*, not a hiring decision. It
    represents how strongly the candidate's documented evidence covers the
    job's stated requirements — nothing more, and no claim of certainty or
    objective hiring probability. It is computed entirely by application
    logic (``src.services.alignment_scorer.AlignmentScorer``) from existing
    ``EvidenceMatch`` statuses; the LLM is never asked to invent or adjust
    this number. Missing evidence is never treated as confirmed absence of
    a skill: a zero-evidence requirement earns no credit because there is
    nothing to credit, which is different from asserting the candidate
    lacks it.
    """

    overall_score: int = Field(
        ..., ge=0, le=100, description="Overall alignment score from 0-100, computed deterministically."
    )
    alignment_label: str = Field(
        ..., description="Human-readable band for overall_score, e.g. 'Strong Alignment'."
    )
    required_score: float = Field(
        ..., ge=0.0, le=100.0, description="Sub-score (0-100) for required requirements only."
    )
    preferred_score: float = Field(
        ...,
        ge=0.0,
        le=100.0,
        description=(
            "Sub-score (0-100) for non-required requirements (preferred and "
            "unclear-importance combined — see AlignmentScorer for why)."
        ),
    )
    required_weight: float = Field(
        ..., ge=0.0, le=1.0, description="Weight given to required_score in the overall score."
    )
    preferred_weight: float = Field(
        ..., ge=0.0, le=1.0, description="Weight given to preferred_score in the overall score."
    )
    coverage: list[RequirementCoverage] = Field(
        default_factory=list, description="Evidence-coverage breakdown per importance level."
    )
    methodology_note: str = Field(
        ..., description="Fixed, human-readable explanation of how overall_score was computed."
    )


# ---------------------------------------------------------------------------
# Interview questions
# ---------------------------------------------------------------------------


class InterviewQuestion(BaseModel):
    """A suggested interview question to help a human verify a requirement."""

    question: str = Field(..., description="The suggested interview question text.")
    focus_requirement: str = Field(
        ..., description="The requirement this question is intended to help verify."
    )
    reason: str = Field(
        ..., description="Why this question is suggested, e.g. to clarify missing or ambiguous evidence."
    )
    priority: str = Field(
        ..., description="Suggested priority for asking this question, e.g. 'high', 'medium', 'low'."
    )


# ---------------------------------------------------------------------------
# Recruiter analysis
# ---------------------------------------------------------------------------


class RecruiterAnalysis(BaseModel):
    """An LLM-synthesized, evidence-grounded narrative analysis for a recruiter.

    IMPORTANT: This is a synthesis of evidence already gathered by the
    Evidence Matcher and the deterministic ``AlignmentScore`` — generated
    under strict grounding rules that forbid inventing skills, employers,
    education, experience, or certifications, and forbid treating "no
    evidence found" as proof of absence (gaps must read as "not evidenced
    in the submitted CV", never "does not have"/"lacks"). It supports a
    human recruiter's judgment; it is never itself a hiring decision, and
    it must never reference protected characteristics (race, ethnicity,
    religion, gender, age, disability, marital status, nationality,
    appearance, health, or similar).
    """

    summary: str = Field(
        ..., description="Detailed natural-language synthesis of how the candidate's evidence aligns with the role."
    )
    strengths: list[str] = Field(
        default_factory=list, description="Candidate strengths grounded in explicit evidence."
    )
    gaps: list[str] = Field(
        default_factory=list,
        description=(
            "Requirements not evidenced in the submitted documents, phrased as "
            "'not evidenced', never as a confirmed absence."
        ),
    )
    validation_areas: list[str] = Field(
        default_factory=list,
        description=(
            "The most important things a recruiter should validate directly "
            "with the candidate, prioritized by requirement importance and "
            "evidence gaps."
        ),
    )


# ---------------------------------------------------------------------------
# Candidate dossier
# ---------------------------------------------------------------------------


class CandidateDossier(BaseModel):
    """The final recruiter-facing dossier generated for a candidate.

    Combines the underlying evidence (candidate profile, job requirements,
    per-requirement evidence matches) with a deterministic alignment score
    and an LLM-synthesized recruiter analysis, plus suggested interview
    questions to help a human recruiter verify unclear or missing areas.

    IMPORTANT: ``alignment_score`` and ``recruiter_analysis`` are decision
    *support* — a structured, evidence-grounded starting point for a human
    recruiter's own judgment. Neither is, or should be treated as, an
    objective or certain hiring decision; the human recruiter always makes
    the final call.
    """

    candidate_profile: CandidateProfile = Field(..., description="Structured candidate profile.")
    job_requirements: JobRequirements = Field(..., description="Requirements the candidate is being evaluated against.")
    evidence_matches: list[EvidenceMatch] = Field(
        default_factory=list, description="Evidence matches for each job requirement."
    )
    alignment_score: AlignmentScore = Field(
        ..., description="Deterministic candidate-to-role alignment score, computed from evidence_matches."
    )
    recruiter_analysis: RecruiterAnalysis = Field(
        ..., description="LLM-synthesized, evidence-grounded recruiter analysis."
    )
    interview_questions: list[InterviewQuestion] = Field(
        default_factory=list, description="Suggested interview questions for human follow-up."
    )
    warnings: list[str] = Field(
        default_factory=list, description="Notes about ambiguities, gaps, or issues encountered while generating the dossier."
    )
    generated_at: datetime = Field(
        default_factory=datetime.utcnow, description="Timestamp when this dossier was generated."
    )
