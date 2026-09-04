"""Core data models for HireLens AI.

These Pydantic models define the shared data contracts used across the
document extraction, evidence-matching, and dossier-generation stages of
the pipeline.

IMPORTANT PRODUCT BOUNDARY:
These models represent *evidence extracted from submitted documents only*.
Nothing here should be read as a hiring recommendation, suitability score,
or ranking. In particular, ``EvidenceStatus.no_evidence_found`` means no
explicit supporting evidence was located in the submitted document(s) — it
must never be interpreted as proof that a candidate lacks a given skill.
All hiring decisions remain with human recruiters and hiring managers.
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

    role: str = Field(..., description="Job title or role held by the candidate.")
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
# Candidate dossier
# ---------------------------------------------------------------------------


class CandidateDossier(BaseModel):
    """The final evidence-based dossier generated for a candidate.

    This dossier summarizes evidence found in the candidate's submitted
    documents relative to a job's requirements, and suggests interview
    questions to help a human recruiter verify unclear or missing areas.
    It contains no hiring recommendation, ranking, or suitability judgment.
    """

    candidate_profile: CandidateProfile = Field(..., description="Structured candidate profile.")
    job_requirements: JobRequirements = Field(..., description="Requirements the candidate is being evaluated against.")
    evidence_matches: list[EvidenceMatch] = Field(
        default_factory=list, description="Evidence matches for each job requirement."
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
