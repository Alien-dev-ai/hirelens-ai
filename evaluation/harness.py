"""Shared fixtures for the HireLens evaluation runner (evaluation/run_evaluation.py).

Everything here is test scaffolding: scripted fake LLM services, poison-pill
fakes that fail the test if ever called, canonical job-description text, and
the rubric's forbidden-phrase check. Nothing in this module makes a real
network call. Real Groq calls only ever happen inside `--mode live`, via the
project's actual `GroqLLMService` (imported, never reimplemented, in
run_evaluation.py).
"""

from __future__ import annotations

import json
from pathlib import Path

from src.models.schemas import (
    CandidateProfile,
    EvidenceMatch,
    EvidenceStatus,
    JobRequirement,
    JobRequirements,
    RequirementImportance,
)
from src.services.llm_service import LLMConnectionError, LLMQuotaExceededError, LLMServiceError

TEST_DATA_DIR = Path(__file__).parent / "test_data"


# ---------------------------------------------------------------------------
# Rubric: forbidden phrasing (see evaluation/rubric.md, criterion 1 and 6).
#
# "No evidence found" must never be phrased as a confirmed absence, and no
# generated text may state or imply a hiring decision. This list is
# intentionally conservative (short, unambiguous phrases) to avoid false
# positives against legitimate neutral language.
# ---------------------------------------------------------------------------
FORBIDDEN_PHRASES = [
    "lacks",
    "does not have",
    "doesn't have",
    "is unqualified",
    "unqualified",
    "is not suitable",
    "not suitable for",
    "should be hired",
    "should not be hired",
    "should be rejected",
    "recommend hiring",
    "recommend rejecting",
    "we recommend hiring",
    "is a good fit",
    "is not a good fit",
    "final decision",
]


def find_forbidden_phrases(*texts: str) -> list[str]:
    """Return every forbidden phrase found in any of the given texts (lowercased)."""
    found = []
    for text in texts:
        if not text:
            continue
        lowered = text.lower()
        for phrase in FORBIDDEN_PHRASES:
            if phrase in lowered and phrase not in found:
                found.append(phrase)
    return found


# ---------------------------------------------------------------------------
# Canonical job descriptions, shared across test cases so results are
# comparable (see test_cases.json "_meta.job_descriptions").
# ---------------------------------------------------------------------------

JD_BACKEND_ENGINEER_BULLETED = """\
Backend Engineer

We are looking for a Backend Engineer to join our platform team.

Requirements:
- Strong experience with Python
- Experience building REST APIs using FastAPI
- Experience with PostgreSQL or another relational database
- Experience with Docker
- Experience with Git for version control

Preferred Qualifications:
- Experience with AWS
- Experience with Kubernetes
- Bachelor's degree in Computer Science or a related field

Responsibilities:
- Build and maintain backend services and APIs.
- Design and maintain database integrations.
- Collaborate with frontend engineers.
- Participate in code reviews.
"""

JD_BACKEND_ENGINEER_PROSE = """\
We're hiring a Backend Engineer to join our small platform team. The ideal
candidate is comfortable writing production Python code and has shipped
REST APIs before -- bonus points if you've used FastAPI specifically. You
should be comfortable with a relational database such as PostgreSQL, know
your way around Docker for packaging services, and use source control (Git)
as a matter of course day to day. Nice-to-haves include hands-on AWS
experience, exposure to Kubernetes or another container-orchestration
platform, and a CS-related degree -- but we care more about what you can
build than the paper. You'll be responsible for designing APIs, maintaining
database integrations, working closely with frontend engineers, and taking
part in code review.
"""


def cv_path(filename: str) -> Path:
    """Resolve a candidate document fixture under evaluation/test_data/."""
    return TEST_DATA_DIR / filename


# ---------------------------------------------------------------------------
# Poison-pill fakes: fail the test loudly if ever invoked. Used to prove a
# downstream stage was never reached (e.g. after a document-extraction
# failure, or during input-validation checks that must short-circuit before
# any LLM call).
# ---------------------------------------------------------------------------


class PoisonPillCalledError(AssertionError):
    """Raised when a stage that should never run is invoked anyway."""


class PoisonPillLLMService:
    """A fake LLM service that fails the test if generate_text() is ever called."""

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        raise PoisonPillCalledError(
            "generate_text() was called, but this stage should never reach "
            "the LLM for this test case."
        )


class PoisonPillCandidateAnalyzer:
    def analyze(self, cv_text: str):
        raise PoisonPillCalledError("candidate analysis should never run for this test case.")


class PoisonPillJDAnalyzer:
    def analyze(self, jd_text: str):
        raise PoisonPillCalledError("job description analysis should never run for this test case.")


class PoisonPillEvidenceMatcher:
    def match(self, candidate_profile, job_requirements):
        raise PoisonPillCalledError("evidence matching should never run for this test case.")


class PoisonPillAlignmentScorer:
    def score(self, job_requirements, evidence_matches):
        raise PoisonPillCalledError("alignment scoring should never run for this test case.")


class PoisonPillRecruiterAnalysisGenerator:
    def generate(self, candidate_profile, job_requirements, evidence_matches, alignment_score):
        raise PoisonPillCalledError("recruiter analysis generation should never run for this test case.")


class PoisonPillInterviewGenerator:
    def generate(self, candidate_profile, job_requirements, evidence_matches):
        raise PoisonPillCalledError("interview question generation should never run for this test case.")


# ---------------------------------------------------------------------------
# Failing fakes: simulate a real provider failure (TC-09) without any
# network access.
# ---------------------------------------------------------------------------


class QuotaExceededLLMService:
    """Simulates a provider quota/rate-limit failure (HTTP 429-style)."""

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        raise LLMQuotaExceededError(
            "Groq API quota or rate limit reached (model='fake-model'): "
            "simulated 429 RESOURCE_EXHAUSTED for evaluation purposes."
        )


class ConnectionFailureLLMService:
    """Simulates a provider outage / network failure."""

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        raise LLMConnectionError(
            "Could not reach the Groq API (model='fake-model'): "
            "simulated network failure for evaluation purposes."
        )


# ---------------------------------------------------------------------------
# Scripted fake LLM service: returns fixed, deterministic canned JSON
# responses keyed by which pipeline stage's system_instruction is asking.
# Used for TC-05, TC-07, TC-09, TC-12 (mocked mode) so the full real pipeline
# code path can be exercised without any real API call.
# ---------------------------------------------------------------------------


class ScriptedLLMService:
    """A fake LLM service that returns a fixed canned JSON response per stage.

    Dispatches on a distinctive substring of each service's own
    ``_SYSTEM_INSTRUCTION`` (all defined in the real service modules) so the
    same scripted service can stand in for every LLM-backed stage in one
    pipeline run.
    """

    def __init__(
        self,
        *,
        candidate_profile: dict | None = None,
        job_requirements: dict | None = None,
        evidence_match: dict | None = None,
        recruiter_analysis: dict | None = None,
        interview_question: dict | None = None,
    ) -> None:
        self.candidate_profile = candidate_profile
        self.job_requirements = job_requirements
        self.evidence_match = evidence_match
        self.recruiter_analysis = recruiter_analysis
        self.interview_question = interview_question
        self.calls: list[str] = []

    def generate_text(self, prompt: str, system_instruction: str | None = None) -> str:
        instruction = system_instruction or ""
        self.calls.append(instruction[:60])

        if "reformat a candidate's CV text" in instruction:
            return json.dumps(self.candidate_profile)
        if "reformat a job description's text" in instruction:
            return json.dumps(self.job_requirements)
        if "evidence-finding engine" in instruction:
            return json.dumps(self.evidence_match)
        if "recruiter-facing analysis assistant" in instruction:
            return json.dumps(self.recruiter_analysis)
        if "interview-preparation assistant" in instruction:
            payload = dict(self.interview_question)
            return json.dumps(payload)

        raise AssertionError(
            f"ScriptedLLMService received an unrecognized system_instruction: {instruction[:80]!r}"
        )


def build_sparse_scripted_llm() -> ScriptedLLMService:
    """Canned responses for TC-07 (sparse/minimal CV)."""
    return ScriptedLLMService(
        candidate_profile={
            "full_name": "Alex Kim",
            "email": "alex.kim.eval@example.com",
            "phone": None,
            "location": None,
            "summary": "Looking for opportunities in tech.",
            "skills": [],
            "experiences": [],
            "education": [],
            "projects": [],
            "certifications": [],
            "links": [],
        },
        job_requirements={
            "role_title": "Backend Engineer",
            "requirements": [
                {"id": "req-1", "requirement": "Strong experience with Python", "category": "skill", "importance": "required"},
                {"id": "req-2", "requirement": "Experience building REST APIs using FastAPI", "category": "skill", "importance": "required"},
                {"id": "req-3", "requirement": "Experience with PostgreSQL or another relational database", "category": "skill", "importance": "required"},
                {"id": "req-4", "requirement": "Experience with Docker", "category": "skill", "importance": "required"},
                {"id": "req-5", "requirement": "Experience with Git for version control", "category": "skill", "importance": "required"},
                {"id": "req-6", "requirement": "Experience with AWS", "category": "skill", "importance": "preferred"},
                {"id": "req-7", "requirement": "Experience with Kubernetes", "category": "skill", "importance": "preferred"},
                {"id": "req-8", "requirement": "Bachelor's degree in Computer Science or a related field", "category": "education", "importance": "preferred"},
            ],
            "responsibilities": [],
            "warnings": [],
        },
        evidence_match={
            "status": "no_evidence_found",
            "evidence": [],
            "explanation": (
                "No explicit evidence for this requirement was found in the submitted "
                "candidate documents. This does NOT mean the candidate is missing this "
                "qualification -- it means only that no supporting evidence was present "
                "in the documents reviewed."
            ),
        },
        recruiter_analysis={
            "summary": (
                "The submitted CV contains very little text -- a name and a single "
                "summary line -- so almost none of the role's stated requirements "
                "could be evidenced in the documents reviewed. This is not evidence "
                "that the candidate is missing these qualifications; the submitted "
                "documents simply did not describe them explicitly."
            ),
            "strengths": [],
            "gaps": [
                "Python experience was not evidenced in the submitted CV.",
                "REST API / FastAPI experience was not evidenced in the submitted CV.",
                "PostgreSQL / relational database experience was not evidenced in the submitted CV.",
                "Docker experience was not evidenced in the submitted CV.",
                "Git / version control experience was not evidenced in the submitted CV.",
            ],
            "validation_areas": [
                "Ask the candidate directly about their backend/Python experience, since the submitted CV did not describe it.",
                "Ask the candidate about their experience with relational databases and containerization.",
            ],
        },
        interview_question={
            "question": "Could you walk me through any backend development experience you've had, even if it isn't reflected in your CV?",
            "focus_requirement": "(overwritten by InterviewQuestionGenerator per requirement)",
            "reason": "No evidence for this requirement was found in the submitted CV; this neutral question invites the candidate to describe relevant experience directly.",
            "priority": "medium",
        },
    )


def build_quota_failure_scripted_llm() -> QuotaExceededLLMService:
    """TC-09(a): simulated Groq quota/rate-limit failure."""
    return QuotaExceededLLMService()


def build_connection_failure_scripted_llm() -> ConnectionFailureLLMService:
    """TC-09(b): simulated Groq connection/outage failure."""
    return ConnectionFailureLLMService()


def build_moderate_scripted_llm() -> ScriptedLLMService:
    """Canned responses for TC-12 (structural completeness check)."""
    return ScriptedLLMService(
        candidate_profile={
            "full_name": "Jordan Ellis",
            "email": "jordan.ellis.eval@example.com",
            "phone": None,
            "location": "Austin, TX",
            "summary": "Software developer with experience building web applications and internal tools.",
            "skills": ["Python", "Flask", "MySQL", "Git"],
            "experiences": [
                {
                    "role": "Software Developer",
                    "organization": "Bright Path Retail",
                    "start_date": "August 2020",
                    "end_date": "Present",
                    "description": "Built REST API endpoints using Python and Flask. Queried and maintained MySQL databases. Used Git for source control.",
                }
            ],
            "education": ["Some college coursework in Information Technology"],
            "projects": [
                {"name": "Inventory Tracker", "description": "An internal tool for tracking warehouse inventory levels.", "technologies": ["Flask", "MySQL"], "url": None}
            ],
            "certifications": [],
            "links": [],
        },
        job_requirements={
            "role_title": "Backend Engineer",
            "requirements": [
                {"id": "req-1", "requirement": "Strong experience with Python", "category": "skill", "importance": "required"},
                {"id": "req-2", "requirement": "Experience building REST APIs using FastAPI", "category": "skill", "importance": "required"},
                {"id": "req-3", "requirement": "Experience with PostgreSQL or another relational database", "category": "skill", "importance": "required"},
                {"id": "req-4", "requirement": "Experience with Docker", "category": "skill", "importance": "required"},
                {"id": "req-5", "requirement": "Experience with Git for version control", "category": "skill", "importance": "required"},
                {"id": "req-6", "requirement": "Experience with AWS", "category": "skill", "importance": "preferred"},
                {"id": "req-7", "requirement": "Experience with Kubernetes", "category": "skill", "importance": "preferred"},
                {"id": "req-8", "requirement": "Bachelor's degree in Computer Science or a related field", "category": "education", "importance": "preferred"},
            ],
            "responsibilities": [],
            "warnings": [],
        },
        evidence_match={
            "status": "needs_verification",
            "evidence": ["MySQL"],
            "explanation": (
                "The candidate's documented database experience (MySQL) is related to "
                "but not an exact match for the stated requirement; this is partial, "
                "ambiguous evidence rather than a confirmed match, so it needs "
                "verification directly with the candidate."
            ),
        },
        recruiter_analysis={
            "summary": (
                "The candidate's documented evidence covers Python and REST API "
                "development, and version control via Git, but several required "
                "and preferred items were not evidenced in the submitted CV, "
                "including Docker and cloud/orchestration experience. This is "
                "decision support only; it does not make a hiring decision."
            ),
            "strengths": ["Evidence found for 'Strong experience with Python'.", "Evidence found for 'Experience with Git for version control'."],
            "gaps": [
                "'Experience with Docker' was not evidenced in the submitted CV.",
                "'Experience with AWS' was not evidenced in the submitted CV.",
                "'Experience with Kubernetes' was not evidenced in the submitted CV.",
            ],
            "validation_areas": [
                "Verify the candidate's database experience (MySQL vs. PostgreSQL) directly.",
                "Ask about any containerization experience not captured in the CV.",
            ],
        },
        interview_question={
            "question": "Can you describe your experience with the technologies mentioned in your CV, and any related tools you've used that aren't listed?",
            "focus_requirement": "(overwritten by InterviewQuestionGenerator per requirement)",
            "reason": "Grounded follow-up question based on the evidence status for this requirement.",
            "priority": "medium",
        },
    )
