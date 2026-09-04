"""Tests for src/services/hirelens_pipeline.py.

These tests never make real Groq API calls: every sub-service and the
document extractor are injected as fakes/mocks — including plain,
duck-typed objects that are not instances of the real service classes at
all, demonstrating that the pipeline depends only on the service
interfaces, never on the Groq SDK or any concrete LLM-backed
implementation.
"""

from __future__ import annotations

import ast
import inspect

import pytest

from src.models.schemas import (
    AlignmentScore,
    CandidateDossier,
    CandidateProfile,
    EvidenceMatch,
    EvidenceStatus,
    ExtractedDocument,
    InterviewQuestion,
    JobRequirement,
    JobRequirements,
    RecruiterAnalysis,
    RequirementCoverage,
    RequirementImportance,
)
from src.services.alignment_scorer import AlignmentScorerError
from src.services.candidate_analyzer import CandidateAnalyzerError
from src.services.evidence_matcher import EvidenceMatcherError
from src.services.interview_generator import InterviewQuestionGeneratorError
from src.services.jd_analyzer import JobDescriptionAnalyzerError
from src.services.recruiter_analysis_generator import RecruiterAnalysisGeneratorError
import src.services.hirelens_pipeline as hirelens_pipeline_module
from src.services.hirelens_pipeline import HireLensPipeline, HireLensPipelineError


# --- Fakes -------------------------------------------------------------------


class _FakeCandidateAnalyzer:
    def __init__(self, profile: CandidateProfile | None = None, error: Exception | None = None):
        self.profile = profile
        self.error = error
        self.calls: list[str] = []

    def analyze(self, cv_text: str) -> CandidateProfile:
        self.calls.append(cv_text)
        if self.error is not None:
            raise self.error
        return self.profile


class _FakeJDAnalyzer:
    def __init__(self, requirements: JobRequirements | None = None, error: Exception | None = None):
        self.requirements = requirements
        self.error = error
        self.calls: list[str] = []

    def analyze(self, jd_text: str) -> JobRequirements:
        self.calls.append(jd_text)
        if self.error is not None:
            raise self.error
        return self.requirements


class _FakeEvidenceMatcher:
    def __init__(self, matches: list[EvidenceMatch] | None = None, error: Exception | None = None):
        self.matches = matches or []
        self.error = error
        self.calls: list[tuple] = []

    def match(self, candidate_profile: CandidateProfile, job_requirements: JobRequirements) -> list[EvidenceMatch]:
        self.calls.append((candidate_profile, job_requirements))
        if self.error is not None:
            raise self.error
        return self.matches


class _FakeAlignmentScorer:
    def __init__(self, alignment_score: AlignmentScore | None = None, error: Exception | None = None):
        self.alignment_score = alignment_score
        self.error = error
        self.calls: list[tuple] = []

    def score(self, job_requirements: JobRequirements, evidence_matches: list[EvidenceMatch]) -> AlignmentScore:
        self.calls.append((job_requirements, evidence_matches))
        if self.error is not None:
            raise self.error
        return self.alignment_score


class _FakeRecruiterAnalysisGenerator:
    def __init__(self, recruiter_analysis: RecruiterAnalysis | None = None, error: Exception | None = None):
        self.recruiter_analysis = recruiter_analysis
        self.error = error
        self.calls: list[tuple] = []

    def generate(
        self,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
        alignment_score: AlignmentScore,
    ) -> RecruiterAnalysis:
        self.calls.append((candidate_profile, job_requirements, evidence_matches, alignment_score))
        if self.error is not None:
            raise self.error
        return self.recruiter_analysis


class _FakeInterviewGenerator:
    def __init__(self, questions: list[InterviewQuestion] | None = None, error: Exception | None = None):
        self.questions = questions or []
        self.error = error
        self.calls: list[tuple] = []

    def generate(self, candidate_profile, job_requirements, evidence_matches) -> list[InterviewQuestion]:
        self.calls.append((candidate_profile, job_requirements, evidence_matches))
        if self.error is not None:
            raise self.error
        return self.questions


def _fake_extract_document_fn(text: str = "Jordan Rivera\nExperienced Python developer."):
    def _fn(path):
        return ExtractedDocument(
            filename="candidate.pdf",
            file_type="pdf",
            text=text,
            extraction_success=True,
            error_message=None,
        )

    return _fn


def _fake_failing_extract_document_fn(error_message: str = "File not found: candidate.pdf"):
    def _fn(path):
        return ExtractedDocument(
            filename="candidate.pdf",
            file_type="pdf",
            text="",
            extraction_success=False,
            error_message=error_message,
        )

    return _fn


# --- Shared fixtures data ------------------------------------------------

CANDIDATE_PROFILE = CandidateProfile(full_name="Jordan Rivera", skills=["Python"])
JOB_REQUIREMENTS = JobRequirements(
    role_title="Backend Engineer",
    requirements=[
        JobRequirement(id="req-1", requirement="Python", category="skill", importance=RequirementImportance.required)
    ],
    warnings=["Minimum years of experience not specified."],
)
EVIDENCE_MATCHES = [
    EvidenceMatch(
        requirement_id="req-1",
        requirement="Python",
        status=EvidenceStatus.evidence_found,
        evidence=["Python"],
        explanation="Found in skills.",
        confidence=0.9,
    )
]
ALIGNMENT_SCORE = AlignmentScore(
    overall_score=100,
    alignment_label="Strong Alignment",
    required_score=100.0,
    preferred_score=0.0,
    required_weight=1.0,
    preferred_weight=0.0,
    coverage=[
        RequirementCoverage(
            importance=RequirementImportance.required,
            total=1,
            evidence_found=1,
            no_evidence_found=0,
            needs_verification=0,
        )
    ],
    methodology_note="Computed deterministically from evidence-match statuses.",
)
RECRUITER_ANALYSIS = RecruiterAnalysis(
    summary="Strong alignment with the core requirements.",
    strengths=["Evidence found for 'Python'."],
    gaps=[],
    validation_areas=[],
)
INTERVIEW_QUESTIONS = [
    InterviewQuestion(
        question="Can you walk me through your Python experience?",
        focus_requirement="Python",
        reason="Evidence found; explore in depth.",
        priority="medium",
    )
]


def _build_pipeline(
    profile=CANDIDATE_PROFILE,
    requirements=JOB_REQUIREMENTS,
    matches=EVIDENCE_MATCHES,
    alignment_score=ALIGNMENT_SCORE,
    recruiter_analysis=RECRUITER_ANALYSIS,
    questions=INTERVIEW_QUESTIONS,
    candidate_error=None,
    jd_error=None,
    matcher_error=None,
    scorer_error=None,
    analysis_error=None,
    generator_error=None,
    extract_fn=None,
):
    candidate_analyzer = _FakeCandidateAnalyzer(profile=profile, error=candidate_error)
    jd_analyzer = _FakeJDAnalyzer(requirements=requirements, error=jd_error)
    evidence_matcher = _FakeEvidenceMatcher(matches=matches, error=matcher_error)
    alignment_scorer = _FakeAlignmentScorer(alignment_score=alignment_score, error=scorer_error)
    recruiter_analysis_generator = _FakeRecruiterAnalysisGenerator(
        recruiter_analysis=recruiter_analysis, error=analysis_error
    )
    interview_generator = _FakeInterviewGenerator(questions=questions, error=generator_error)

    pipeline = HireLensPipeline(
        candidate_analyzer=candidate_analyzer,
        jd_analyzer=jd_analyzer,
        evidence_matcher=evidence_matcher,
        alignment_scorer=alignment_scorer,
        recruiter_analysis_generator=recruiter_analysis_generator,
        interview_generator=interview_generator,
        extract_document_fn=extract_fn or _fake_extract_document_fn(),
    )
    return (
        pipeline,
        candidate_analyzer,
        jd_analyzer,
        evidence_matcher,
        alignment_scorer,
        recruiter_analysis_generator,
        interview_generator,
    )


# --- Tests ---------------------------------------------------------------


def test_successful_end_to_end_pipeline_execution():
    """A full run with all fakes succeeding should return a valid CandidateDossier."""
    pipeline, *_ = _build_pipeline()

    dossier = pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert isinstance(dossier, CandidateDossier)
    assert dossier.candidate_profile == CANDIDATE_PROFILE
    assert dossier.job_requirements == JOB_REQUIREMENTS
    assert dossier.evidence_matches == EVIDENCE_MATCHES
    assert dossier.alignment_score == ALIGNMENT_SCORE
    assert dossier.recruiter_analysis == RECRUITER_ANALYSIS
    assert dossier.interview_questions == INTERVIEW_QUESTIONS


def test_document_extraction_failure_raises_pipeline_error():
    """A failed document extraction should stop the pipeline with a clear error."""
    (
        pipeline,
        candidate_analyzer,
        jd_analyzer,
        evidence_matcher,
        alignment_scorer,
        recruiter_analysis_generator,
        interview_generator,
    ) = _build_pipeline(
        extract_fn=_fake_failing_extract_document_fn("File not found: candidate.pdf"),
    )

    with pytest.raises(HireLensPipelineError, match="extraction"):
        pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    # Downstream steps must never run after an extraction failure.
    assert candidate_analyzer.calls == []
    assert jd_analyzer.calls == []
    assert evidence_matcher.calls == []
    assert alignment_scorer.calls == []
    assert recruiter_analysis_generator.calls == []
    assert interview_generator.calls == []


def test_candidate_analyzer_failure_raises_pipeline_error_with_cause():
    original_error = CandidateAnalyzerError("The LLM's response did not match the expected structure.")
    pipeline, *_ = _build_pipeline(candidate_error=original_error)

    with pytest.raises(HireLensPipelineError) as exc_info:
        pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert "candidate analysis" in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is original_error


def test_job_description_analyzer_failure_raises_pipeline_error_with_cause():
    original_error = JobDescriptionAnalyzerError("Cannot analyze empty job description text.")
    (
        pipeline,
        candidate_analyzer,
        jd_analyzer,
        evidence_matcher,
        alignment_scorer,
        recruiter_analysis_generator,
        interview_generator,
    ) = _build_pipeline(jd_error=original_error)

    with pytest.raises(HireLensPipelineError) as exc_info:
        pipeline.analyze("candidate.pdf", "")

    assert "job description" in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is original_error
    # Candidate analysis runs before JD analysis, so it should have been called.
    assert len(candidate_analyzer.calls) == 1
    # Everything downstream of JD analysis must never run after this failure.
    assert evidence_matcher.calls == []
    assert alignment_scorer.calls == []
    assert recruiter_analysis_generator.calls == []
    assert interview_generator.calls == []


def test_evidence_matcher_failure_raises_pipeline_error_with_cause():
    original_error = EvidenceMatcherError("candidate_profile is required.")
    (
        pipeline,
        candidate_analyzer,
        jd_analyzer,
        evidence_matcher,
        alignment_scorer,
        recruiter_analysis_generator,
        interview_generator,
    ) = _build_pipeline(matcher_error=original_error)

    with pytest.raises(HireLensPipelineError) as exc_info:
        pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert "evidence matching" in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is original_error
    # Everything downstream of evidence matching must never run after this failure.
    assert alignment_scorer.calls == []
    assert recruiter_analysis_generator.calls == []
    assert interview_generator.calls == []


def test_alignment_scorer_failure_raises_pipeline_error_with_cause():
    original_error = AlignmentScorerError("job_requirements is required.")
    (
        pipeline,
        candidate_analyzer,
        jd_analyzer,
        evidence_matcher,
        alignment_scorer,
        recruiter_analysis_generator,
        interview_generator,
    ) = _build_pipeline(scorer_error=original_error)

    with pytest.raises(HireLensPipelineError) as exc_info:
        pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert "alignment scoring" in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is original_error
    # Everything downstream of alignment scoring must never run after this failure.
    assert recruiter_analysis_generator.calls == []
    assert interview_generator.calls == []


def test_recruiter_analysis_generator_failure_raises_pipeline_error_with_cause():
    original_error = RecruiterAnalysisGeneratorError("alignment_score is required.")
    (
        pipeline,
        candidate_analyzer,
        jd_analyzer,
        evidence_matcher,
        alignment_scorer,
        recruiter_analysis_generator,
        interview_generator,
    ) = _build_pipeline(analysis_error=original_error)

    with pytest.raises(HireLensPipelineError) as exc_info:
        pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert "recruiter analysis" in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is original_error
    # Interview generation must never run after this failure.
    assert interview_generator.calls == []


def test_interview_generator_failure_raises_pipeline_error_with_cause():
    original_error = InterviewQuestionGeneratorError("job_requirements is required.")
    pipeline, *_ = _build_pipeline(generator_error=original_error)

    with pytest.raises(HireLensPipelineError) as exc_info:
        pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert "interview question generation" in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is original_error


def test_unexpected_exception_is_wrapped_not_leaked():
    """A non-application exception from a sub-service must still be wrapped, not leaked raw."""
    original_error = RuntimeError("boom")
    pipeline, *_ = _build_pipeline(candidate_error=original_error)

    with pytest.raises(HireLensPipelineError) as exc_info:
        pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert "unexpectedly" in str(exc_info.value).lower()
    assert exc_info.value.__cause__ is original_error


def test_evidence_matcher_invoked_with_candidate_profile_and_job_requirements():
    pipeline, _, _, evidence_matcher, *_ = _build_pipeline()

    pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert len(evidence_matcher.calls) == 1
    called_profile, called_requirements = evidence_matcher.calls[0]
    assert called_profile == CANDIDATE_PROFILE
    assert called_requirements == JOB_REQUIREMENTS


def test_alignment_scorer_invoked_with_job_requirements_and_evidence_matches():
    pipeline, _, _, _, alignment_scorer, _, _ = _build_pipeline()

    pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert len(alignment_scorer.calls) == 1
    called_requirements, called_matches = alignment_scorer.calls[0]
    assert called_requirements == JOB_REQUIREMENTS
    assert called_matches == EVIDENCE_MATCHES


def test_recruiter_analysis_generator_invoked_with_profile_requirements_matches_and_score():
    pipeline, _, _, _, _, recruiter_analysis_generator, _ = _build_pipeline()

    pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert len(recruiter_analysis_generator.calls) == 1
    called_profile, called_requirements, called_matches, called_score = recruiter_analysis_generator.calls[0]
    assert called_profile == CANDIDATE_PROFILE
    assert called_requirements == JOB_REQUIREMENTS
    assert called_matches == EVIDENCE_MATCHES
    assert called_score == ALIGNMENT_SCORE


def test_interview_generator_invoked_with_profile_requirements_and_matches():
    pipeline, _, _, _, _, _, interview_generator = _build_pipeline()

    pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert len(interview_generator.calls) == 1
    called_profile, called_requirements, called_matches = interview_generator.calls[0]
    assert called_profile == CANDIDATE_PROFILE
    assert called_requirements == JOB_REQUIREMENTS
    assert called_matches == EVIDENCE_MATCHES


def test_candidate_dossier_assembled_correctly():
    pipeline, *_ = _build_pipeline()

    dossier = pipeline.analyze("candidate.pdf", "Backend Engineer job description")

    assert dossier.candidate_profile == CANDIDATE_PROFILE
    assert dossier.job_requirements == JOB_REQUIREMENTS
    assert dossier.evidence_matches == EVIDENCE_MATCHES
    assert dossier.alignment_score == ALIGNMENT_SCORE
    assert dossier.recruiter_analysis == RECRUITER_ANALYSIS
    assert dossier.interview_questions == INTERVIEW_QUESTIONS
    # Warnings surfaced by the JD analyzer are carried forward, unmodified.
    assert dossier.warnings == JOB_REQUIREMENTS.warnings
    assert dossier.generated_at is not None


def test_dependency_injection_with_plain_duck_typed_objects():
    """The pipeline must work with any object exposing the right methods, not just the real services."""

    class _MinimalCandidateAnalyzer:
        def analyze(self, cv_text):
            return CandidateProfile(full_name="Minimal Candidate")

    class _MinimalJDAnalyzer:
        def analyze(self, jd_text):
            return JobRequirements(requirements=[])

    class _MinimalEvidenceMatcher:
        def match(self, candidate_profile, job_requirements):
            return []

    class _MinimalAlignmentScorer:
        def score(self, job_requirements, evidence_matches):
            return AlignmentScore(
                overall_score=0,
                alignment_label="Minimal Alignment",
                required_score=0.0,
                preferred_score=0.0,
                required_weight=0.0,
                preferred_weight=0.0,
                coverage=[],
                methodology_note="No requirements were extracted.",
            )

    class _MinimalRecruiterAnalysisGenerator:
        def generate(self, candidate_profile, job_requirements, evidence_matches, alignment_score):
            return RecruiterAnalysis(summary="No requirements were extracted to analyze.")

    class _MinimalInterviewGenerator:
        def generate(self, candidate_profile, job_requirements, evidence_matches):
            return []

    pipeline = HireLensPipeline(
        candidate_analyzer=_MinimalCandidateAnalyzer(),
        jd_analyzer=_MinimalJDAnalyzer(),
        evidence_matcher=_MinimalEvidenceMatcher(),
        alignment_scorer=_MinimalAlignmentScorer(),
        recruiter_analysis_generator=_MinimalRecruiterAnalysisGenerator(),
        interview_generator=_MinimalInterviewGenerator(),
        extract_document_fn=_fake_extract_document_fn(),
    )

    dossier = pipeline.analyze("candidate.pdf", "Some job description")

    assert isinstance(dossier, CandidateDossier)
    assert dossier.candidate_profile.full_name == "Minimal Candidate"
    assert dossier.alignment_score.overall_score == 0
    assert dossier.recruiter_analysis.summary


def test_pipeline_module_does_not_import_groq_sdk():
    """The pipeline orchestrator must never import the Groq SDK directly.

    Checked via the actual parsed import statements (not a substring search
    over the source) so that prose in docstrings/comments explaining this
    very constraint (which legitimately mentions "groq") can't produce a
    false positive.
    """
    source = inspect.getsource(hirelens_pipeline_module)
    tree = ast.parse(source)

    imported_module_names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imported_module_names.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported_module_names.add(node.module)
            imported_module_names.update(f"{node.module}.{alias.name}" for alias in node.names)

    assert not any(name == "groq" or name.startswith("groq.") for name in imported_module_names)
    assert not hasattr(hirelens_pipeline_module, "groq")


def test_empty_job_description_handling_delegates_to_jd_analyzer():
    """Empty/invalid job description text is validated by the JD analyzer, not duplicated in the pipeline."""
    # Simulate the real JobDescriptionAnalyzer's contract: it raises
    # JobDescriptionAnalyzerError for empty/whitespace-only text.
    empty_text_error = JobDescriptionAnalyzerError(
        "Cannot analyze empty job description text: no content was provided."
    )

    class _StrictFakeJDAnalyzer:
        def analyze(self, jd_text):
            if not jd_text or not jd_text.strip():
                raise empty_text_error
            return JOB_REQUIREMENTS

    pipeline = HireLensPipeline(
        candidate_analyzer=_FakeCandidateAnalyzer(profile=CANDIDATE_PROFILE),
        jd_analyzer=_StrictFakeJDAnalyzer(),
        evidence_matcher=_FakeEvidenceMatcher(matches=EVIDENCE_MATCHES),
        alignment_scorer=_FakeAlignmentScorer(alignment_score=ALIGNMENT_SCORE),
        recruiter_analysis_generator=_FakeRecruiterAnalysisGenerator(recruiter_analysis=RECRUITER_ANALYSIS),
        interview_generator=_FakeInterviewGenerator(questions=INTERVIEW_QUESTIONS),
        extract_document_fn=_fake_extract_document_fn(),
    )

    with pytest.raises(HireLensPipelineError) as exc_info:
        pipeline.analyze("candidate.pdf", "   ")

    assert exc_info.value.__cause__ is empty_text_error


def test_create_classmethod_wires_all_services_to_shared_llm_service():
    """HireLensPipeline.create() should build every LLM-backed sub-service around the same llm_service."""
    from src.services.alignment_scorer import AlignmentScorer
    from src.services.candidate_analyzer import CandidateAnalyzer
    from src.services.evidence_matcher import EvidenceMatcher
    from src.services.interview_generator import InterviewQuestionGenerator
    from src.services.jd_analyzer import JobDescriptionAnalyzer
    from src.services.recruiter_analysis_generator import RecruiterAnalysisGenerator

    class _FakeLLMService:
        def generate_text(self, prompt, system_instruction=None):
            return "{}"

    fake_llm = _FakeLLMService()
    pipeline = HireLensPipeline.create(fake_llm)

    assert isinstance(pipeline, HireLensPipeline)
    assert isinstance(pipeline._candidate_analyzer, CandidateAnalyzer)
    assert isinstance(pipeline._jd_analyzer, JobDescriptionAnalyzer)
    assert isinstance(pipeline._evidence_matcher, EvidenceMatcher)
    assert isinstance(pipeline._alignment_scorer, AlignmentScorer)
    assert isinstance(pipeline._recruiter_analysis_generator, RecruiterAnalysisGenerator)
    assert isinstance(pipeline._interview_generator, InterviewQuestionGenerator)
