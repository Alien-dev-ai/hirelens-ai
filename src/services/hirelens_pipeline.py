"""HireLens pipeline orchestrator.

Wires together document extraction, candidate analysis, job description
analysis, evidence matching, deterministic alignment scoring, LLM recruiter
analysis synthesis, and interview question generation into a single,
end-to-end workflow that produces a
:class:`~src.models.schemas.CandidateDossier`.

PRODUCT DIRECTION:
HireLens is a recruiter-facing candidate analysis and *decision-support*
system: this pipeline computes a deterministic alignment score
(``AlignmentScorer``) and an LLM-synthesized recruiter analysis
(``RecruiterAnalysisGenerator``) on top of the underlying evidence, in
addition to the interview questions it always produced. It still preserves
the boundaries that make this decision support rather than a hiring
decision: "no evidence found" means no explicit evidence was found in the
submitted documents, and must never be interpreted as meaning the
candidate does not possess that skill or qualification; nothing in this
pipeline claims certainty, an objective hiring probability, or considers
protected characteristics; and all hiring decisions remain with the human
recruiter.

ARCHITECTURE:
This module is a thin orchestrator — it reuses the existing
``CandidateAnalyzer``, ``JobDescriptionAnalyzer``, ``EvidenceMatcher``,
``AlignmentScorer``, ``RecruiterAnalysisGenerator``, and
``InterviewQuestionGenerator`` services (and the ``extract_document``
function) rather than duplicating any of their logic. Every dependency is
injected through the constructor, so the pipeline can be fully exercised in
tests with fakes/mocks and never needs to make a real Groq API call. This
module never imports the Groq SDK directly. The pipeline stage order is:

    CV extraction -> candidate analysis -> JD analysis -> evidence matching
    -> deterministic alignment scoring -> LLM recruiter analysis
    -> interview question generation

so that the score and the narrative it explains are always computed from
the same, already-validated evidence, and the LLM is never asked to
invent or adjust the score itself.

SECURITY NOTE: Candidate personal information and raw document text are
never logged by this module. It emits lightweight, stage-boundary logging
(via the standard ``logging`` module) recording only: the stage name,
whether it started/succeeded/failed, and elapsed duration — plus, on
failure, the exception's string representation, which every underlying
service already keeps free of CV/JD content (static or Groq-SDK-derived
error text only, never the prompt or document text). No CV text, JD text,
extracted document contents, API keys, or credentials are ever logged.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Callable

from src.extraction.document_extractor import extract_document
from src.models.schemas import (
    AlignmentScore,
    CandidateDossier,
    CandidateProfile,
    EvidenceMatch,
    ExtractedDocument,
    InterviewQuestion,
    JobRequirements,
    RecruiterAnalysis,
)
from src.services.alignment_scorer import AlignmentScorer, AlignmentScorerError
from src.services.candidate_analyzer import CandidateAnalyzer, CandidateAnalyzerError
from src.services.evidence_matcher import EvidenceMatcher, EvidenceMatcherError
from src.services.interview_generator import (
    InterviewQuestionGenerator,
    InterviewQuestionGeneratorError,
)
from src.services.jd_analyzer import JobDescriptionAnalyzer, JobDescriptionAnalyzerError
from src.services.llm_service import GroqLLMService, LLMQuotaExceededError
from src.services.recruiter_analysis_generator import (
    RecruiterAnalysisGenerator,
    RecruiterAnalysisGeneratorError,
)

# Type of the injectable document-extraction function: accepts a file path
# (or path-like string) and returns an ExtractedDocument. Matches the
# signature of src.extraction.document_extractor.extract_document.
DocumentExtractorFn = Callable[[str | Path], ExtractedDocument]

logger = logging.getLogger(__name__)


class HireLensPipelineError(Exception):
    """Raised when any stage of the HireLens analysis pipeline fails.

    The original underlying exception (if any) is preserved as this
    exception's ``__cause__``, so callers can inspect the root cause while
    still catching a single, pipeline-level exception type.
    """

    @property
    def is_quota_exceeded(self) -> bool:
        """True if a Groq quota/rate-limit error caused this failure.

        Walks the ``__cause__`` chain (populated via ``raise ... from exc``
        at every stage of the pipeline) looking for an
        :class:`~src.services.llm_service.LLMQuotaExceededError`. Lets
        callers such as the UI show a friendly "try again later" message
        without needing to know about the LLM service's internal
        exception hierarchy.
        """
        current: BaseException | None = self
        seen: set[int] = set()
        while current is not None and id(current) not in seen:
            seen.add(id(current))
            if isinstance(current, LLMQuotaExceededError):
                return True
            current = current.__cause__
        return False


class HireLensPipeline:
    """Orchestrates the full HireLens candidate analysis workflow.

    Given a candidate CV file and job description text, runs document
    extraction, candidate profile analysis, job requirement analysis,
    evidence matching, and interview question generation — in that order —
    and assembles the results into a single ``CandidateDossier``.

    All dependencies are injected via the constructor so this class can be
    tested end-to-end with fakes/mocks, with no real Groq API calls. Use
    :meth:`create` for the common case of wiring every service to the same
    ``GroqLLMService``.
    """

    def __init__(
        self,
        candidate_analyzer: CandidateAnalyzer,
        jd_analyzer: JobDescriptionAnalyzer,
        evidence_matcher: EvidenceMatcher,
        alignment_scorer: AlignmentScorer,
        recruiter_analysis_generator: RecruiterAnalysisGenerator,
        interview_generator: InterviewQuestionGenerator,
        extract_document_fn: DocumentExtractorFn = extract_document,
    ) -> None:
        """Initialize the pipeline with its component services.

        Args:
            candidate_analyzer: Service used to turn extracted CV text into
                a ``CandidateProfile``.
            jd_analyzer: Service used to turn job description text into
                ``JobRequirements``.
            evidence_matcher: Service used to match the candidate profile
                against the job requirements.
            alignment_scorer: Service used to compute a deterministic
                candidate-to-role ``AlignmentScore`` from the evidence
                matches.
            recruiter_analysis_generator: Service used to synthesize a
                ``RecruiterAnalysis`` from the evidence matches and
                alignment score.
            interview_generator: Service used to generate interview
                questions from the requirements and evidence matches.
            extract_document_fn: Callable used to extract text from the CV
                file. Defaults to
                ``src.extraction.document_extractor.extract_document``.
                Injectable so tests can avoid real file I/O.
        """
        self._candidate_analyzer = candidate_analyzer
        self._jd_analyzer = jd_analyzer
        self._evidence_matcher = evidence_matcher
        self._alignment_scorer = alignment_scorer
        self._recruiter_analysis_generator = recruiter_analysis_generator
        self._interview_generator = interview_generator
        self._extract_document_fn = extract_document_fn

    @classmethod
    def create(cls, llm_service: GroqLLMService) -> "HireLensPipeline":
        """Build a pipeline with every service wired to the same LLM service.

        Convenience constructor for production use, where every
        LLM-backed analysis service should share one ``GroqLLMService``
        instance. ``AlignmentScorer`` has no LLM dependency (it is purely
        deterministic), so it is constructed with no arguments.

        Args:
            llm_service: The LLM service to inject into every LLM-backed
                sub-service.

        Returns:
            A fully configured ``HireLensPipeline``.
        """
        return cls(
            candidate_analyzer=CandidateAnalyzer(llm_service),
            jd_analyzer=JobDescriptionAnalyzer(llm_service),
            evidence_matcher=EvidenceMatcher(llm_service),
            alignment_scorer=AlignmentScorer(),
            recruiter_analysis_generator=RecruiterAnalysisGenerator(llm_service),
            interview_generator=InterviewQuestionGenerator(llm_service),
        )

    def analyze(self, cv_file_path: str | Path, job_description: str) -> CandidateDossier:
        """Run the full pipeline and return the resulting candidate dossier.

        Args:
            cv_file_path: Path to the candidate's CV file (PDF or DOCX).
            job_description: Raw job description text.

        Returns:
            A ``CandidateDossier`` assembled from the candidate profile, job
            requirements, evidence matches, deterministic alignment score,
            recruiter analysis, and interview questions.

        Raises:
            HireLensPipelineError: If any stage of the pipeline fails —
                document extraction, candidate analysis, job description
                analysis, evidence matching, alignment scoring, recruiter
                analysis synthesis, or interview question generation. The
                original exception, where one exists, is preserved as the
                raised error's ``__cause__``.
        """
        pipeline_start = time.monotonic()
        logger.info("HireLens pipeline started")

        document = self._extract_cv(cv_file_path)
        candidate_profile = self._analyze_candidate(document.text)
        job_requirements = self._analyze_job_description(job_description)
        evidence_matches = self._match_evidence(candidate_profile, job_requirements)
        alignment_score = self._compute_alignment_score(job_requirements, evidence_matches)
        recruiter_analysis = self._generate_recruiter_analysis(
            candidate_profile, job_requirements, evidence_matches, alignment_score
        )
        interview_questions = self._generate_interview_questions(
            candidate_profile, job_requirements, evidence_matches
        )

        dossier = CandidateDossier(
            candidate_profile=candidate_profile,
            job_requirements=job_requirements,
            evidence_matches=evidence_matches,
            alignment_score=alignment_score,
            recruiter_analysis=recruiter_analysis,
            interview_questions=interview_questions,
            # Carry forward any ambiguities/gaps the JD analyzer already
            # noted while parsing the job description. The pipeline adds no
            # judgments of its own beyond the deterministic alignment score
            # and the grounded recruiter analysis computed above.
            warnings=list(job_requirements.warnings),
        )

        logger.info(
            "HireLens pipeline completed successfully in %.2fs",
            time.monotonic() - pipeline_start,
        )
        return dossier

    # -- Stage-boundary logging helpers -------------------------------------
    #
    # Deliberately minimal: log only the stage name, timing, and (on
    # failure) the already-sanitized exception text — never CV/JD content,
    # extracted document text, API keys, or credentials. See this module's
    # docstring "SECURITY NOTE".

    @staticmethod
    def _log_stage_start(stage: str) -> float:
        logger.info("Stage started: %s", stage)
        return time.monotonic()

    @staticmethod
    def _log_stage_success(stage: str, start: float) -> None:
        logger.info("Stage completed: %s (%.2fs)", stage, time.monotonic() - start)

    @staticmethod
    def _log_stage_failure(stage: str, start: float, detail: object) -> None:
        logger.error(
            "Stage failed: %s (%.2fs): %s", stage, time.monotonic() - start, detail
        )

    # -- Individual pipeline stages, each with its own error handling ------

    def _extract_cv(self, cv_file_path: str | Path) -> ExtractedDocument:
        """Stage 1: extract text from the candidate's CV file."""
        stage = "1/7 document_extraction"
        start = self._log_stage_start(stage)
        try:
            document = self._extract_document_fn(cv_file_path)
        except Exception as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(
                f"Document extraction failed unexpectedly: {exc}"
            ) from exc

        if not document.extraction_success:
            error_detail = (
                document.error_message or "no text could be extracted from the document."
            )
            self._log_stage_failure(stage, start, error_detail)
            raise HireLensPipelineError(f"Document extraction failed: {error_detail}")

        self._log_stage_success(stage, start)
        return document

    def _analyze_candidate(self, cv_text: str) -> CandidateProfile:
        """Stage 2: analyze the extracted CV text into a candidate profile."""
        stage = "2/7 candidate_analysis"
        start = self._log_stage_start(stage)
        try:
            profile = self._candidate_analyzer.analyze(cv_text)
        except CandidateAnalyzerError as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(f"Candidate analysis failed: {exc}") from exc
        except Exception as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(
                f"Candidate analysis failed unexpectedly: {exc}"
            ) from exc

        self._log_stage_success(stage, start)
        return profile

    def _analyze_job_description(self, job_description: str) -> JobRequirements:
        """Stage 3: analyze the job description into structured requirements."""
        stage = "3/7 job_description_analysis"
        start = self._log_stage_start(stage)
        try:
            requirements = self._jd_analyzer.analyze(job_description)
        except JobDescriptionAnalyzerError as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(f"Job description analysis failed: {exc}") from exc
        except Exception as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(
                f"Job description analysis failed unexpectedly: {exc}"
            ) from exc

        self._log_stage_success(stage, start)
        return requirements

    def _match_evidence(
        self,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
    ) -> list[EvidenceMatch]:
        """Stage 4: match the candidate profile against the job requirements."""
        stage = "4/7 evidence_matching"
        start = self._log_stage_start(stage)
        try:
            matches = self._evidence_matcher.match(candidate_profile, job_requirements)
        except EvidenceMatcherError as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(f"Evidence matching failed: {exc}") from exc
        except Exception as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(
                f"Evidence matching failed unexpectedly: {exc}"
            ) from exc

        self._log_stage_success(stage, start)
        return matches

    def _compute_alignment_score(
        self,
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
    ) -> AlignmentScore:
        """Stage 5: compute the deterministic candidate-to-role alignment score."""
        stage = "5/7 alignment_scoring"
        start = self._log_stage_start(stage)
        try:
            score = self._alignment_scorer.score(job_requirements, evidence_matches)
        except AlignmentScorerError as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(f"Alignment scoring failed: {exc}") from exc
        except Exception as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(
                f"Alignment scoring failed unexpectedly: {exc}"
            ) from exc

        self._log_stage_success(stage, start)
        return score

    def _generate_recruiter_analysis(
        self,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
        alignment_score: AlignmentScore,
    ) -> RecruiterAnalysis:
        """Stage 6: synthesize the recruiter-facing analysis from evidence and score."""
        stage = "6/7 recruiter_analysis_generation"
        start = self._log_stage_start(stage)
        try:
            analysis = self._recruiter_analysis_generator.generate(
                candidate_profile, job_requirements, evidence_matches, alignment_score
            )
        except RecruiterAnalysisGeneratorError as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(f"Recruiter analysis generation failed: {exc}") from exc
        except Exception as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(
                f"Recruiter analysis generation failed unexpectedly: {exc}"
            ) from exc

        self._log_stage_success(stage, start)
        return analysis

    def _generate_interview_questions(
        self,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
    ) -> list[InterviewQuestion]:
        """Stage 7: generate interview questions from requirements and evidence."""
        stage = "7/7 interview_question_generation"
        start = self._log_stage_start(stage)
        try:
            questions = self._interview_generator.generate(
                candidate_profile, job_requirements, evidence_matches
            )
        except InterviewQuestionGeneratorError as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(
                f"Interview question generation failed: {exc}"
            ) from exc
        except Exception as exc:
            self._log_stage_failure(stage, start, exc)
            raise HireLensPipelineError(
                f"Interview question generation failed unexpectedly: {exc}"
            ) from exc

        self._log_stage_success(stage, start)
        return questions
