"""Recruiter Analysis Generator for HireLens AI.

Synthesizes the evidence already gathered by the Evidence Matcher and the
deterministic :class:`~src.services.alignment_scorer.AlignmentScorer` into
a natural-language, recruiter-facing :class:`~src.models.schemas.RecruiterAnalysis`
— a narrative summary, strengths, gaps, and interview validation areas.

PRODUCT DIRECTION / CRITICAL RULES:
HireLens is a recruiter-facing decision-support system: this module IS
allowed to synthesize an analysis, unlike the earlier "normalization only"
services. It must never, however:

- Invent experience, skills, employers, education, years of experience,
  certifications, or technologies not already present in the evidence it
  is given.
- Treat "no evidence found" as proof the candidate lacks a skill. Gaps
  must be phrased as "not evidenced in the submitted CV", never as
  "does not have" / "lacks" / "is missing", unless the CV itself
  explicitly states the candidate lacks it.
- Claim certainty, an objective hiring probability, or state/imply an
  actual hire/reject decision. The analysis supports a human recruiter's
  own judgment; it never replaces it.
- Reference or infer any protected characteristic (race, ethnicity,
  religion, gender, age, disability, marital status, nationality,
  appearance, health, or similar).
- Invent or adjust the alignment score itself — that number always comes
  from ``AlignmentScorer``, computed by application logic; this module
  only explains and synthesizes it in prose.

ARCHITECTURE:
Like interview question generation, this is a language-generation task, so
an injected LLM is the primary path when available. A deterministic,
template-based fallback (built directly from the evidence and alignment
score, no LLM) is used whenever no LLM service is injected, or whenever
the LLM call fails, returns malformed output, or fails schema validation —
so a single bad LLM response never crashes the pipeline.

This module depends only on the reusable :class:`GroqLLMService`
abstraction (``src.services.llm_service``) — it never imports the Groq SDK
directly — and the generated analysis is validated against the existing
``RecruiterAnalysis`` Pydantic model.

SECURITY NOTE: Candidate personal information, raw prompts, and raw LLM
responses are never logged by this module. Only the requirement text,
evidence excerpts already surfaced by the Evidence Matcher, and the
deterministic alignment score (never full CandidateProfile contact
details) are sent to the LLM.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from src.models.schemas import (
    AlignmentScore,
    CandidateProfile,
    EvidenceMatch,
    EvidenceStatus,
    JobRequirements,
    RecruiterAnalysis,
    RequirementImportance,
)
from src.services.llm_service import GroqLLMService, LLMServiceError

# Matches a JSON payload wrapped in a markdown code fence, e.g. ```json ... ```
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)

# Order in which importance levels are prioritized for validation areas and
# prompt presentation: required first, then preferred, then unclear.
_IMPORTANCE_PRIORITY = {
    RequirementImportance.required: 0,
    RequirementImportance.preferred: 1,
    RequirementImportance.unclear: 2,
}

# Maximum number of items in the deterministic fallback's lists, so a JD
# with many requirements doesn't produce an unreadably long dump.
_MAX_FALLBACK_ITEMS = 8

_SYSTEM_INSTRUCTION = (
    "You are a recruiter-facing analysis assistant for a tool called "
    "HireLens AI. Your job is to synthesize an evidence-grounded analysis "
    "that helps a human recruiter understand how a candidate's documented "
    "experience aligns with a job's stated requirements. This is decision "
    "SUPPORT, not a hiring decision: the human recruiter always makes the "
    "final call, and you must never state or imply that the candidate "
    "should be hired, rejected, or is or isn't 'suitable' for the role.\n\n"
    "GROUNDING RULES (follow strictly):\n"
    "- Base your analysis only on the requirement/evidence data and the "
    "alignment score provided to you below. Never invent experience, "
    "skills, employers, education, years of experience, certifications, "
    "or technologies that are not present in what was provided.\n"
    "- Never state or imply certainty, an objective probability of hiring "
    "success, or an actual hire/reject outcome.\n"
    "- For any requirement with status 'no_evidence_found', phrase it as "
    "'not evidenced in the submitted CV' (or equivalent neutral phrasing) "
    "— never as 'does not have', 'lacks', or 'is missing', unless the "
    "evidence explicitly states the candidate lacks it.\n"
    "- For any requirement with status 'needs_verification', describe it "
    "as partially or ambiguously supported, not as confirmed either way.\n"
    "- Never reference or infer race, ethnicity, religion, gender, age, "
    "disability, marital status, nationality, appearance, health, or any "
    "other protected or personal characteristic.\n"
    "- Do not invent or adjust the alignment score — treat the provided "
    "score and methodology as fixed facts to explain, not to change.\n"
    "- Return structured JSON only, with no prose, commentary, or "
    "markdown formatting around it."
)

_PROMPT_TEMPLATE = """\
Synthesize a recruiter-facing analysis from the evidence and alignment \
score below.

ROLE: {role_title}

ALIGNMENT SCORE: {overall_score}/100 ({alignment_label})
SCORING METHODOLOGY: {methodology_note}

REQUIREMENTS AND THEIR EVIDENCE STATUS (the only candidate information you \
may reference):
{requirement_block}

Return a single JSON object with EXACTLY this shape:

{{
  "summary": string,
  "strengths": [string, ...],
  "gaps": [string, ...],
  "validation_areas": [string, ...]
}}

Field notes:
- "summary": a detailed, multi-sentence narrative explaining how the \
candidate's evidence aligns with the role — what is strongly supported, \
what is only partially supported, what preferred qualifications are \
missing, and the overall alignment. Reference the alignment score and \
methodology, but do not state or imply a hiring decision.
- "strengths": specific strengths grounded in requirements with \
'evidence_found' status.
- "gaps": specific requirements with 'no_evidence_found' or \
'needs_verification' status, phrased as "not evidenced in the submitted \
CV" (or "partially evidenced" / "needs verification" as appropriate) — \
never as a confirmed absence.
- "validation_areas": the most important things a recruiter should verify \
directly with the candidate, prioritized by requirement importance \
(required first) and by evidence gaps.
- Return JSON only — no markdown code fences, no explanation outside the \
JSON object.
"""


class RecruiterAnalysisGeneratorError(Exception):
    """Raised for invalid usage of :class:`RecruiterAnalysisGenerator`.

    This is reserved for programming errors (e.g. missing required
    arguments). Ordinary generation problems are instead handled internally
    via the deterministic fallback, so an LLM failure never crashes the
    pipeline.
    """


class RecruiterAnalysisGenerator:
    """Synthesizes an evidence-grounded :class:`RecruiterAnalysis`.

    Uses an injected :class:`GroqLLMService` (or any compatible stub/mock
    exposing ``generate_text(prompt, system_instruction=None) -> str``) as
    the primary synthesis path when available, and a deterministic,
    template-based fallback otherwise (or whenever the LLM call fails,
    returns malformed output, or fails schema validation).
    """

    def __init__(self, llm_service: GroqLLMService | None = None) -> None:
        """Initialize the generator.

        Args:
            llm_service: Optional LLM service used to synthesize the
                narrative analysis. If omitted, the analysis is generated
                deterministically from templates, avoiding any LLM calls.
        """
        self._llm_service = llm_service

    def generate(
        self,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
        alignment_score: AlignmentScore,
    ) -> RecruiterAnalysis:
        """Synthesize a recruiter analysis from the evidence and alignment score.

        Args:
            candidate_profile: The candidate's structured profile. Not sent
                to the LLM directly — the evidence excerpts already
                surfaced by the Evidence Matcher are the vetted, minimal
                representation of relevant candidate information, so
                re-sending the full profile would only add redundant PII
                exposure without adding grounding value. Accepted here for
                interface consistency with the rest of the pipeline.
            job_requirements: The job's structured requirements.
            evidence_matches: The Evidence Matcher's results.
            alignment_score: The deterministic alignment score to explain.

        Returns:
            A ``RecruiterAnalysis``.

        Raises:
            RecruiterAnalysisGeneratorError: If ``candidate_profile``,
                ``job_requirements``, ``evidence_matches``, or
                ``alignment_score`` is not provided.
        """
        if candidate_profile is None:
            raise RecruiterAnalysisGeneratorError("candidate_profile is required.")
        if job_requirements is None:
            raise RecruiterAnalysisGeneratorError("job_requirements is required.")
        if evidence_matches is None:
            raise RecruiterAnalysisGeneratorError("evidence_matches is required.")
        if alignment_score is None:
            raise RecruiterAnalysisGeneratorError("alignment_score is required.")

        if self._llm_service is not None:
            llm_analysis = self._llm_generate_analysis(
                job_requirements, evidence_matches, alignment_score
            )
            if llm_analysis is not None:
                return llm_analysis

        return self._deterministic_analysis(job_requirements, evidence_matches, alignment_score)

    # -- LLM-based synthesis -------------------------------------------------

    def _llm_generate_analysis(
        self,
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
        alignment_score: AlignmentScore,
    ) -> RecruiterAnalysis | None:
        """Attempt to synthesize the analysis via the LLM. Returns None on any failure."""
        prompt = _PROMPT_TEMPLATE.format(
            role_title=job_requirements.role_title or "(not stated)",
            overall_score=alignment_score.overall_score,
            alignment_label=alignment_score.alignment_label,
            methodology_note=alignment_score.methodology_note,
            requirement_block=self._build_requirement_block(job_requirements, evidence_matches),
        )

        try:
            raw_response = self._llm_service.generate_text(
                prompt=prompt,
                system_instruction=_SYSTEM_INSTRUCTION,
            )
        except LLMServiceError:
            return None

        payload = self._parse_json_response(raw_response)
        if payload is None:
            return None

        try:
            return RecruiterAnalysis.model_validate(payload)
        except ValidationError:
            return None

    @staticmethod
    def _build_requirement_block(
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
    ) -> str:
        """Build the per-requirement evidence block sent to the LLM."""
        matches_by_id = {match.requirement_id: match for match in evidence_matches}
        lines = []

        for requirement in job_requirements.requirements:
            match = matches_by_id.get(requirement.id)
            status = match.status.value if match is not None else "needs_verification"
            evidence = ", ".join(match.evidence) if match is not None and match.evidence else "(none)"
            lines.append(
                f"- [{requirement.importance.value}] \"{requirement.requirement}\" "
                f"— status: {status}; evidence: {evidence}"
            )

        return "\n".join(lines) if lines else "(no requirements extracted)"

    # -- Deterministic fallback -----------------------------------------------

    def _deterministic_analysis(
        self,
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
        alignment_score: AlignmentScore,
    ) -> RecruiterAnalysis:
        """Build a template-based analysis without calling the LLM.

        This is guaranteed to succeed and produce a schema-valid analysis,
        so it also serves as the ultimate fallback if the LLM path fails.
        """
        requirements_by_id = {requirement.id: requirement for requirement in job_requirements.requirements}

        strengths = []
        gaps = []
        validation_candidates: list[tuple[int, str]] = []

        for match in evidence_matches:
            requirement = requirements_by_id.get(match.requirement_id)
            importance = requirement.importance if requirement is not None else RequirementImportance.unclear

            if match.status == EvidenceStatus.evidence_found:
                strengths.append(f"Evidence found for '{match.requirement}'.")
            elif match.status == EvidenceStatus.no_evidence_found:
                gaps.append(f"'{match.requirement}' was not evidenced in the submitted CV.")
                validation_candidates.append((_IMPORTANCE_PRIORITY.get(importance, 2), match.requirement))
            else:  # needs_verification
                gaps.append(f"'{match.requirement}' was only partially evidenced and needs verification.")
                validation_candidates.append((_IMPORTANCE_PRIORITY.get(importance, 2), match.requirement))

        validation_candidates.sort(key=lambda item: item[0])
        validation_areas = [text for _, text in validation_candidates[:_MAX_FALLBACK_ITEMS]]

        summary = (
            f"Overall alignment score: {alignment_score.overall_score}/100 "
            f"({alignment_score.alignment_label}) for the role of "
            f"'{job_requirements.role_title or 'this position'}'. "
            f"{alignment_score.methodology_note} "
        )
        if strengths:
            summary += (
                f"{len(strengths)} of {len(evidence_matches)} requirement(s) have explicit "
                f"evidence in the submitted documents. "
            )
        if gaps:
            summary += (
                f"{len(gaps)} requirement(s) were not evidenced or only partially "
                f"evidenced in the submitted documents — this does not mean the "
                f"candidate lacks these qualifications, only that they were not "
                f"confirmed in the documents reviewed. "
            )
        summary += (
            "This analysis is decision support only; it does not make a hiring "
            "decision, and the human recruiter should verify open items directly "
            "with the candidate."
        )

        return RecruiterAnalysis(
            summary=summary,
            strengths=strengths[:_MAX_FALLBACK_ITEMS],
            gaps=gaps[:_MAX_FALLBACK_ITEMS],
            validation_areas=validation_areas,
        )

    # -- JSON parsing ----------------------------------------------------------

    @staticmethod
    def _parse_json_response(raw_response: str) -> dict | None:
        """Safely parse the LLM's raw text response into a JSON object.

        Strips a surrounding markdown code fence if present. Returns
        ``None`` (rather than raising) on any parsing problem, so the
        caller can fall back to deterministic generation instead of
        crashing.
        """
        if raw_response is None or not raw_response.strip():
            return None

        cleaned = raw_response.strip()
        fence_match = _CODE_FENCE_RE.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError:
            return None

        if not isinstance(payload, dict):
            return None

        return payload
