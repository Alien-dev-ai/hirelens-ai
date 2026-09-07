"""Evidence Matcher for HireLens AI.

Compares explicitly stated job requirements against information explicitly
present in a candidate's structured profile, and returns one
:class:`~src.models.schemas.EvidenceMatch` per
:class:`~src.models.schemas.JobRequirement`.

IMPORTANT PRODUCT BOUNDARY / CRITICAL RULE:
"No evidence found in the submitted CV" does NOT mean "the candidate does
not have this skill." It means only that no explicit supporting evidence
was located in the documents reviewed. This module must never phrase a
``no_evidence_found`` result as a negative claim about the candidate (e.g.
"lacks", "unqualified", "unsuitable"), and it never scores, ranks, or makes
any hiring/acceptance/rejection recommendation about the candidate.

ARCHITECTURE — hybrid matching:
1. Deterministic matching first: the candidate's structured fields (skills,
   experience, education, projects, certifications, summary) are searched
   directly for requirement terms. This is fast, fully explainable, and
   needs no LLM call.
2. Only when deterministic matching is inconclusive, and an LLM service was
   injected, a Groq call is used for semantic matching/verification —
   scoped strictly to evidence-finding (never scoring, ranking, or hiring
   recommendations) and required to ground any returned evidence in the
   candidate data actually supplied.

This module depends only on the reusable :class:`GroqLLMService`
abstraction (``src.services.llm_service``) — it never imports the Groq SDK
directly — and every match is validated against the existing
``EvidenceMatch`` Pydantic model.

SECURITY NOTE: Candidate personal information, raw prompts, and raw LLM
responses are never logged by this module. Only the structured fields
relevant to evidence-finding (skills, experience, education, projects,
certifications, summary) are ever sent to the LLM — contact details
(``full_name``, ``email``, ``phone``, ``location``, ``links``) are excluded
from the LLM context because they are not needed for matching.
"""

from __future__ import annotations

import json
import re

from src.models.schemas import (
    CandidateProfile,
    EvidenceMatch,
    EvidenceStatus,
    JobRequirement,
    JobRequirements,
)
from src.services.llm_service import GroqLLMService, LLMServiceError

# ---------------------------------------------------------------------------
# Confidence levels (design choice, documented for maintainers).
#
# These represent the matcher's confidence that the assigned *status* is
# correct given the available information, not any judgment about the
# candidate. Deterministic (exact, explainable) matches are trusted more
# than LLM-derived (semantic) ones; a matching failure gets the lowest
# confidence because the matcher genuinely does not know the answer.
# ---------------------------------------------------------------------------
_CONFIDENCE_DETERMINISTIC_SKILL_MATCH = 0.95
_CONFIDENCE_DETERMINISTIC_TEXT_MATCH = 0.85
_CONFIDENCE_DETERMINISTIC_NO_EVIDENCE = 0.6
_CONFIDENCE_LLM_EVIDENCE_FOUND = 0.65
_CONFIDENCE_LLM_NO_EVIDENCE = 0.55
_CONFIDENCE_LLM_NEEDS_VERIFICATION = 0.4
_CONFIDENCE_MATCH_FAILURE = 0.0

_NO_EVIDENCE_EXPLANATION_TEMPLATE = (
    "No explicit evidence for '{requirement}' was found in the submitted "
    "candidate documents. This does NOT mean the candidate is missing this "
    "qualification — it means only that no supporting evidence was present "
    "in the documents reviewed."
)

_MATCH_FAILURE_EXPLANATION = (
    "Automated evidence matching for this requirement was inconclusive due "
    "to an unexpected error. This is not a claim about the candidate — "
    "human verification is recommended."
)

# Boilerplate lead-in phrases stripped from requirement text before
# extracting the core term(s) to search for, e.g. "3+ years of experience
# with Python" -> "python".
_BOILERPLATE_PHRASES = (
    "hands-on experience with",
    "working knowledge of",
    "strong understanding of",
    "solid understanding of",
    "experience with",
    "experience in",
    "experience using",
    "knowledge of",
    "familiarity with",
    "proficiency in",
    "proficient in",
    "understanding of",
    "background in",
    "expertise in",
    "skilled in",
    "exposure to",
    "ability to",
    "bachelor's degree in",
    "bachelor degree in",
    "master's degree in",
    "certification in",
    "certified in",
    "degree in",
)

_YEARS_PATTERN = re.compile(r"\b\d+\+?\s*(?:years?|yrs?)\b", re.IGNORECASE)
_SPLIT_PATTERN = re.compile(r",| and | or | / |/")

# Requirement categories (as emitted by JobDescriptionAnalyzer, see
# src/services/jd_analyzer.py) that represent a formal credential the
# candidate either explicitly holds or does not. For these categories,
# deterministic matching must be scoped to the candidate source label(s)
# below rather than searching every collected source — an incidental word
# overlap in an unrelated field (e.g. "media" in the skill "Social Media
# Content Creation") is not evidence that the candidate holds a Film/Media
# degree. Labels are the existing ``source_label`` values produced by
# ``_collect_candidate_sources``; no parallel classification is introduced.
#
# ``CandidateProfile`` has no dedicated "license" field, so license
# requirements are scoped to the "certification" source label — the
# existing field where license-type credentials (e.g. "Registered Nurse
# (RN) License") are recorded.
#
# Any category not listed here (e.g. "skill", "experience", "tool", or an
# unrecognized/empty category) is left unscoped, preserving prior behavior.
_CREDENTIAL_CATEGORY_SOURCE_LABELS: dict[str, frozenset[str]] = {
    "education": frozenset({"education"}),
    "certification": frozenset({"certification"}),
    "license": frozenset({"certification"}),
}

# Matches a JSON payload wrapped in a markdown code fence, e.g. ```json ... ```
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)

_VALID_STATUS_VALUES = {status.value for status in EvidenceStatus}

# System instruction sent with every LLM semantic-matching request.
_SYSTEM_INSTRUCTION = (
    "You are a strict evidence-finding engine for a recruiting tool called "
    "HireLens AI. Your only job is to determine whether a single job "
    "requirement is explicitly supported by a candidate's structured "
    "profile data. You are NOT a recruiter, evaluator, or judge: you must "
    "never assign the candidate a score or a ranking, and you must never "
    "state or imply whether the candidate should be hired, accepted, or "
    "turned down for any role. This is evidence lookup, not candidate "
    "evaluation.\n\n"
    "GROUNDING RULES (follow strictly):\n"
    "- Only use the candidate profile data provided to you below. Do not "
    "use outside knowledge or assumptions about the candidate.\n"
    "- Never invent or fabricate evidence. Every string you return in "
    "'evidence' must be copied verbatim (or as a short verbatim excerpt) "
    "from the candidate profile data provided.\n"
    "- 'no_evidence_found' means only that no explicit supporting evidence "
    "was present in the provided data. It is NOT a claim that the "
    "candidate lacks the skill or qualification — never phrase it that "
    "way.\n"
    "- If the evidence is partial, indirect, or ambiguous, use "
    "'needs_verification' rather than guessing 'evidence_found'.\n"
    "- Return structured JSON only, with no prose, commentary, or "
    "markdown formatting around it."
)

_PROMPT_TEMPLATE = """\
Determine whether the following job requirement is explicitly supported by \
the candidate profile data below.

JOB REQUIREMENT:
"{requirement_text}"

CANDIDATE PROFILE DATA (the only information you may use):
{candidate_context}

Return a single JSON object with EXACTLY this shape:

{{
  "status": "evidence_found" | "no_evidence_found" | "needs_verification",
  "evidence": [string, ...],
  "explanation": string
}}

Rules:
- "evidence" must contain only verbatim excerpts copied from the candidate \
profile data above. Leave it empty ([]) if status is "no_evidence_found".
- "explanation" must be a short, neutral, factual explanation of the \
status. It must never claim the candidate lacks a skill, is unqualified, \
or is unsuitable, and must never recommend a hiring decision.
- Return JSON only — no markdown code fences, no explanation outside the \
JSON object.
"""


class EvidenceMatcherError(Exception):
    """Raised for invalid usage of :class:`EvidenceMatcher`.

    This is reserved for programming errors (e.g. missing required
    arguments), not for ordinary "no evidence found" outcomes, which are
    represented as a normal ``EvidenceMatch`` with
    ``status=EvidenceStatus.no_evidence_found``.
    """


class EvidenceMatcher:
    """Matches job requirements against evidence in a candidate profile.

    Uses deterministic keyword/phrase matching against the candidate's
    structured fields first, and falls back to an injected
    :class:`GroqLLMService` (or any compatible stub/mock exposing
    ``generate_text(prompt, system_instruction=None) -> str``) for semantic
    matching only when deterministic matching is inconclusive.
    """

    def __init__(self, llm_service: GroqLLMService | None = None) -> None:
        """Initialize the matcher.

        Args:
            llm_service: Optional LLM service used for semantic matching
                when deterministic matching cannot find or rule out
                evidence. If omitted, the matcher is fully deterministic
                and any inconclusive requirement is reported as
                ``no_evidence_found``.
        """
        self._llm_service = llm_service

    def match(
        self,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
    ) -> list[EvidenceMatch]:
        """Match every job requirement against the candidate profile.

        Args:
            candidate_profile: The candidate's structured profile.
            job_requirements: The job's structured requirements.

        Returns:
            A list of ``EvidenceMatch`` objects, exactly one per
            requirement in ``job_requirements.requirements``, in the same
            order. A per-requirement failure never prevents the other
            requirements from being matched — it is reported as its own
            ``EvidenceMatch`` with ``status=needs_verification`` instead.

        Raises:
            EvidenceMatcherError: If ``candidate_profile`` or
                ``job_requirements`` is not provided.
        """
        if candidate_profile is None:
            raise EvidenceMatcherError("candidate_profile is required.")
        if job_requirements is None:
            raise EvidenceMatcherError("job_requirements is required.")

        sources = self._collect_candidate_sources(candidate_profile)
        matches: list[EvidenceMatch] = []

        for requirement in job_requirements.requirements:
            try:
                matches.append(self._match_single_requirement(requirement, sources, candidate_profile))
            except Exception:
                # A single requirement's matching failure must never take
                # down the rest of the batch.
                matches.append(
                    EvidenceMatch(
                        requirement_id=requirement.id,
                        requirement=requirement.requirement,
                        status=EvidenceStatus.needs_verification,
                        evidence=[],
                        explanation=_MATCH_FAILURE_EXPLANATION,
                        confidence=_CONFIDENCE_MATCH_FAILURE,
                    )
                )

        return matches

    # -- Single-requirement matching -------------------------------------

    def _match_single_requirement(
        self,
        requirement: JobRequirement,
        sources: list[tuple[str, str]],
        candidate_profile: CandidateProfile,
    ) -> EvidenceMatch:
        """Match one requirement, deterministically first, then via LLM if needed."""
        deterministic_match = self._deterministic_match(requirement, sources)
        if deterministic_match is not None:
            return deterministic_match

        if self._llm_service is not None:
            return self._llm_match(requirement, candidate_profile)

        return EvidenceMatch(
            requirement_id=requirement.id,
            requirement=requirement.requirement,
            status=EvidenceStatus.no_evidence_found,
            evidence=[],
            explanation=_NO_EVIDENCE_EXPLANATION_TEMPLATE.format(requirement=requirement.requirement),
            confidence=_CONFIDENCE_DETERMINISTIC_NO_EVIDENCE,
        )

    # -- Deterministic matching --------------------------------------------

    @staticmethod
    def _collect_candidate_sources(profile: CandidateProfile) -> list[tuple[str, str]]:
        """Build a flat list of (source_label, text) pairs to search for evidence.

        Only fields relevant to evidence-finding are included. Contact
        details are irrelevant to requirement matching and are excluded.
        """
        sources: list[tuple[str, str]] = []

        for skill in profile.skills:
            if skill and skill.strip():
                sources.append(("skill", skill.strip()))

        if profile.summary and profile.summary.strip():
            sources.append(("summary", profile.summary.strip()))

        for experience in profile.experiences:
            if experience.role and experience.role.strip():
                sources.append(("experience", experience.role.strip()))
            if experience.organization and experience.organization.strip():
                sources.append(("experience", experience.organization.strip()))
            if experience.description and experience.description.strip():
                sources.append(("experience", experience.description.strip()))

        for education_entry in profile.education:
            if education_entry and education_entry.strip():
                sources.append(("education", education_entry.strip()))

        for project in profile.projects:
            if project.name and project.name.strip():
                sources.append(("project", project.name.strip()))
            if project.description and project.description.strip():
                sources.append(("project", project.description.strip()))
            for technology in project.technologies:
                if technology and technology.strip():
                    sources.append(("project", technology.strip()))

        for certification in profile.certifications:
            if certification and certification.strip():
                sources.append(("certification", certification.strip()))

        return sources

    @classmethod
    def _deterministic_match(
        cls,
        requirement: JobRequirement,
        sources: list[tuple[str, str]],
    ) -> EvidenceMatch | None:
        """Attempt to find evidence via direct keyword/phrase matching.

        Returns:
            An ``EvidenceMatch`` with ``status=evidence_found`` if a clear,
            explainable match is found; otherwise ``None`` (meaning
            deterministic matching was inconclusive, not that no evidence
            exists — the caller decides what to do next).
        """
        if not sources:
            return None

        scoped_sources = cls._scope_sources_for_category(requirement.category, sources)
        if not scoped_sources:
            return None

        requirement_lower = requirement.requirement.lower()
        terms = cls._extract_core_terms(requirement.requirement)

        matched_evidence: list[str] = []
        matched_via_skill = False

        for source_label, source_text in scoped_sources:
            source_lower = source_text.lower()

            is_match = False
            if source_label == "skill" and (
                source_lower in requirement_lower or source_lower in terms
            ):
                is_match = True
                matched_via_skill = True
            elif any(term and term in source_lower for term in terms):
                is_match = True
                if source_label == "skill":
                    matched_via_skill = True

            if is_match and source_text not in matched_evidence:
                matched_evidence.append(source_text)

        if not matched_evidence:
            return None

        confidence = (
            _CONFIDENCE_DETERMINISTIC_SKILL_MATCH
            if matched_via_skill
            else _CONFIDENCE_DETERMINISTIC_TEXT_MATCH
        )
        explanation = (
            f"The requirement '{requirement.requirement}' is explicitly "
            f"supported by information found in the candidate's submitted "
            f"profile."
        )

        return EvidenceMatch(
            requirement_id=requirement.id,
            requirement=requirement.requirement,
            status=EvidenceStatus.evidence_found,
            evidence=matched_evidence,
            explanation=explanation,
            confidence=confidence,
        )

    @staticmethod
    def _scope_sources_for_category(
        category: str | None,
        sources: list[tuple[str, str]],
    ) -> list[tuple[str, str]]:
        """Restrict candidate sources to those relevant to a credential category.

        For requirement categories that represent a formal credential (see
        ``_CREDENTIAL_CATEGORY_SOURCE_LABELS``), deterministic matching must
        only consider the corresponding structured source(s) — never
        unrelated free-text fields such as skills, experience descriptions,
        summary, or projects, where an incidental word overlap does not mean
        the candidate holds the credential.

        For every other category, all collected sources remain in scope,
        preserving prior behavior unchanged.
        """
        if not category:
            return sources

        allowed_labels = _CREDENTIAL_CATEGORY_SOURCE_LABELS.get(category.strip().lower())
        if allowed_labels is None:
            return sources

        return [source for source in sources if source[0] in allowed_labels]

    @staticmethod
    def _extract_core_terms(requirement_text: str) -> list[str]:
        """Extract normalized, searchable term phrases from requirement text.

        Strips common boilerplate lead-ins (e.g. "experience with",
        "3+ years of") and splits the remainder into individual candidate
        phrases, e.g. "Experience with Python and SQL" -> ["python", "sql"].
        """
        text = requirement_text.lower()
        for phrase in _BOILERPLATE_PHRASES:
            text = text.replace(phrase, " ")
        text = _YEARS_PATTERN.sub(" ", text)

        parts = _SPLIT_PATTERN.split(text)
        terms: list[str] = []
        for part in parts:
            cleaned = part.strip(" .;:()'\"")
            if cleaned and len(cleaned) >= 2:
                terms.append(cleaned)
        return terms

    # -- LLM-based semantic matching ---------------------------------------

    def _llm_match(
        self,
        requirement: JobRequirement,
        candidate_profile: CandidateProfile,
    ) -> EvidenceMatch:
        """Ask the LLM whether the requirement is semantically supported.

        Only used when deterministic matching is inconclusive. Any evidence
        returned by the LLM that cannot be verbatim-located in the supplied
        candidate context is discarded (never trusted blindly), and the
        status is downgraded to ``needs_verification`` if that leaves an
        ``evidence_found`` result with no traceable evidence.
        """
        context_text = self._build_llm_context(candidate_profile)
        prompt = _PROMPT_TEMPLATE.format(
            requirement_text=requirement.requirement,
            candidate_context=context_text,
        )

        try:
            raw_response = self._llm_service.generate_text(
                prompt=prompt,
                system_instruction=_SYSTEM_INSTRUCTION,
            )
        except LLMServiceError:
            return self._match_failure(requirement)

        payload = self._parse_json_response(raw_response)
        if payload is None:
            return self._match_failure(requirement)

        status_value = payload.get("status")
        if status_value not in _VALID_STATUS_VALUES:
            return self._match_failure(requirement)

        raw_evidence = payload.get("evidence") or []
        explanation = payload.get("explanation") or ""
        if not isinstance(raw_evidence, list) or not isinstance(explanation, str) or not explanation.strip():
            return self._match_failure(requirement)

        context_lower = context_text.lower()
        traceable_evidence = [
            item
            for item in raw_evidence
            if isinstance(item, str) and item.strip() and item.strip().lower() in context_lower
        ]

        status = EvidenceStatus(status_value)

        if status == EvidenceStatus.evidence_found and not traceable_evidence:
            # The LLM claimed evidence but none of it could be verified
            # against the candidate data we actually supplied — do not
            # trust it blindly.
            status = EvidenceStatus.needs_verification
            explanation = (
                "The language model reported possible evidence for this "
                "requirement, but it could not be verified against the "
                "candidate's submitted profile data. Human verification is "
                "recommended."
            )
            confidence = _CONFIDENCE_LLM_NEEDS_VERIFICATION
        elif status == EvidenceStatus.evidence_found:
            confidence = _CONFIDENCE_LLM_EVIDENCE_FOUND
        elif status == EvidenceStatus.needs_verification:
            confidence = _CONFIDENCE_LLM_NEEDS_VERIFICATION
        else:  # no_evidence_found
            traceable_evidence = []
            confidence = _CONFIDENCE_LLM_NO_EVIDENCE

        return EvidenceMatch(
            requirement_id=requirement.id,
            requirement=requirement.requirement,
            status=status,
            evidence=traceable_evidence,
            explanation=explanation,
            confidence=confidence,
        )

    def _match_failure(self, requirement: JobRequirement) -> EvidenceMatch:
        """Build the standard fallback result for an LLM matching failure."""
        return EvidenceMatch(
            requirement_id=requirement.id,
            requirement=requirement.requirement,
            status=EvidenceStatus.needs_verification,
            evidence=[],
            explanation=_MATCH_FAILURE_EXPLANATION,
            confidence=_CONFIDENCE_MATCH_FAILURE,
        )

    @staticmethod
    def _build_llm_context(profile: CandidateProfile) -> str:
        """Build the compact, PII-minimized candidate context sent to the LLM.

        Only fields relevant to evidence-finding are included; contact
        details (name, email, phone, location, links) are deliberately
        excluded because they are not needed to determine whether a
        requirement is supported.
        """
        context = {
            "summary": profile.summary,
            "skills": profile.skills,
            "experiences": [
                {
                    "role": experience.role,
                    "organization": experience.organization,
                    "description": experience.description,
                }
                for experience in profile.experiences
            ],
            "education": profile.education,
            "projects": [
                {
                    "name": project.name,
                    "description": project.description,
                    "technologies": project.technologies,
                }
                for project in profile.projects
            ],
            "certifications": profile.certifications,
        }
        return json.dumps(context, indent=2)

    @staticmethod
    def _parse_json_response(raw_response: str) -> dict | None:
        """Safely parse the LLM's raw text response into a JSON object.

        Returns ``None`` (rather than raising) on any parsing problem, so
        the caller can fall back to a ``needs_verification`` result instead
        of crashing.
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
