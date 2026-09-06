"""Interview Question Generator for HireLens AI.

Generates evidence-grounded, structured interview questions
(:class:`~src.models.schemas.InterviewQuestion`) that help a recruiter
verify or clarify each job requirement, based on the evidence status found
for it by the Evidence Matcher.

IMPORTANT PRODUCT BOUNDARY / CRITICAL RULE:
This module does NOT score, rank, recommend, accept, reject, or determine
whether a candidate should be hired. It only prepares neutral, evidence-
grounded questions for a human recruiter to ask. "No evidence found in the
submitted CV" does NOT mean the candidate does not possess that skill or
qualification — it means only that no explicit supporting evidence was
found in the submitted documents — and every question generated for a
``no_evidence_found`` requirement must stay neutral and must never imply
the candidate lacks the skill.

ARCHITECTURE:
Question generation is a language-generation task (unlike the Evidence
Matcher's extraction task), so an injected LLM is the primary path when
available — it produces more natural, specifically-tailored questions.
A deterministic, template-based fallback is used whenever no LLM service
is injected at all (avoiding an unnecessary call), and per-requirement
whenever the LLM call fails, returns malformed output, or fails schema
validation — so a single bad LLM response never crashes the whole batch.

This module depends only on the reusable :class:`GroqLLMService`
abstraction (``src.services.llm_service``) — it never imports the Groq SDK
directly — and every generated question is validated against the existing
``InterviewQuestion`` Pydantic model.

SECURITY NOTE: Candidate personal information, raw prompts, and raw LLM
responses are never logged by this module. Only the requirement text and
the evidence excerpts already surfaced by the Evidence Matcher (never full
CandidateProfile contact details) are sent to the LLM.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from src.models.schemas import (
    CandidateProfile,
    EvidenceMatch,
    EvidenceStatus,
    InterviewQuestion,
    JobRequirement,
    JobRequirements,
    RequirementImportance,
)
from src.services.llm_service import GroqLLMService, LLMServiceError

# Matches a JSON payload wrapped in a markdown code fence, e.g. ```json ... ```
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)

# System instruction sent with every LLM question-generation request.
_SYSTEM_INSTRUCTION = (
    "You are a strict interview-preparation assistant for a recruiting tool "
    "called HireLens AI. Your only job is to draft ONE structured interview "
    "question that helps a human recruiter verify or clarify a single job "
    "requirement. You are NOT a recruiter, evaluator, or judge: you must "
    "never assign the candidate a score or a ranking, and you must never "
    "state or imply whether the candidate should be hired, accepted, or "
    "turned down for any role. You do not make hiring decisions — the "
    "human recruiter does.\n\n"
    "GROUNDING RULES (follow strictly):\n"
    "- Base the question only on the requirement text and evidence excerpts "
    "provided to you below. Do not invent additional candidate experience, "
    "skills, or achievements that are not present in what was provided.\n"
    "- If the evidence status is 'evidence_found', write a question that "
    "invites the candidate to elaborate on or walk through the specific "
    "evidence provided — do not judge or assess that evidence.\n"
    "- If the evidence status is 'no_evidence_found', the question must be "
    "neutral and exploratory. It must NEVER imply or state that the "
    "candidate lacks the skill, is missing the qualification, or is "
    "unqualified — 'no evidence found' only means no explicit evidence was "
    "present in the documents reviewed, not that the candidate lacks the "
    "skill.\n"
    "- If the evidence status is 'needs_verification' or unknown, write a "
    "neutral question that helps clarify the ambiguous or incomplete "
    "evidence.\n"
    "- Return structured JSON only, with no prose, commentary, or markdown "
    "formatting around it."
)

_PROMPT_TEMPLATE = """\
Draft one interview question for the following job requirement, given the \
evidence status found for it in the candidate's submitted documents.

JOB REQUIREMENT:
"{requirement_text}"

REQUIREMENT IMPORTANCE: {importance}

EVIDENCE STATUS: {status}

EVIDENCE EXCERPTS FOUND IN THE CANDIDATE'S SUBMITTED DOCUMENTS (the only \
candidate information you may reference; empty if none):
{evidence_block}

Return a single JSON object with EXACTLY this shape:

{{
  "question": string,
  "focus_requirement": string,
  "reason": string,
  "priority": "high" | "medium" | "low"
}}

Rules:
- "question": the interview question itself, addressed to the candidate.
- "focus_requirement": copy this exact text verbatim: "{requirement_text}"
- "reason": a short, neutral explanation of why this question is being \
asked, grounded in the evidence status above. It must never claim the \
candidate lacks the skill, is unqualified, or is unsuitable, and must \
never recommend a hiring decision.
- "priority": how important it is for the recruiter to ask this question, \
considering the requirement's importance ({importance}) and evidence \
status ({status}).
- Return JSON only — no markdown code fences, no explanation outside the \
JSON object.
"""


class InterviewQuestionGeneratorError(Exception):
    """Raised for invalid usage of :class:`InterviewQuestionGenerator`.

    This is reserved for programming errors (e.g. missing required
    arguments). Ordinary per-requirement generation problems are instead
    handled internally via the deterministic fallback, so a single
    requirement never crashes the whole batch.
    """


class InterviewQuestionGenerator:
    """Generates evidence-grounded interview questions for job requirements.

    Uses an injected :class:`GroqLLMService` (or any compatible stub/mock
    exposing ``generate_text(prompt, system_instruction=None) -> str``) as
    the primary question-generation path when available, and a
    deterministic, template-based fallback otherwise (or whenever the LLM
    call fails, returns malformed output, or fails schema validation for a
    given requirement).
    """

    def __init__(self, llm_service: GroqLLMService | None = None) -> None:
        """Initialize the generator.

        Args:
            llm_service: Optional LLM service used to draft natural-language
                interview questions. If omitted, questions are generated
                deterministically from templates for every requirement,
                avoiding any LLM calls.
        """
        self._llm_service = llm_service

    def generate(
        self,
        candidate_profile: CandidateProfile,
        job_requirements: JobRequirements,
        evidence_matches: list[EvidenceMatch],
    ) -> list[InterviewQuestion]:
        """Generate one interview question per job requirement.

        Args:
            candidate_profile: The candidate's structured profile. Not sent
                to the LLM directly — the evidence excerpts already
                surfaced by the Evidence Matcher (via ``evidence_matches``)
                are the vetted, minimal representation of relevant
                candidate information, so re-sending the full profile would
                only add redundant PII exposure without adding grounding
                value. It is accepted here for interface consistency with
                the rest of the pipeline and to allow future use.
            job_requirements: The job's structured requirements.
            evidence_matches: The Evidence Matcher's results. Matched to
                requirements by ``requirement_id``.

        Returns:
            A list of ``InterviewQuestion`` objects, exactly one per
            requirement in ``job_requirements.requirements``, in the same
            order. Each question's ``focus_requirement`` is always set to
            that requirement's exact text, so it is reliably traceable to
            its requirement regardless of how it was generated.

        Raises:
            InterviewQuestionGeneratorError: If ``candidate_profile``,
                ``job_requirements``, or ``evidence_matches`` is not
                provided.
        """
        if candidate_profile is None:
            raise InterviewQuestionGeneratorError("candidate_profile is required.")
        if job_requirements is None:
            raise InterviewQuestionGeneratorError("job_requirements is required.")
        if evidence_matches is None:
            raise InterviewQuestionGeneratorError("evidence_matches is required.")

        matches_by_requirement_id = {match.requirement_id: match for match in evidence_matches}

        questions: list[InterviewQuestion] = []
        for requirement in job_requirements.requirements:
            match = matches_by_requirement_id.get(requirement.id)
            question = self._generate_single_question(requirement, match)
            questions.append(question)

        return questions

    # -- Per-requirement generation ----------------------------------------

    def _generate_single_question(
        self,
        requirement: JobRequirement,
        match: EvidenceMatch | None,
    ) -> InterviewQuestion:
        """Generate one question, preferring the LLM but always falling back safely."""
        if self._llm_service is not None:
            llm_question = self._llm_generate_question(requirement, match)
            if llm_question is not None:
                # Never trust the LLM's own copy of the requirement text —
                # force it to the ground truth so the connection to the
                # correct requirement is always guaranteed.
                return llm_question.model_copy(update={"focus_requirement": requirement.requirement})

        return self._deterministic_question(requirement, match)

    def _llm_generate_question(
        self,
        requirement: JobRequirement,
        match: EvidenceMatch | None,
    ) -> InterviewQuestion | None:
        """Attempt to generate a question via the LLM. Returns None on any failure."""
        status = match.status if match is not None else EvidenceStatus.needs_verification
        evidence = match.evidence if match is not None else []
        evidence_block = "\n".join(f"- {item}" for item in evidence) if evidence else "(none)"

        prompt = _PROMPT_TEMPLATE.format(
            requirement_text=requirement.requirement,
            importance=requirement.importance.value,
            status=status.value,
            evidence_block=evidence_block,
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
            question = InterviewQuestion.model_validate(payload)
        except ValidationError:
            return None

        # Guard against the LLM just restating the requirement instead of
        # asking a question about it (e.g. under degraded/rushed output).
        # Deliberately simple: exact match after trimming whitespace and
        # case-folding only — no similarity threshold, so a legitimate
        # question that happens to reuse requirement wording is never
        # rejected. Treated the same as any other invalid LLM output: fall
        # back to the deterministic template rather than raising.
        if self._is_requirement_echo(question.question, requirement.requirement):
            return None

        return question

    @staticmethod
    def _is_requirement_echo(question_text: str, requirement_text: str) -> bool:
        """True if ``question_text`` is just the requirement text restated."""
        return question_text.strip().casefold() == requirement_text.strip().casefold()

    def _deterministic_question(
        self,
        requirement: JobRequirement,
        match: EvidenceMatch | None,
    ) -> InterviewQuestion:
        """Build a template-based question without calling the LLM.

        This is guaranteed to succeed and produce a schema-valid question,
        so it also serves as the ultimate fallback if the LLM path fails.
        """
        status = match.status if match is not None else None
        evidence = match.evidence if match is not None else []

        if status == EvidenceStatus.evidence_found:
            question, reason = self._evidence_found_question(requirement, evidence)
        elif status == EvidenceStatus.no_evidence_found:
            question, reason = self._no_evidence_found_question(requirement)
        else:
            # Covers EvidenceStatus.needs_verification and the case where no
            # evidence match was found at all for this requirement.
            question, reason = self._needs_verification_question(requirement, match)

        return InterviewQuestion(
            question=question,
            focus_requirement=requirement.requirement,
            reason=reason,
            priority=self._determine_priority(requirement.importance, status),
        )

    @staticmethod
    def _evidence_found_question(requirement: JobRequirement, evidence: list[str]) -> tuple[str, str]:
        snippet = evidence[0] if evidence else requirement.requirement
        question = (
            f"You mentioned the following in your submitted documents: "
            f"\"{snippet}\" — can you walk me through this in more detail, "
            f"including your specific role and the technical decisions you "
            f"made, as it relates to '{requirement.requirement}'?"
        )
        reason = (
            f"Evidence for '{requirement.requirement}' was found in the "
            f"candidate's submitted documents; this question helps verify "
            f"and explore that evidence in more depth."
        )
        return question, reason

    @staticmethod
    def _no_evidence_found_question(requirement: JobRequirement) -> tuple[str, str]:
        question = (
            f"Can you describe any experience you have with "
            f"'{requirement.requirement}' that may not be included in your "
            f"submitted documents?"
        )
        reason = (
            f"No explicit evidence for '{requirement.requirement}' was "
            f"found in the submitted documents. This does not mean the "
            f"candidate is missing this qualification — it means only that "
            f"no supporting evidence was present in the documents "
            f"reviewed. This question offers a neutral way to clarify it "
            f"directly."
        )
        return question, reason

    @staticmethod
    def _needs_verification_question(requirement: JobRequirement, match: EvidenceMatch | None) -> tuple[str, str]:
        question = (
            f"Could you clarify or provide more detail about your "
            f"experience related to '{requirement.requirement}'?"
        )
        if match is not None:
            reason = (
                f"The evidence found for '{requirement.requirement}' was "
                f"partial or ambiguous; this question helps clarify it "
                f"directly with the candidate."
            )
        else:
            reason = (
                f"No evidence assessment was available for "
                f"'{requirement.requirement}'; this question helps clarify "
                f"it directly with the candidate."
            )
        return question, reason

    @staticmethod
    def _determine_priority(importance: RequirementImportance, status: EvidenceStatus | None) -> str:
        """Derive a simple priority for asking this question.

        A required requirement with no confirmed evidence is the most
        important to raise with the candidate; a preferred/unclear
        requirement that already has confirmed evidence is the least
        urgent to revisit.
        """
        evidence_confirmed = status == EvidenceStatus.evidence_found

        if importance == RequirementImportance.required:
            return "medium" if evidence_confirmed else "high"
        if importance == RequirementImportance.preferred:
            return "low" if evidence_confirmed else "medium"
        return "low" if evidence_confirmed else "medium"  # unclear importance

    # -- JSON parsing --------------------------------------------------------

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
