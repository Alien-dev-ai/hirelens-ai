"""Candidate Analyzer for HireLens AI.

Transforms unstructured candidate CV text into a structured, evidence-
grounded :class:`~src.models.schemas.CandidateProfile`.

IMPORTANT PRODUCT BOUNDARY:
This module performs *normalization only* — it reformats information that
is explicitly present in the submitted CV into a structured shape. It does
NOT evaluate, score, rank, or make any hiring/rejection/suitability
judgment about the candidate, and it must never invent skills, employers,
job titles, education, projects, certifications, links, years of
experience, or any other personal information that is not explicitly
supported by the CV text. If something is unclear or absent, the
corresponding field is left empty/omitted rather than guessed.

This module depends only on the reusable :class:`GroqLLMService`
abstraction (``src.services.llm_service``) — it never imports the Groq SDK
directly — and validates the LLM's output against the existing
``CandidateProfile`` Pydantic model before returning it.

SECURITY NOTE: CV text may contain candidate personal information and is
never logged by this module.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from src.models.schemas import CandidateProfile
from src.services.llm_service import (
    GroqLLMService,
    LLMQuotaExceededError,
    LLMServiceError,
)

# System instruction sent with every request. Establishes the analyzer's
# role and the hard grounding rules the model must follow.
_SYSTEM_INSTRUCTION = (
    "You are a strict data-normalization engine for a recruiting tool called "
    "HireLens AI. Your only job is to reformat a candidate's CV text into "
    "structured JSON. You are NOT a recruiter, evaluator, or judge: you must "
    "never assess, score, rank, or comment on the candidate's suitability "
    "for any role. This is normalization, not candidate evaluation.\n\n"
    "GROUNDING RULES (follow strictly):\n"
    "- Extract only explicitly supported information that is literally "
    "present in the CV text.\n"
    "- Do not infer missing facts (for example, do not guess a skill from a "
    "job title, or estimate years of experience that are not stated).\n"
    "- Do not fabricate or hallucinate skills, employers, job titles, "
    "education, projects, certifications, links, or any other detail.\n"
    "- If a field is unclear, ambiguous, or absent from the CV, omit it or "
    "leave it empty rather than guessing.\n"
    "- Return structured JSON only, with no prose, commentary, or "
    "markdown formatting around it."
)

_PROMPT_TEMPLATE = """\
Transform the following CV text into a single JSON object with EXACTLY \
this shape (all fields optional unless noted; use null/empty values when \
information is not explicitly present in the CV — never invent a value):

{{
  "full_name": string or null,
  "email": string or null,
  "phone": string or null,
  "location": string or null,
  "summary": string or null,
  "skills": [string, ...],
  "experiences": [
    {{
      "role": string or null,
      "organization": string or null,
      "start_date": string or null,
      "end_date": string or null,
      "description": string or null
    }}, ...
  ],
  "education": [string, ...],
  "projects": [
    {{
      "name": string,
      "description": string or null,
      "technologies": [string, ...],
      "url": string or null
    }}, ...
  ],
  "certifications": [string, ...],
  "links": [string, ...]
}}

Remember the grounding rules: extract only explicitly supported \
information, do not infer, do not fabricate. If uncertain, omit it. \
Return JSON only — no markdown code fences, no explanation.

CV TEXT:
\"\"\"
{cv_text}
\"\"\"
"""

# Matches a JSON payload wrapped in a markdown code fence, e.g. ```json ... ```
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class CandidateAnalyzerError(Exception):
    """Raised for any failure while analyzing a candidate CV.

    This covers invalid input, LLM service failures, and cases where the
    LLM's response cannot be parsed into a valid ``CandidateProfile``.
    """


class CandidateAnalyzer:
    """Normalizes candidate CV text into a structured ``CandidateProfile``.

    Uses an injected :class:`GroqLLMService` (or any compatible stub/mock
    exposing a ``generate_text(prompt, system_instruction=None) -> str``
    method) to perform the extraction, keeping this class independent of
    any specific LLM SDK.
    """

    def __init__(self, llm_service: GroqLLMService) -> None:
        """Initialize the analyzer.

        Args:
            llm_service: An LLM service used to perform text generation.
                Any object implementing ``generate_text`` is accepted, so
                tests can inject a fake or mock in place of
                ``GroqLLMService``.
        """
        self._llm_service = llm_service

    def analyze(self, cv_text: str) -> CandidateProfile:
        """Analyze CV text and return a validated ``CandidateProfile``.

        Args:
            cv_text: Extracted CV text (e.g. from
                ``src.extraction.document_extractor``).

        Returns:
            A ``CandidateProfile`` built only from information explicitly
            supported by ``cv_text``.

        Raises:
            CandidateAnalyzerError: If ``cv_text`` is empty/whitespace-only,
                the LLM service fails, the LLM's response is not valid JSON,
                or the parsed JSON does not match the ``CandidateProfile``
                schema.
        """
        if not cv_text or not cv_text.strip():
            raise CandidateAnalyzerError(
                "Cannot analyze empty CV text: no content was provided."
            )

        prompt = _PROMPT_TEMPLATE.format(cv_text=cv_text)

        try:
            raw_response = self._llm_service.generate_text(
                prompt=prompt,
                system_instruction=_SYSTEM_INSTRUCTION,
            )
        except LLMQuotaExceededError as exc:
            raise CandidateAnalyzerError(
                "Candidate analysis failed because the Groq API quota or "
                "rate limit has been reached. Please wait a bit and try "
                "again later."
            ) from exc
        except LLMServiceError as exc:
            raise CandidateAnalyzerError(
                f"Candidate analysis failed because the LLM service could "
                f"not generate a response: {exc}"
            ) from exc

        payload = self._parse_json_response(raw_response)

        try:
            return CandidateProfile.model_validate(payload)
        except ValidationError as exc:
            raise CandidateAnalyzerError(
                "The LLM's response did not match the expected candidate "
                "profile structure."
            ) from exc

    @staticmethod
    def _parse_json_response(raw_response: str) -> dict:
        """Safely parse the LLM's raw text response into a JSON object.

        Strips a surrounding markdown code fence (e.g. ```json ... ```) if
        present, then parses the result as JSON.

        Args:
            raw_response: The raw text returned by the LLM service.

        Returns:
            The parsed JSON object as a ``dict``.

        Raises:
            CandidateAnalyzerError: If the response is empty, is not valid
                JSON, or does not decode to a JSON object.
        """
        if raw_response is None or not raw_response.strip():
            raise CandidateAnalyzerError(
                "The LLM returned an empty response while analyzing the CV."
            )

        cleaned = raw_response.strip()
        fence_match = _CODE_FENCE_RE.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise CandidateAnalyzerError(
                "The LLM's response could not be parsed as valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise CandidateAnalyzerError(
                "The LLM's response was valid JSON but not a JSON object "
                "matching the candidate profile structure."
            )

        return payload
