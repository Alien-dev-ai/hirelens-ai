"""Job Description Analyzer for HireLens AI.

Transforms unstructured job description text into structured, evidence-
grounded :class:`~src.models.schemas.JobRequirements`.

IMPORTANT PRODUCT BOUNDARY:
This module performs *requirement normalization only* — it reformats
requirements that are explicitly stated in the submitted job description
into a structured shape. It does NOT evaluate candidates, and it must never
invent required skills, preferred skills, years of experience, education
requirements, responsibilities, qualifications, technologies, or
certifications that are not explicitly stated in the JD text. If something
is unstated or unclear, it is omitted, or its importance is recorded as
``unclear`` rather than guessed as required/preferred.

This module depends only on the reusable :class:`GeminiLLMService`
abstraction (``src.services.llm_service``) — it never imports
``google.genai`` directly — and validates the LLM's output against the
existing ``JobRequirements`` Pydantic model before returning it.

SECURITY NOTE: Job description text and LLM responses are never logged by
this module.
"""

from __future__ import annotations

import json
import re

from pydantic import ValidationError

from src.models.schemas import JobRequirements
from src.services.llm_service import (
    GeminiLLMService,
    LLMQuotaExceededError,
    LLMServiceError,
)

# System instruction sent with every request. Establishes the analyzer's
# role and the hard grounding rules the model must follow.
_SYSTEM_INSTRUCTION = (
    "You are a strict data-normalization engine for a recruiting tool called "
    "HireLens AI. Your only job is to reformat a job description's text into "
    "structured JSON describing its stated requirements. You are NOT a "
    "recruiter, evaluator, or judge: you must never assess, score, rank, or "
    "compare any candidate against this job description, and you must never "
    "suggest how a candidate should be evaluated. This is requirement "
    "normalization, not candidate evaluation.\n\n"
    "GROUNDING RULES (follow strictly):\n"
    "- Extract only explicitly stated requirements that are literally "
    "present in the job description text.\n"
    "- Do not infer missing requirements (for example, do not assume a "
    "typical requirement for the role's title if the text does not state "
    "it).\n"
    "- Do not fabricate or hallucinate skills, qualifications, "
    "technologies, certifications, years of experience, or education "
    "requirements.\n"
    "- Preserve whether a requirement is required or preferred exactly as "
    "the text indicates. If the text does not clearly indicate whether a "
    "requirement is required or preferred, mark its importance as "
    "'unclear' rather than guessing.\n"
    "- If a field is unclear, ambiguous, or absent from the job "
    "description, omit it or leave it empty rather than guessing.\n"
    "- Return structured JSON only, with no prose, commentary, or "
    "markdown formatting around it."
)

_PROMPT_TEMPLATE = """\
Transform the following job description text into a single JSON object \
with EXACTLY this shape (use null/empty values when information is not \
explicitly present in the text — never invent a value):

{{
  "role_title": string or null,
  "requirements": [
    {{
      "id": string,
      "requirement": string,
      "category": string,
      "importance": "required" | "preferred" | "unclear"
    }}, ...
  ],
  "responsibilities": [string, ...],
  "warnings": [string, ...]
}}

Field notes:
- "id": a short stable identifier you assign to each requirement, e.g. \
"req-1", "req-2", ... in the order the requirements appear.
- "category": a short label for the kind of requirement, e.g. "skill", \
"experience", "education", "certification", "tool".
- "importance": use "required" or "preferred" only when the text clearly \
indicates that; otherwise use "unclear". Never guess.
- "responsibilities": duties/tasks explicitly listed for the role, as \
plain strings.
- "warnings": short notes about ambiguities, missing information, or gaps \
you noticed while extracting requirements (e.g. "Years of experience not \
specified for the 'Python' requirement."). Leave empty if there are none.

Remember the grounding rules: extract only explicitly stated \
requirements, do not infer, do not fabricate. If uncertain, omit the \
requirement or mark its importance as "unclear". Return JSON only — no \
markdown code fences, no explanation.

JOB DESCRIPTION TEXT:
\"\"\"
{jd_text}
\"\"\"
"""

# Matches a JSON payload wrapped in a markdown code fence, e.g. ```json ... ```
_CODE_FENCE_RE = re.compile(r"^```(?:json)?\s*(.*?)\s*```$", re.DOTALL | re.IGNORECASE)


class JobDescriptionAnalyzerError(Exception):
    """Raised for any failure while analyzing a job description.

    This covers invalid input, LLM service failures, and cases where the
    LLM's response cannot be parsed into a valid ``JobRequirements``.
    """


class JobDescriptionAnalyzer:
    """Normalizes job description text into structured ``JobRequirements``.

    Uses an injected :class:`GeminiLLMService` (or any compatible stub/mock
    exposing a ``generate_text(prompt, system_instruction=None) -> str``
    method) to perform the extraction, keeping this class independent of
    any specific LLM SDK.
    """

    def __init__(self, llm_service: GeminiLLMService) -> None:
        """Initialize the analyzer.

        Args:
            llm_service: An LLM service used to perform text generation.
                Any object implementing ``generate_text`` is accepted, so
                tests can inject a fake or mock in place of
                ``GeminiLLMService``.
        """
        self._llm_service = llm_service

    def analyze(self, jd_text: str) -> JobRequirements:
        """Analyze job description text and return validated ``JobRequirements``.

        Args:
            jd_text: Raw job description text.

        Returns:
            A ``JobRequirements`` built only from requirements explicitly
            stated in ``jd_text``.

        Raises:
            JobDescriptionAnalyzerError: If ``jd_text`` is empty/
                whitespace-only, the LLM service fails, the LLM's response
                is not valid JSON, or the parsed JSON does not match the
                ``JobRequirements`` schema.
        """
        if not jd_text or not jd_text.strip():
            raise JobDescriptionAnalyzerError(
                "Cannot analyze empty job description text: no content was "
                "provided."
            )

        prompt = _PROMPT_TEMPLATE.format(jd_text=jd_text)

        try:
            raw_response = self._llm_service.generate_text(
                prompt=prompt,
                system_instruction=_SYSTEM_INSTRUCTION,
            )
        except LLMQuotaExceededError as exc:
            raise JobDescriptionAnalyzerError(
                "Job description analysis failed because the Gemini API "
                "quota or rate limit has been reached. Please wait a bit "
                "and try again later."
            ) from exc
        except LLMServiceError as exc:
            raise JobDescriptionAnalyzerError(
                f"Job description analysis failed because the LLM service "
                f"could not generate a response: {exc}"
            ) from exc

        payload = self._parse_json_response(raw_response)

        try:
            return JobRequirements.model_validate(payload)
        except ValidationError as exc:
            raise JobDescriptionAnalyzerError(
                "The LLM's response did not match the expected job "
                "requirements structure."
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
            JobDescriptionAnalyzerError: If the response is empty, is not
                valid JSON, or does not decode to a JSON object.
        """
        if raw_response is None or not raw_response.strip():
            raise JobDescriptionAnalyzerError(
                "The LLM returned an empty response while analyzing the job "
                "description."
            )

        cleaned = raw_response.strip()
        fence_match = _CODE_FENCE_RE.match(cleaned)
        if fence_match:
            cleaned = fence_match.group(1).strip()

        try:
            payload = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise JobDescriptionAnalyzerError(
                "The LLM's response could not be parsed as valid JSON."
            ) from exc

        if not isinstance(payload, dict):
            raise JobDescriptionAnalyzerError(
                "The LLM's response was valid JSON but not a JSON object "
                "matching the job requirements structure."
            )

        return payload
