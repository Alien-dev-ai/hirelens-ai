# HireLens AI — Failure Analysis

This document records real failure scenarios encountered or deliberately
exercised in this project: one from actual development history (the
Gemini→Groq migration), several reproduced by the evaluation package's
mocked failure tests (`TC-08`, `TC-09`), and one genuine product bug this
evaluation package itself discovered by actually running a live case against
the real Groq API (`TC-10`). Nothing here is hypothetical dressed up as
observed; each scenario states plainly whether it is a historical
development failure, a reproduced/simulated one, or a live defect this
evaluation found and fixed — with the original failure preserved, not
erased, and the before/after evidence shown.

---

## 1. Gemini API quota exhaustion (historical development failure)

**Scenario:** During development, HireLens originally used Google's Gemini
API (`GeminiLLMService`, reading a `GEMINI_API_KEY` / `GEMINI_MODEL`) as its
LLM provider for candidate/JD analysis, evidence matching, and analysis
synthesis.

**Root cause:** The Gemini free-tier quota was exhausted during
development/testing use, surfacing as an HTTP 429 response with a
`RESOURCE_EXHAUSTED` status. This is a real constraint of Gemini's free
tier, not a bug in HireLens's own code — but it meant the workflow could not
complete whenever the quota was hit, regardless of how correct the rest of
the pipeline was.

**Impact:** The candidate analysis workflow could not complete while the
quota was exhausted. Any recruiter attempting an analysis during that window
would see a failure with no useful path forward other than waiting for the
quota to reset.

**Resolution:** The project was migrated from Gemini to Groq as the LLM
provider (see git commit `fd6f623`, "Migrate LLM provider to Groq; add
deterministic alignment scoring and recruiter analysis"). `llm_service.py`
was rewritten around the Groq SDK, preserving the same
`generate_text(prompt, system_instruction=None)` interface so every
downstream service (`CandidateAnalyzer`, `JobDescriptionAnalyzer`,
`EvidenceMatcher`, `RecruiterAnalysisGenerator`,
`InterviewQuestionGenerator`) required no changes beyond the rename from
`GeminiLLMService` to `GroqLLMService`. `LLMQuotaExceededError` was kept (and
extended with `LLMAuthenticationError`, `LLMInvalidModelError`,
`LLMConnectionError`) so the same quota-aware error handling in
`HireLensPipelineError.is_quota_exceeded` and the Streamlit UI (`app.py`)
continued to work unchanged.

**Important caveat — this is not solved, only mitigated:** Groq is not
immune to the same class of failure. It has its own rate limits and quotas,
and can return its own 429/`RateLimitError` responses (mapped to the same
`LLMQuotaExceededError` in `llm_service.py`), it can reject an invalid/renamed
model (`LLMInvalidModelError`), and it can be unreachable due to network
issues (`LLMConnectionError`). The migration did not make provider failure
impossible — it made the *handling* of provider failure explicit,
typed, and provider-agnostic at the boundary (`llm_service.py` is the only
module that imports a provider SDK at all), so a future provider swap would
again require changing only that one module.

**Verified by:** `TC-09` in the evaluation package reproduces both a
simulated quota-exceeded failure and a simulated connection failure against
the *current* Groq-based pipeline (using a scripted fake LLM service that
raises `LLMQuotaExceededError` / `LLMConnectionError` — no real network call
is made), confirming that `HireLensPipelineError.is_quota_exceeded` still
correctly distinguishes the two, and that neither failure crashes the
pipeline with an unwrapped exception.

**Lessons:**

- Provider limits (free-tier or otherwise) are an operational dependency,
  not an edge case — they will be hit in real usage, not just in testing.
- An AI-dependent workflow needs explicit, typed provider-failure handling,
  not a bare `except Exception`, so the UI can distinguish "try again in a
  bit" from "something is actually broken."
- Isolating all provider-SDK usage behind one internal abstraction
  (`GroqLLMService` / `LLMServiceError` and its subclasses) made the
  provider migration itself a contained, single-module change instead of a
  rewrite of every analysis service.
- A clear, specific, user-facing error message is strictly better than a
  generic "something went wrong" — see `app.py`'s handling of
  `HireLensPipelineError.is_quota_exceeded`.

---

## 2. Ambiguous evidence downgraded rather than guessed

**Scenario:** A candidate's documented experience is *related to* a
requirement but does not clearly confirm it (e.g. "worked with cloud-based
deployment tools" vs. a stated requirement for "cloud infrastructure
experience"). Deterministic keyword matching is inconclusive, so the
Evidence Matcher falls back to LLM semantic matching.

**Root cause:** Natural-language CVs and job descriptions frequently
describe the same underlying skill in different words. A purely
deterministic keyword matcher would either (a) miss this evidence entirely
(false negative) or (b) require an ever-growing, brittle synonym list to
catch it. An LLM-based semantic fallback closes some of this gap, but
introduces a new risk: the LLM could claim confident evidence that isn't
actually well-supported.

**System behavior:** `EvidenceMatcher._llm_match` (in
`src/services/evidence_matcher.py`) never trusts the LLM's evidence claim
blindly — it verifies that any evidence string returned is actually
present (verbatim, case-insensitive) in the candidate context that was sent.
If the LLM reports `evidence_found` but the evidence it cites cannot be
verified against the supplied candidate data, the result is *downgraded* to
`needs_verification` rather than accepted at face value. This means the
system is deliberately biased toward flagging ambiguity for human review
rather than resolving it silently.

**Impact:** The recruiter sees a `needs_verification` status with a neutral
explanation, rather than either a false "confirmed" match or a false "no
evidence" result. This is by design, not a defect — but it does mean
HireLens will sometimes ask a recruiter to verify something a human reader
might have confidently resolved themselves from context.

**Fix/mitigation:** Already implemented as described above — the
evidence-traceability check is not a proposed fix, it is the current,
tested behavior (see `tests/test_evidence_matcher.py` and this package's
`TC-05`).

**Remaining limitation:** The verbatim-substring traceability check is
itself fairly strict — it can undercount evidence that is paraphrased
faithfully by the LLM but does not appear as an exact substring of the
supplied context. This trades a small number of avoidable
"needs_verification" results for a stronger guarantee against fabricated
evidence, which is the direction this product should err in.

---

## 3. Malformed or non-JSON LLM response

**Scenario:** Every LLM-backed stage (`CandidateAnalyzer`,
`JobDescriptionAnalyzer`, `EvidenceMatcher`'s LLM fallback,
`RecruiterAnalysisGenerator`, `InterviewQuestionGenerator`) asks the model to
return structured JSON, and expects that JSON to validate against a
specific Pydantic schema. LLM output is not guaranteed to be well-formed
JSON, or to match the schema, on every call.

**Root cause:** LLMs occasionally wrap JSON in markdown code fences, add
stray prose, omit a required field, or return a shape that doesn't validate
even when it is syntactically valid JSON.

**System behavior:** Handling differs meaningfully by stage, and this
difference is itself a documented design decision:

- `CandidateAnalyzer` and `JobDescriptionAnalyzer` are *normalization*
  tasks with no safe deterministic fallback (there is no reasonable
  "template" candidate profile or job requirement list) — a parse or
  validation failure here raises `CandidateAnalyzerError` /
  `JobDescriptionAnalyzerError`, which the pipeline wraps as
  `HireLensPipelineError` and stops the run. The recruiter sees a clear
  error rather than a wrong, silently-accepted profile or requirement list.
- `RecruiterAnalysisGenerator` and `InterviewQuestionGenerator` are
  *language-generation* tasks with a safe deterministic fallback available
  (a template built directly from the already-computed evidence/score) — a
  parse or validation failure here is caught internally and the
  deterministic fallback is used instead, so a single bad LLM response
  never stops the pipeline at these later stages.
- `EvidenceMatcher`'s LLM fallback similarly degrades to
  `needs_verification` (see Scenario 2) rather than raising, since an
  inconclusive match is itself a valid, useful outcome.

**Impact:** A malformed response at an early, unrecoverable stage
(candidate or JD analysis) means the recruiter has to retry the whole
analysis; a malformed response at a later, recoverable stage is invisible
to the recruiter (they get the deterministic template output for that one
section instead).

**Fix/mitigation:** This tiered strategy (fail fast where there's no safe
fallback, degrade gracefully where there is) is the current implementation,
not a proposed change.

**Remaining limitation:** When the deterministic fallback is used for
`RecruiterAnalysisGenerator` or `InterviewQuestionGenerator`, the recruiter
currently has no visible indicator that a fallback (rather than the
LLM-synthesized version) was used for that section. Surfacing that
distinction in the UI (e.g. a small "generated from template" note) is a
reasonable future improvement, not yet implemented.

---

## 4. Invalid or corrupted document input

**Scenario:** A recruiter uploads a file that is not a valid CV document at
all — wrong file type, a file that doesn't exist at the given path, or a
file with a `.docx`/`.pdf` extension whose actual contents are corrupted or
not a real document.

**Root cause:** Uploaded files come from outside the system's control; any
of the above can happen from an honest mistake (wrong file selected,
partial/interrupted upload) with no malicious intent required.

**System behavior:** `extract_document()` (in
`src/extraction/document_extractor.py`) never raises for any of these
cases — it always returns an `ExtractedDocument` with `extraction_success`
and a specific `error_message` (e.g. "Unsupported file type: '.txt'...",
"File not found: ...", "Failed to read DOCX file: ..."). The pipeline
(`HireLensPipeline._extract_cv`) checks `extraction_success` and raises
`HireLensPipelineError` with that message if extraction failed, *before* any
LLM-backed stage (or the LLM service itself) is even constructed — no API
quota is spent on an input that was never going to produce a valid
analysis.

**Impact:** The recruiter sees a specific, actionable error message (wrong
file type / file not found / corrupted document) rather than a generic
crash or a confusing downstream failure from a later stage trying to
process empty text.

**Fix/mitigation:** Already implemented; this is the current, tested
behavior (see `tests/test_extraction.py` and this package's `TC-08`, which
additionally proves — via poison-pill fakes for every downstream service —
that no downstream stage runs after an extraction failure).

**Remaining limitation:** Currently only `.pdf` and `.docx` are supported.
A recruiter with a `.doc` (legacy Word), `.rtf`, plain-text, or
LinkedIn-export CV gets a clear "unsupported file type" error rather than a
best-effort conversion. This is a scope limitation, not a bug — see the
README's "Future Improvements" list.

---

## 5. `CandidateExperience.role` schema mismatch — discovered live by TC-10

**This is a real product bug this evaluation package found by actually
running the real pipeline against the real Groq API — not a simulated or
hypothetical scenario, and not one caught by the existing 157-test suite.**
The sequence below is preserved in full, including the original failure,
per this evaluation's own ground rules.

### The original failure (not hidden, not erased)

Running `TC-10` (Input variation: a candidate CV written as dense prose with
no `"Job Title — Company"` heading) live, for the first time, produced:

```
Running TC-10: Input variation (formatting / wording) ... ERROR (2.8211s)
    -> HireLensPipelineError: Candidate analysis failed: The LLM's response
       did not match the expected candidate profile structure.
```

This was recorded as `ERROR` (an unhandled exception from the real pipeline,
not a clean assertion failure) before anything was changed. See the
`live_run_artifacts/` history and `results.json` run log for this original
outcome.

### Root cause

`docs/test_data/candidate_variation.docx` (fictional candidate "Riley
Morgan") describes two work experiences without ever stating an explicit
job title — e.g. *"Prior role at Fenwick Data (2020-present) where
day-to-day work included API development..."*. Reproducing the exact call
made by `CandidateAnalyzer` showed Groq correctly following its own
grounding instructions ("do not infer... omit rather than guess") and
returning `"role": null` for both experience entries, since no title is
stated anywhere in the CV.

`CandidateExperience.role` in `src/models/schemas.py` was declared as:

```python
role: str = Field(..., description="Job title or role held by the candidate.")
```

— a **required, non-nullable** string, and the *only* required field in
`CandidateExperience` (every sibling field — `organization`, `start_date`,
`end_date`, `description` — was already optional). Pydantic rejected
`role: null`, `CandidateAnalyzer` raised `CandidateAnalyzerError`, and the
pipeline failed the entire analysis for an entirely realistic CV.

Corroborating evidence this was an unintentional oversight, not a
deliberate invariant: `src/services/evidence_matcher.py` already guarded
`if experience.role and experience.role.strip()` before using the field —
the rest of the codebase was already written assuming `role` could be
falsy/`None`.

### Classification

**Product bug** — not an evaluation-harness bug (the test CV was an
honestly-written, realistic variation, not an adversarial input designed to
break the schema), not a provider/API failure (Groq responded correctly and
followed its instructions), and not an "expected limitation" (it is
directly fixable and affects a common, unremarkable CV style).

### Fix applied

Two changes, both approved before being made, no unrelated changes:

1. `src/models/schemas.py` — `CandidateExperience.role` changed from
   `str` (required) to `str | None = None`, matching every other field in
   the model.
2. `src/services/candidate_analyzer.py` — the prompt's documented JSON
   shape for `"role"` changed from `string` to `string or null`, so the
   instructions the LLM is given now match what the schema actually
   accepts.

No other file was changed to make this fix work — in particular, the
evidence matcher and every other consumer of `experience.role` already
handled `None` correctly.

### Verification (before → after)

| | Before fix | After fix |
|---|---|---|
| Relevant existing tests (`test_models.py`, `test_candidate_analyzer.py`, `test_evidence_matcher.py`, `test_hirelens_pipeline.py`) | — (bug found via live run, not by the existing suite) | 65 passed, 0 failed |
| Full `pytest` suite | 157 passed | 157 passed (unchanged — the fix broadens a field, it doesn't remove behavior anything relied on) |
| `TC-10` live run | `ERROR` — `HireLensPipelineError: Candidate analysis failed: The LLM's response did not match the expected candidate profile structure.` (2.82s) | `PASS` (59.56s) — `overall_score=72`, `alignment_label=Moderate Alignment`, all 9 parsed requirements matched, no forbidden phrase |

No evaluation criterion was loosened to make `TC-10` pass — the pass/fail
criteria in `test_cases.json` for `TC-10` are unchanged from before the bug
was found; the pipeline itself now produces a schema-valid, correctly
structured result on the same real CV and the same real Groq call.

### Impact if this had shipped unfixed

Any real candidate whose CV describes work experience without a distinct
job-title line — a common pattern in prose-style resumes, career-change
narratives, or contract/freelance descriptions — would have hard-failed
the *entire* analysis with a generic "did not match expected structure"
error, rather than the graceful "field omitted, everything else proceeds"
behavior every other optional field already received.

### Remaining limitation

This fix addresses `role` specifically because it was the only required
field in `CandidateExperience`. It does not constitute a full audit of
every schema field across the codebase for the same class of issue — that
would be a reasonable follow-up (a schema-wide review of which fields are
truly always present vs. sometimes legitimately absent), but was out of
scope for this fix, which was intentionally kept minimal and targeted at
the specific failure `TC-10` surfaced.

**Lessons:**

- A live evaluation case caught a real bug that 157 passing mocked/fake
  unit tests did not, because none of those tests happened to construct a
  CV where an experience entry has no stated title. Mocked tests validate
  that the code does what the test author expected; only a real model
  call can surface what a real model actually does with realistic,
  unscripted input.
- "Never infer, omit if not stated" as a grounding principle has to be
  backed by schemas that actually allow omission everywhere it's
  legitimate — a single overlooked required field undoes the guarantee for
  every CV that hits it.
- The smallest fix that matches the codebase's own existing conventions
  (here: make `role` consistent with its sibling fields, which downstream
  code already treated defensively) is preferable to a larger, speculative
  rewrite.
