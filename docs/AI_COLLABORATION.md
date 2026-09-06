# AI Collaboration Note

## 1. Overview

HireLens AI was developed using AI-assisted engineering. AI tools were used
to help scaffold and implement code, draft and edit documentation, run
repository inspections, and assist with routine Git operations. Product
direction, architecture decisions, evaluation criteria, acceptance or
rejection of any proposed change, and final submission decisions remained
human-owned throughout.

This note documents that collaboration honestly, using only what is
verifiable from the repository itself: `git log`, commit messages, the
existing documentation (`README.md`, `docs/CASE_STUDY.md`), the test suite
(`tests/`), and the evaluation package (`evaluation/`). It does not claim
that AI independently built, owned, or made decisions about the product.

## 2. AI Tools Used

**Claude Code** was used as an interactive engineering assistant across the
project's later development sessions: inspecting the existing repository
state (source, tests, docs, git history), implementing and editing code and
documentation changes on request, running the test suite, and performing
Git operations (staging, committing, pushing) under explicit human
direction and review at each step. The evaluation-mode-count correction to
`docs/CASE_STUDY.md` (committed as `3609caa`, see §6) and the creation of
this file are direct, first-hand examples of that role.

**ChatGPT** was used elsewhere in the project's development process (e.g.
for drafting, ideation, or reviewing text/approach outside of what Git
tracks). Unlike Claude Code's session-based work, ChatGPT's specific
contributions are not separately attributable from repository artifacts
alone — there is no commit trailer, comment, or file in this repository
that isolates what ChatGPT produced versus what a human wrote or edited
afterward. This note does not assign ChatGPT specific claims (files,
commits, or outcomes) that the repository cannot support.

**Groq** (`groq` Python SDK, via `src/services/llm_service.py`) is
explicitly **not** described here as a development-assistance tool. It is
the application's runtime LLM provider — used at runtime by the shipped
product to normalize candidate/JD text, perform semantic evidence matching,
and synthesize recruiter analysis and interview questions (see
`README.md`'s "Integrations" section). It did not assist in writing
HireLens' own code or documentation; it is the AI capability HireLens
*ships with*, not a tool used to build HireLens.

## 3. Work Delegated to AI

The following categories are supported by the actual repository history and
this session's own record. Scope is described narrowly to what is
evidenced — not assumed.

- **Code scaffolding and implementation.** New modules such as
  `src/services/alignment_scorer.py` and
  `src/services/recruiter_analysis_generator.py`, and schema additions in
  `src/models/schemas.py` (`fd6f623`), were implemented with AI assistance
  against explicit human direction (the commit message itself notes the
  product-direction change was "per explicit instruction"). Human review
  gate: all resulting code was committed by the project owner after the
  full test suite was verified to pass (157 tests, per the commit message).

- **Refactoring.** The Gemini → Groq provider migration (`fd6f623`) renamed
  `GeminiLLMService` to `GroqLLMService` and remapped exception types
  end-to-end while preserving the existing `generate_text(...)` interface —
  a refactor verified by the existing test suite continuing to pass.

- **Debugging.** The `CandidateExperience.role` schema bug (§6) was
  diagnosed by tracing a live pipeline failure to a specific over-strict
  Pydantic field, then cross-checked against how `evidence_matcher.py`
  already used that field defensively elsewhere in the codebase.

- **Test creation.** `tests/test_alignment_scorer.py` and
  `tests/test_recruiter_analysis_generator.py` were added alongside the new
  services in `fd6f623`; the suite grew to 157 tests. Acceptance criterion:
  the suite had to pass in full before the commit was made.

- **Evaluation harness development.** The entire `evaluation/` package
  (`run_evaluation.py`, `test_cases.json`, `rubric.md`, `baseline.md`,
  `failure_analysis.md`) was built in commit `9a43a58`, including the
  deterministic/mocked/live mode split and the `--yes` gate before any real
  Groq API call. This was accepted after all 12 cases were confirmed
  passing, including four live cases run against the real API.

- **Documentation drafting/editing.** `README.md` and
  `docs/CASE_STUDY.md` were drafted/edited with AI assistance (`544a78e`,
  `9a43a58`, and this session's edit to `docs/CASE_STUDY.md`). Every such
  edit was checked against the actual implementation before being accepted
  — for example, `544a78e`'s commit message states the alignment-scoring
  description was "verified against the actual implementation, not
  assumed."

- **Repository inspection.** This session, and prior ones per the commit
  history, used `git log`, file reads, and grep-style searches to confirm
  claims before writing them (e.g., confirming no remaining Gemini
  references before `544a78e`; confirming the true evaluation-mode counts
  against `evaluation/test_cases.json` before correcting §6 below).

- **Test execution assistance.** `pytest` was run as part of verifying
  changes (this session re-ran the full suite — see §5 — without modifying
  any code).

- **Implementation of requested changes.** Every change in this session
  (the Section 07 documentation fix, this file) was made strictly to the
  scope requested, with no unrelated edits, per direct instruction each
  time.

- **Git-related operational assistance.** Staging, committing (with an
  exact, user-specified commit message), and pushing were performed only
  after `git status`/`git diff` were shown to confirm scope, and only on
  explicit instruction — never autonomously.

Not delegated, or not evidenced as delegated: the initial project concept,
the choice of problem domain, the LLM provider decision itself (a human
instruction preceded the Groq migration per `fd6f623`'s message), and any
hiring-related judgment (the system is explicitly designed to never make
one — see §4/§8).

## 4. Human-Owned Decisions

The following decisions are supported by commit messages, code comments,
and documentation and were not delegated to AI judgment:

- **Selecting the recruiter-facing CV/JD alignment problem** as the
  product's focus (established at `bf1df30`, the initial commit).
- **Defining scope and non-goals** — e.g. `README.md`'s "Future
  Improvements" section explicitly lists what is *not* implemented, framed
  as deliberate scope, not an oversight.
- **Deciding the score represents documented alignment/evidence coverage,
  not job-success probability** — stated explicitly in `README.md`
  ("Alignment Score": *"This is an alignment indicator, not a
  prediction"*) and reinforced in `docs/CASE_STUDY.md`.
- **Defining the deterministic scoring approach** — `AlignmentScorer` is
  pure application code with no LLM call (`README.md`, `src/services/
  alignment_scorer.py`), an explicit design choice documented in
  `fd6f623`'s commit message.
- **Deciding how required vs. preferred requirements are weighted** —
  the 70/30 required/preferred split is a fixed, documented business rule
  (`README.md`'s "Alignment Score" section), not something inferred by an
  LLM.
- **Defining the human recruiter as the final decision-maker** — stated
  repeatedly and consistently across `README.md` ("Responsible AI /
  Decision Support"), the `RecruiterAnalysisGenerator` prompt constraints
  (forbidding hiring-decision language), and `docs/CASE_STUDY.md`.
- **Deciding how missing evidence should be represented** — the "not
  evidenced in the submitted CV" phrasing rule (never "candidate lacks
  this skill") is enforced in code and prompts per `README.md`'s
  "Evidence-Grounded Analysis" section.
- **Provider/integration decisions** — the Gemini → Groq migration was a
  human-directed change (`fd6f623`'s commit message: "Replace Google
  Gemini SDK with the official Groq SDK..."), driven by a real quota
  exhaustion event documented in `evaluation/failure_analysis.md` §1.
- **Evaluation criteria and pass/fail expectations** — defined explicitly
  in `evaluation/test_cases.json` and `evaluation/rubric.md`, and
  `docs/CASE_STUDY.md` §9 notes that "No pass/fail criterion for TC-10 was
  loosened to make it pass."
- **Acceptance/rejection of implementation changes** — every commit in
  this repository was made only after the full test suite passed (per each
  commit message), and this session's two documentation edits were shown
  to the user via diff before being committed, per explicit instruction.
- **Final submission decisions** — committing and pushing to `origin/main`
  was performed only on direct, explicit request each time (this session's
  transcript: the case-study fix and its push were separate, explicitly
  authorized steps).

## 5. Verification Process

AI-assisted and AI-generated work was verified using concrete, repository-
grounded checks rather than accepted on inspection alone:

- **Automated pytest results.** The full suite was re-run in this session
  (`pytest`, via the project's `venv`) with **no code changes made**,
  confirming **157 passed** — matching the count documented in `README.md`
  and `evaluation/failure_analysis.md`.
- **Evaluation results.** `evaluation/results.md` (last updated
  2026-09-05) records **12/12 evaluation cases passing**, split 4
  deterministic / 4 mocked / 4 live (TC-01–TC-12), each with a measured
  wall-clock execution time and machine-readable backing in
  `evaluation/results.json`.
- **Schema validation.** Every LLM response in the pipeline is validated
  against a Pydantic schema before use (`README.md`'s "Integrations"
  section); the TC-10 bug (§6) was itself surfaced by this validation
  layer rejecting a real, unexpected LLM output.
- **Deterministic scoring tests.** `tests/test_alignment_scorer.py`
  exercises `AlignmentScorer.score()` directly, with no LLM involved.
- **Mocked tests.** `evaluation/test_cases.json` TC-05/07/09/12 and the
  broader `tests/` suite use scripted fake LLM services — no network
  calls — to check evidence-matching, sparse-input, and provider-failure
  paths.
- **Live evaluation runs.** TC-01, TC-02, TC-03, and TC-10 were executed
  against the real Groq API (`evaluation/results.md` records real
  `groq_model=openai/gpt-oss-120b` responses and real latencies, e.g.
  95.5s for TC-03).
- **Regression testing.** After the TC-10 schema fix, `evaluation/
  results.md`'s "Regression history" table preserves the original failure
  (`ERROR`, 2.82s) alongside the passing re-run (`PASS`, 59.56s), rather
  than erasing it.
- **Reviewing diffs / inspecting Git history.** This session used `git
  diff`, `git status`, and `git log` before every commit to confirm scope
  (e.g. confirming only `docs/CASE_STUDY.md` was staged for `3609caa`).
- **Checking repository state.** `git status` was checked immediately
  before and after each commit/push in this session, and is shown in §6
  and in this session's record.
- **Manually reviewing outputs.** The exact text of the Section 07 fix
  was shown to the user as a diff before committing (per this session's
  own record), rather than committed unreviewed.
- **Checking for forbidden/unsafe language.** `evaluation/test_cases.json`
  TC-12 explicitly checks that no forbidden hiring-decision phrase appears
  anywhere in a full dossier output (`docs/CASE_STUDY.md` §9: "no forbidden
  phrase").

All numbers above (157 tests, 12/12 cases, the 4/4/4 mode split, the
95.5s/59.56s/2.82s timings) are quoted directly from `evaluation/
results.md`, `evaluation/results.json`, or this session's own `pytest` run
— none are estimated.

## 6. Important Corrections and Rejected/Changed AI Output

**TC-10 — `CandidateExperience.role` schema bug (implementation error,
category A).**

- **What happened:** A live evaluation case (TC-10) fed the pipeline a
  realistic CV written as dense prose with no explicit job-title line.
  The real Groq API correctly followed its own grounding instructions and
  returned `"role": null` for the affected experience entries. The
  pipeline raised `HireLensPipelineError` and the entire analysis failed.
- **Why the original implementation failed:** `CandidateExperience.role`
  in `src/models/schemas.py` was declared as a required, non-nullable
  `str` — the only required field on that model; every sibling field was
  already optional. Pydantic rejected the (correct, honest) `null` value.
- **Root cause:** Confirmed, not assumed — `src/services/
  evidence_matcher.py` already guarded `if experience.role and
  experience.role.strip()` elsewhere in the codebase, showing the rest of
  the system already anticipated a falsy/`None` role. The required-field
  constraint was an unintentional oversight, not a deliberate invariant.
- **What was changed:** (1) `CandidateExperience.role` changed to
  `str | None = None` in `src/models/schemas.py`; (2) the
  `CandidateAnalyzer` prompt's documented JSON shape for `"role"` updated
  from `string` to `string or null` in `src/services/candidate_analyzer.py`
  — no other change.
- **How it was verified afterward:** TC-10 re-run went from **ERROR**
  (2.82s) to **PASS** (59.56s, `overall_score=72`); the directly relevant
  test files (`test_models.py`, `test_candidate_analyzer.py`,
  `test_evidence_matcher.py`, `test_hirelens_pipeline.py`) — 65 tests —
  passed; the full suite remained at 157/157 passing; no TC-10 pass/fail
  criterion in `test_cases.json` was loosened (`docs/CASE_STUDY.md` §9).
- **Why this demonstrates review rather than blind acceptance:** the bug
  was found precisely *because* a live case was run against the real
  model instead of only mocked unit tests — 157 passing mocked tests had
  not caught it, since none happened to construct a titleless CV. The fix
  was scoped to exactly the two lines that caused the failure and
  re-verified against both the specific case and the full suite before
  being accepted.

**`docs/CASE_STUDY.md` Section 07 evaluation-mode-count error
(documentation error, category A).**

- **What happened:** Section 07 of the case study stated the mode split
  as "mocked (5 cases...)" and "live (3 cases...)," while the IDs listed
  underneath each (4 mocked IDs, 4 live IDs) did not match those counts.
- **Verification performed:** The true per-case mode was read directly
  from `evaluation/test_cases.json` in this session, confirming the
  correct split is 4 deterministic / 4 mocked / 4 live (12 total), and
  that the discrepancy was a drafting error, not a change in the
  underlying evaluation package.
- **What was changed:** Only the two miscounted lines in
  `docs/CASE_STUDY.md` §7 were corrected ("5 cases" → "4 cases," "3 cases"
  → "4 cases"); the unsupported "plus scripted fakes" padding phrase tied
  to the wrong count was removed. No other section, claim, score, or
  timestamp was touched.
- **Outcome:** Shown to the user as an exact diff before being committed
  (`3609caa`, "fix: correct evaluation mode counts in case study"), then
  pushed to `origin/main` only after a separate, explicit push request.

**Distinguishing the three categories, using only what's evidenced here:**

- **(A) Errors in AI-assisted implementation:** the `role`-field schema
  bug, and the §7 evaluation-mode-count documentation error above.
- **(B) Unexpected-but-correct behavior from the application LLM:** Groq
  itself did not misbehave in the TC-10 case — it correctly returned
  `null` per its own grounding rules; the *schema* was wrong to reject
  that correct output. No case of genuinely incorrect LLM output being
  silently accepted is evidenced in this repository (malformed-response
  handling exists — `evaluation/failure_analysis.md` §3 — but the
  documented outcome there is the deterministic fallback engaging, not a
  bad output being trusted).
- **(C) Human decisions to change product direction:** the Gemini → Groq
  migration and the "normalization only" → recruiter-facing
  decision-support pivot (both `fd6f623`, described in its commit message
  as directed changes) — neither was an error being corrected; both were
  deliberate scope decisions made by the project owner.

## 7. AI Collaboration Principles

- AI tools proposed or implemented specific, requested changes; they did
  not independently decide what to build next.
- Every proposed change was inspected by a human before being accepted,
  modified, or rejected — evidenced by commit messages describing
  verification steps taken "before committing" (`544a78e`, `9a43a58`) and
  by this session's diff-before-commit pattern.
- Deterministic logic was deliberately preferred over LLM judgment for
  outcomes that must be reproducible and inspectable — the alignment score
  is pure application code (`AlignmentScorer`), never an LLM output
  (`README.md`).
- Structured schemas (Pydantic models in `src/models/schemas.py`)
  constrain every LLM response before it can flow further into the
  pipeline, so malformed or unexpected model output fails loudly rather
  than propagating silently.
- Tests and evaluations were the actual acceptance gate — every commit
  message in this repository states the test suite status at the time of
  that commit, and the evaluation package's `--yes` gate exists
  specifically to keep real API calls (and their cost/quota) under
  explicit human control.
- Important product and safety decisions — what the score means, who
  makes the hiring call, how missing evidence is phrased — remained
  human-owned and are enforced in code and prompts, not left to an LLM's
  discretion at runtime.
- AI output, whether from a development-assistance tool or from the
  application's own Groq integration, was treated as something to verify
  against the actual codebase, tests, or evaluation results — never
  accepted purely on the basis of looking plausible.

## 8. What AI Did NOT Own

AI tools used during development, and the Groq LLM integration used at
runtime, did not independently own:

- The final product scope (defined by the project owner; see §4).
- Final hiring decisions (the system is explicitly designed never to make
  one — `README.md`'s "Responsible AI / Decision Support" section).
- Candidate hiring recommendations (the `RecruiterAnalysisGenerator`
  prompt explicitly forbids stating or implying a hiring decision).
- Evaluation acceptance criteria (fixed in `evaluation/test_cases.json`
  and `evaluation/rubric.md`, and explicitly not loosened to pass TC-10 —
  §6).
- Final interpretation of evaluation results (results are reported as
  measured facts in `evaluation/results.md`; what they mean for the
  project's status was a human judgment).
- Final submission decisions (every commit and push in this repository's
  history, including in this session, was made only on direct human
  instruction).

The system itself is built so the recruiter — not the AI, and not
HireLens — remains responsible for the final hiring decision on any
candidate it analyzes.

## 9. Key Lesson

HireLens' own history is the clearest evidence for this: the 157-test
mocked suite passed the whole time the `CandidateExperience.role` bug
existed, because every mocked fixture happened to include a job title.
What actually found the bug was a live evaluation run against the real
Groq API, paired with a human-legible root-cause trace (`evidence_matcher.py`
already handling a falsy `role` elsewhere) that confirmed the fix was
correct before it was accepted. AI assistance made it fast to scaffold the
evaluation harness, the new services, and the documentation that describes
them — but it was the deterministic scoring design, the explicit evaluation
criteria, the live-vs-mocked test split, and a human checking the diff
before every commit that turned "code that runs" into a result that could
be trusted enough to ship and document truthfully. AI can accelerate
implementation, but reliable delivery still required human-owned
architecture, evaluation, verification, and correction at every step.

## 10. Evidence

- Git history: `git log --oneline --decorate --all` (`bf1df30`, `fd6f623`,
  `544a78e`, `9a43a58`, `3609caa`) and each commit's full message.
- Source code: `src/models/schemas.py`, `src/services/alignment_scorer.py`,
  `src/services/candidate_analyzer.py`, `src/services/evidence_matcher.py`,
  `src/services/recruiter_analysis_generator.py`,
  `src/services/hirelens_pipeline.py`.
- Tests: `tests/` (157 passing, verified in this session), in particular
  `tests/test_models.py`, `tests/test_candidate_analyzer.py`,
  `tests/test_evidence_matcher.py`, `tests/test_alignment_scorer.py`.
- Evaluation package: `evaluation/test_cases.json`, `evaluation/rubric.md`,
  `evaluation/results.md`, `evaluation/results.json`,
  `evaluation/run_evaluation.py`.
- Failure analysis: `evaluation/failure_analysis.md` (five documented
  scenarios, including the Gemini→Groq migration and the TC-10 schema
  bug).
- Case study: `docs/CASE_STUDY.md` (§7 Evaluation Strategy, §9 A Real
  Failure We Found and Fixed, §10 Results).
- README: `README.md` ("Alignment Score," "Integrations," "Responsible AI
  / Decision Support," "Evaluation" sections).
