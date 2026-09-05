# HireLens AI — Baseline Comparison

This document describes the workflow HireLens replaces, and how it compares
to HireLens on the metrics that matter for a recruiter's day-to-day review.
It intentionally separates what has actually been **measured** from what is
an **assumption** or a **proposed metric not yet collected** — per the
project's evaluation ground rules, no number below is invented.

## The baseline workflow: manual recruiter review

Without HireLens, a recruiter evaluating one candidate against one job
description typically:

1. Reads the candidate's CV.
2. Reads the job description.
3. Identifies the job's stated requirements (mentally, or in a scratch
   document/spreadsheet).
4. Searches the CV for evidence relevant to each requirement.
5. Notes strengths and gaps informally (memory, a notes doc, an ATS
   free-text field).
6. Identifies which items are unclear and need to be verified in an
   interview.
7. Prepares interview questions, usually from general experience or a
   generic template, not systematically derived from the specific gaps in
   this candidate's specific CV.

This is repeated, independently, for every candidate the recruiter screens —
with no guarantee that the same requirement is checked the same way, or
even checked at all, across candidates.

### A secondary baseline concept: generic LLM prompting

A recruiter might instead paste a CV and a job description into a
general-purpose chat assistant (e.g. "does this candidate match this job
description?"). This is a real, increasingly common baseline, but it is
**not measured in this evaluation package** — doing so honestly would
require running a comparable prompt against a comparable model under
comparable conditions, which has not been done. It is documented here only
as a concept worth comparing against in a future iteration of this
evaluation, not as a measured data point. Structurally, this baseline is
expected to differ from HireLens in that it (a) has no deterministic,
reproducible score, (b) has no requirement-by-requirement traceability back
to source text, and (c) is not guaranteed to distinguish "no evidence
found" from "candidate lacks this" — but none of that is asserted as a
measured fact here.

## Comparison

| Metric | Manual baseline | HireLens AI | Status |
|---|---|---|---|
| Time per candidate review | Not yet measured | Not yet measured | **Not yet measured** — no timed manual-review sessions or timed HireLens sessions with a human recruiter have been conducted. `run_evaluation.py` records wall-clock *pipeline execution time* for each case in `results.md` (`execution_time_seconds`), which is a useful proxy for machine processing time, but is not the same thing as a recruiter's end-to-end review time and should not be read as one. |
| Number of manual review steps | 7 (see workflow above) | 1 (upload CV + paste JD + click "Analyze Candidate") | **Assumption** — the manual step count is the workflow description above, not a timed/observed study; the HireLens step count is directly observable from `app.py`'s UI flow. |
| Requirement coverage visibility | Implicit / recruiter's own notes, inconsistent across candidates | Explicit, per-requirement `EvidenceMatch` for every requirement, every time (see `AlignmentScore.coverage` and the Requirement Breakdown / Evidence Coverage Summary sections in `app.py`) | **Observed from the codebase** — HireLens is structurally guaranteed to produce one status per requirement (`src/services/evidence_matcher.py`); the manual baseline has no such guarantee. This is a structural claim, not a user-study result. |
| Structured output completeness | None (free-text notes, if any) | Structured `CandidateDossier`: profile, requirements, evidence matches, alignment score, recruiter analysis, interview questions — see Rubric criterion 5 | **Observed from the codebase / evaluation** — see `evaluation/rubric.md` §5 and the TC-01/02/03/07/10/12 structural checks in `results.md`. |
| Human intervention required | 100% of the analysis (fully manual) | Recruiter still makes every hiring decision; HireLens produces the evidence breakdown and score, but every "Needs Verification" and every gap is explicitly flagged for the recruiter to resolve directly with the candidate | **By design, not fully quantifiable as a single number** — HireLens is decision *support*: it never removes the recruiter from the loop, it changes what the recruiter starts from. The fraction of requirements flagged `needs_verification` per candidate is directly observable per run (`AlignmentScore.coverage`) and is a proposed metric for a future controlled comparison. |
| Consistency across repeated cases | Not measured; known to vary by recruiter, mood, fatigue, and how carefully the CV is read | Deterministic for the alignment score (same input → same score, always — `AlignmentScorer` is pure code, no LLM); the LLM-generated narrative (`RecruiterAnalysis`, interview questions) can vary in *wording* between runs on identical input, but not in the underlying evidence data it is grounded in | **Partially observed from the codebase** — the alignment score's determinism is a property of the code (`src/services/alignment_scorer.py`, no LLM call); the narrative's variability under repeated identical-input runs has not been separately measured (would require running the same case through `--mode live` multiple times and diffing outputs, which has not been done as part of this evaluation). |

## What would be required to responsibly measure "time per candidate review"

Listed honestly as future work, not as something this package claims to
have done:

- A small set of recruiters (or recruiter-proxies) reviewing the same
  candidate/JD pairs manually, timed, without HireLens.
- The same recruiters reviewing the same pairs using HireLens's output as a
  starting point, timed.
- Enough pairs and enough reviewers to say something beyond anecdote.

None of this has been conducted for this submission. The `results.md` file
distinguishes machine-measured pipeline execution time (real, observed) from
this kind of human-in-the-loop timing study (not conducted).
