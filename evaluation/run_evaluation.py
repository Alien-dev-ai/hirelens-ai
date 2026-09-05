#!/usr/bin/env python3
"""HireLens AI evaluation runner.

Executes the evaluation cases defined in evaluation/test_cases.json against
the actual HireLens pipeline components (never a reimplementation of them),
records real pass/fail outcomes and measured execution time, and writes:

  - evaluation/results.json  (machine-readable, cumulative across runs)
  - evaluation/results.md    (human-readable, regenerated from results.json)

Usage:
    python evaluation/run_evaluation.py --list
    python evaluation/run_evaluation.py --mode deterministic
    python evaluation/run_evaluation.py --mode mocked
    python evaluation/run_evaluation.py --mode live            # dry-run: shows the plan, does not call the API
    python evaluation/run_evaluation.py --mode live --yes      # actually makes real Groq API calls
    python evaluation/run_evaluation.py --case TC-06           # run a single case by ID

Modes (see test_cases.json "_meta.modes" for the full definitions):
  deterministic  Pure application code, no LLM constructed at all. Free.
  mocked         Real pipeline/service classes, scripted fake LLM. Free.
  live           Real GroqLLMService, real API calls. Costs quota -- gated
                 behind --yes, and always preceded by a printed call-count
                 estimate so nothing runs by surprise.

Nothing in this script fabricates a result: a case's `actual_result` and
`status` are only ever written after that case has actually executed. Any
case not selected for a given run is left (or reported) as "not yet
executed" in results.md.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.extraction.document_extractor import extract_document  # noqa: E402
from src.models.schemas import (  # noqa: E402
    AlignmentScore,
    CandidateExperience,
    CandidateProfile,
    EvidenceMatch,
    EvidenceStatus,
    JobRequirement,
    JobRequirements,
    RequirementCoverage,
    RequirementImportance,
)
from src.services.alignment_scorer import AlignmentScorer  # noqa: E402
from src.services.candidate_analyzer import CandidateAnalyzer, CandidateAnalyzerError  # noqa: E402
from src.services.evidence_matcher import EvidenceMatcher  # noqa: E402
from src.services.hirelens_pipeline import HireLensPipeline, HireLensPipelineError  # noqa: E402
from src.services.jd_analyzer import JobDescriptionAnalyzer, JobDescriptionAnalyzerError  # noqa: E402
from src.services.llm_service import LLMServiceError  # noqa: E402
from src.services.recruiter_analysis_generator import RecruiterAnalysisGenerator  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
import harness  # noqa: E402

EVAL_DIR = Path(__file__).resolve().parent
TEST_CASES_PATH = EVAL_DIR / "test_cases.json"
RESULTS_JSON_PATH = EVAL_DIR / "results.json"
RESULTS_MD_PATH = EVAL_DIR / "results.md"

KNOWN_ALIGNMENT_LABELS = {
    "Strong Alignment",
    "Moderate Alignment",
    "Limited Alignment",
    "Minimal Alignment",
}


# ===========================================================================
# Individual test-case implementations
#
# Each function takes no arguments, performs its own assertions with plain
# `assert`, and returns a dict describing what was actually observed. An
# AssertionError means the case FAILED; any other exception is recorded as
# an execution ERROR (e.g. a live API call failing).
# ===========================================================================


def run_tc01_strong_alignment() -> dict:
    return _run_live_pipeline_case(
        case_id="TC-01",
        cv_filename="candidate_strong.docx",
        jd_text=harness.JD_BACKEND_ENGINEER_BULLETED,
    )


def run_tc02_moderate_alignment() -> dict:
    return _run_live_pipeline_case(
        case_id="TC-02",
        cv_filename="candidate_moderate.docx",
        jd_text=harness.JD_BACKEND_ENGINEER_BULLETED,
    )


def run_tc03_low_alignment() -> dict:
    return _run_live_pipeline_case(
        case_id="TC-03",
        cv_filename="candidate_low.docx",
        jd_text=harness.JD_BACKEND_ENGINEER_BULLETED,
    )


def run_tc10_input_variation() -> dict:
    return _run_live_pipeline_case(
        case_id="TC-10",
        cv_filename="candidate_variation.docx",
        jd_text=harness.JD_BACKEND_ENGINEER_PROSE,
    )


LIVE_ARTIFACTS_DIR = EVAL_DIR / "live_run_artifacts"


def _run_live_pipeline_case(*, case_id: str, cv_filename: str, jd_text: str) -> dict:
    """Shared execution path for the four live test cases (TC-01/02/03/10).

    Makes real Groq API calls via the real GroqLLMService/HireLensPipeline --
    no mocking. Only reachable through --mode live --yes.
    """
    from src.services.llm_service import GroqLLMService

    llm_service = GroqLLMService()
    pipeline = HireLensPipeline.create(llm_service)

    dossier = pipeline.analyze(
        cv_file_path=harness.cv_path(cv_filename),
        job_description=jd_text,
    )

    assert len(dossier.evidence_matches) == len(dossier.job_requirements.requirements), (
        "Expected exactly one evidence match per job requirement."
    )
    assert 0 <= dossier.alignment_score.overall_score <= 100
    assert dossier.alignment_score.alignment_label in KNOWN_ALIGNMENT_LABELS

    all_texts = [
        dossier.recruiter_analysis.summary,
        *dossier.recruiter_analysis.strengths,
        *dossier.recruiter_analysis.gaps,
        *dossier.recruiter_analysis.validation_areas,
        *[m.explanation for m in dossier.evidence_matches],
    ]
    forbidden = harness.find_forbidden_phrases(*all_texts)
    assert not forbidden, f"Forbidden phrase(s) found in generated text: {forbidden}"

    status_by_requirement = [
        {
            "requirement": m.requirement,
            "status": m.status.value,
            "evidence": m.evidence,
            "confidence": m.confidence,
        }
        for m in dossier.evidence_matches
    ]

    # Persist the full dossier for human inspection. This is audit/debug
    # scaffolding only -- it is never read back by the runner itself, and
    # never used to decide pass/fail.
    LIVE_ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    artifact_path = LIVE_ARTIFACTS_DIR / f"{case_id}_dossier.json"
    artifact_path.write_text(dossier.model_dump_json(indent=2))

    return {
        "overall_score": dossier.alignment_score.overall_score,
        "alignment_label": dossier.alignment_score.alignment_label,
        "requirement_statuses": status_by_requirement,
        "recruiter_analysis_summary": dossier.recruiter_analysis.summary,
        "strengths": dossier.recruiter_analysis.strengths,
        "gaps": dossier.recruiter_analysis.gaps,
        "validation_areas": dossier.recruiter_analysis.validation_areas,
        "warnings": dossier.warnings,
        "groq_model": llm_service.model,
        "full_dossier_artifact": str(artifact_path.relative_to(REPO_ROOT)),
    }


def run_tc04_missing_required_skill() -> dict:
    profile = CandidateProfile(
        full_name="Test Candidate",
        skills=["Python", "Docker", "Git"],
        experiences=[
            CandidateExperience(
                role="Backend Developer",
                organization="Example Co",
                description="Built services with Python, Docker, and Git.",
            )
        ],
    )
    requirement = JobRequirement(
        id="req-1",
        requirement="Experience with Kubernetes",
        category="skill",
        importance=RequirementImportance.required,
    )
    job_requirements = JobRequirements(role_title="Backend Engineer", requirements=[requirement])

    matcher = EvidenceMatcher(llm_service=None)  # no LLM constructed at all
    matches = matcher.match(profile, job_requirements)

    assert len(matches) == 1
    match = matches[0]
    assert match.status == EvidenceStatus.no_evidence_found
    assert match.evidence == []
    explanation_lower = match.explanation.lower()
    assert "does not mean" in explanation_lower or "does n0t mean" in explanation_lower, (
        f"Explanation did not include the 'does not mean [lacks the skill]' framing: {match.explanation!r}"
    )
    forbidden = harness.find_forbidden_phrases(match.explanation)
    assert not forbidden, f"Forbidden phrase(s) found in explanation: {forbidden}"

    score = AlignmentScorer().score(job_requirements, matches)
    assert score.overall_score == 0
    assert score.required_score == 0.0

    return {
        "status": match.status.value,
        "explanation": match.explanation,
        "overall_score": score.overall_score,
    }


def run_tc05_ambiguous_partial_evidence() -> dict:
    profile = CandidateProfile(
        full_name="Test Candidate",
        experiences=[
            CandidateExperience(
                role="Cloud Engineer",
                organization="Example Co",
                description="Worked with cloud-based deployment tools and automation scripts.",
            )
        ],
    )
    requirement = JobRequirement(
        id="req-1",
        requirement="Experience with cloud infrastructure",
        category="skill",
        importance=RequirementImportance.required,
    )
    job_requirements = JobRequirements(role_title="Platform Engineer", requirements=[requirement])

    fake_llm = harness.ScriptedLLMService(
        evidence_match={
            "status": "needs_verification",
            "evidence": ["cloud-based deployment tools"],
            "explanation": (
                "The candidate's documented experience with cloud-based deployment "
                "tools is related to, but does not explicitly confirm, the stated "
                "cloud infrastructure requirement. This is partial/ambiguous "
                "evidence and should be verified directly with the candidate."
            ),
        }
    )
    matcher = EvidenceMatcher(llm_service=fake_llm)
    matches = matcher.match(profile, job_requirements)

    assert len(matches) == 1
    match = matches[0]
    assert match.status == EvidenceStatus.needs_verification, (
        f"Expected needs_verification, got {match.status.value}"
    )
    assert fake_llm.calls, "Deterministic matching should have been inconclusive, falling back to the fake LLM."

    score = AlignmentScorer().score(job_requirements, matches)
    assert score.overall_score == 50, f"Expected 0.5 credit -> 50 overall_score, got {score.overall_score}"

    return {
        "status": match.status.value,
        "evidence": match.evidence,
        "overall_score": score.overall_score,
    }


def run_tc06_preferred_missing() -> dict:
    required_ids = ["req-1", "req-2", "req-3", "req-4"]
    preferred_ids = ["req-5", "req-6", "req-7"]

    requirements = [
        JobRequirement(id=rid, requirement=f"Required item {rid}", category="skill", importance=RequirementImportance.required)
        for rid in required_ids
    ] + [
        JobRequirement(id=rid, requirement=f"Preferred item {rid}", category="skill", importance=RequirementImportance.preferred)
        for rid in preferred_ids
    ]
    job_requirements = JobRequirements(role_title="Backend Engineer", requirements=requirements)

    matches = [
        EvidenceMatch(
            requirement_id=rid,
            requirement=f"Required item {rid}",
            status=EvidenceStatus.evidence_found,
            evidence=["matched"],
            explanation="Explicitly supported by the candidate profile.",
            confidence=0.9,
        )
        for rid in required_ids
    ] + [
        EvidenceMatch(
            requirement_id=rid,
            requirement=f"Preferred item {rid}",
            status=EvidenceStatus.no_evidence_found,
            evidence=[],
            explanation="No explicit evidence found in the submitted candidate documents.",
            confidence=0.6,
        )
        for rid in preferred_ids
    ]

    score = AlignmentScorer().score(job_requirements, matches)
    assert score.overall_score == 70, f"Expected 70 (0.7*100 + 0.3*0), got {score.overall_score}"
    assert score.required_weight == 0.7
    assert score.preferred_weight == 0.3
    assert score.alignment_label not in {"Minimal Alignment", "Limited Alignment"}, (
        f"Missing preferred qualifications alone should not crater the score into "
        f"'{score.alignment_label}'."
    )

    generator = RecruiterAnalysisGenerator(llm_service=None)  # forces the deterministic template fallback
    candidate_profile = CandidateProfile(full_name="Test Candidate")
    analysis = generator.generate(candidate_profile, job_requirements, matches, score)

    assert analysis.gaps, "Expected the missing preferred requirements to be listed as gaps."
    gaps_text = " ".join(analysis.gaps).lower()
    assert "not evidenced" in gaps_text, f"Gaps should be phrased as 'not evidenced': {analysis.gaps!r}"
    forbidden = harness.find_forbidden_phrases(*analysis.gaps)
    assert not forbidden, f"Forbidden phrase(s) found in gaps: {forbidden}"

    return {
        "overall_score": score.overall_score,
        "alignment_label": score.alignment_label,
        "required_weight": score.required_weight,
        "preferred_weight": score.preferred_weight,
        "gaps": analysis.gaps,
    }


def run_tc07_sparse_minimal_cv() -> dict:
    fake_llm = harness.build_sparse_scripted_llm()
    pipeline = HireLensPipeline.create(fake_llm)

    dossier = pipeline.analyze(
        cv_file_path=harness.cv_path("candidate_sparse.docx"),
        job_description=harness.JD_BACKEND_ENGINEER_BULLETED,
    )

    assert dossier.alignment_score.overall_score < 40, (
        f"Expected a low score for a near-empty CV, got {dossier.alignment_score.overall_score}"
    )
    assert dossier.recruiter_analysis.validation_areas, "Expected non-empty validation areas."

    all_texts = [
        dossier.recruiter_analysis.summary,
        *dossier.recruiter_analysis.strengths,
        *dossier.recruiter_analysis.gaps,
        *dossier.recruiter_analysis.validation_areas,
        *[m.explanation for m in dossier.evidence_matches],
    ]
    forbidden = harness.find_forbidden_phrases(*all_texts)
    assert not forbidden, f"Forbidden phrase(s) found in generated text: {forbidden}"

    return {
        "overall_score": dossier.alignment_score.overall_score,
        "alignment_label": dossier.alignment_score.alignment_label,
        "requirement_count": len(dossier.job_requirements.requirements),
        "llm_calls_made": len(fake_llm.calls),
    }


def run_tc08_invalid_document_input() -> dict:
    results = {}

    # -- (a) unsupported file extension --------------------------------
    unsupported = extract_document(harness.cv_path("candidate_unsupported.txt"))
    assert unsupported.extraction_success is False
    assert unsupported.error_message
    results["unsupported_extension"] = unsupported.error_message

    # -- (b) missing file -----------------------------------------------
    missing = extract_document(harness.cv_path("candidate_does_not_exist.docx"))
    assert missing.extraction_success is False
    assert missing.error_message
    results["missing_file"] = missing.error_message

    # -- (c) corrupted docx -----------------------------------------------
    corrupted = extract_document(harness.cv_path("candidate_corrupted.docx"))
    assert corrupted.extraction_success is False
    assert corrupted.error_message
    results["corrupted_document"] = corrupted.error_message

    # -- pipeline-level: extraction failure must short-circuit everything --
    for label, path in (
        ("unsupported_extension", harness.cv_path("candidate_unsupported.txt")),
        ("missing_file", harness.cv_path("candidate_does_not_exist.docx")),
        ("corrupted_document", harness.cv_path("candidate_corrupted.docx")),
    ):
        pipeline = HireLensPipeline(
            candidate_analyzer=harness.PoisonPillCandidateAnalyzer(),
            jd_analyzer=harness.PoisonPillJDAnalyzer(),
            evidence_matcher=harness.PoisonPillEvidenceMatcher(),
            alignment_scorer=harness.PoisonPillAlignmentScorer(),
            recruiter_analysis_generator=harness.PoisonPillRecruiterAnalysisGenerator(),
            interview_generator=harness.PoisonPillInterviewGenerator(),
            extract_document_fn=extract_document,
        )
        try:
            pipeline.analyze(path, "Backend Engineer job description")
            raise AssertionError(f"Expected HireLensPipelineError for {label}, but no exception was raised.")
        except HireLensPipelineError as exc:
            assert "extraction" in str(exc).lower(), (
                f"Expected an extraction-stage error for {label}, got: {exc}"
            )

    return results


def run_tc09_llm_service_failure() -> dict:
    results = {}

    # -- (a) quota / rate-limit failure -----------------------------------
    pipeline = HireLensPipeline.create(harness.build_quota_failure_scripted_llm())
    try:
        pipeline.analyze(harness.cv_path("candidate_strong.docx"), harness.JD_BACKEND_ENGINEER_BULLETED)
        raise AssertionError("Expected HireLensPipelineError for a quota-exceeded LLM failure.")
    except HireLensPipelineError as exc:
        assert exc.is_quota_exceeded is True
        assert "candidate analysis" in str(exc).lower()
        assert "Traceback" not in str(exc)
        results["quota_exceeded"] = {"is_quota_exceeded": exc.is_quota_exceeded, "message": str(exc)}

    # -- (b) connection / outage failure -----------------------------------
    pipeline = HireLensPipeline.create(harness.build_connection_failure_scripted_llm())
    try:
        pipeline.analyze(harness.cv_path("candidate_strong.docx"), harness.JD_BACKEND_ENGINEER_BULLETED)
        raise AssertionError("Expected HireLensPipelineError for a connection-failure LLM error.")
    except HireLensPipelineError as exc:
        assert exc.is_quota_exceeded is False
        assert "candidate analysis" in str(exc).lower()
        results["connection_failure"] = {"is_quota_exceeded": exc.is_quota_exceeded, "message": str(exc)}

    return results


def run_tc11_empty_input_validation() -> dict:
    poison_pill = harness.PoisonPillLLMService()

    for empty_text in ("", "   "):
        try:
            CandidateAnalyzer(llm_service=poison_pill).analyze(empty_text)
            raise AssertionError(f"Expected CandidateAnalyzerError for cv_text={empty_text!r}")
        except CandidateAnalyzerError:
            pass

    for empty_text in ("", "   "):
        try:
            JobDescriptionAnalyzer(llm_service=poison_pill).analyze(empty_text)
            raise AssertionError(f"Expected JobDescriptionAnalyzerError for jd_text={empty_text!r}")
        except JobDescriptionAnalyzerError:
            pass

    return {
        "candidate_analyzer_validated": ["", "   "],
        "jd_analyzer_validated": ["", "   "],
        "llm_ever_called": False,
    }


def run_tc12_structural_completeness() -> dict:
    fake_llm = harness.build_moderate_scripted_llm()
    pipeline = HireLensPipeline.create(fake_llm)

    dossier = pipeline.analyze(
        cv_file_path=harness.cv_path("candidate_moderate.docx"),
        job_description=harness.JD_BACKEND_ENGINEER_BULLETED,
    )

    assert len(dossier.evidence_matches) == len(dossier.job_requirements.requirements)
    assert len(dossier.interview_questions) == len(dossier.job_requirements.requirements)
    assert dossier.alignment_score.alignment_label in KNOWN_ALIGNMENT_LABELS
    assert dossier.recruiter_analysis.summary

    all_texts = [
        dossier.recruiter_analysis.summary,
        *dossier.recruiter_analysis.strengths,
        *dossier.recruiter_analysis.gaps,
        *dossier.recruiter_analysis.validation_areas,
        *[m.explanation for m in dossier.evidence_matches],
        *[q.question for q in dossier.interview_questions],
        *[q.reason for q in dossier.interview_questions],
    ]
    forbidden = harness.find_forbidden_phrases(*all_texts)
    assert not forbidden, f"Forbidden phrase(s) found in generated text: {forbidden}"

    return {
        "overall_score": dossier.alignment_score.overall_score,
        "alignment_label": dossier.alignment_score.alignment_label,
        "evidence_match_count": len(dossier.evidence_matches),
        "interview_question_count": len(dossier.interview_questions),
    }


# Maps test-case ID -> (mode, callable). Must stay in sync with test_cases.json.
CASES = {
    "TC-01": ("live", run_tc01_strong_alignment),
    "TC-02": ("live", run_tc02_moderate_alignment),
    "TC-03": ("live", run_tc03_low_alignment),
    "TC-04": ("deterministic", run_tc04_missing_required_skill),
    "TC-05": ("mocked", run_tc05_ambiguous_partial_evidence),
    "TC-06": ("deterministic", run_tc06_preferred_missing),
    "TC-07": ("mocked", run_tc07_sparse_minimal_cv),
    "TC-08": ("deterministic", run_tc08_invalid_document_input),
    "TC-09": ("mocked", run_tc09_llm_service_failure),
    "TC-10": ("live", run_tc10_input_variation),
    "TC-11": ("deterministic", run_tc11_empty_input_validation),
    "TC-12": ("mocked", run_tc12_structural_completeness),
}


# ===========================================================================
# Runner plumbing
# ===========================================================================


def load_test_case_specs() -> dict:
    with open(TEST_CASES_PATH) as f:
        data = json.load(f)
    return {case["id"]: case for case in data["test_cases"]}


def load_existing_results() -> dict:
    if RESULTS_JSON_PATH.exists():
        with open(RESULTS_JSON_PATH) as f:
            return json.load(f)
    return {"runs": {}}


def execute_case(case_id: str, spec: dict) -> dict:
    mode, fn = CASES[case_id]
    started_at = datetime.now(timezone.utc).isoformat()
    start = time.perf_counter()
    record = {
        "id": case_id,
        "name": spec["name"],
        "mode": mode,
        "started_at": started_at,
    }
    try:
        actual_result = fn()
        record["status"] = "PASS"
        record["actual_result"] = actual_result
        record["error"] = None
    except AssertionError as exc:
        record["status"] = "FAIL"
        record["actual_result"] = None
        record["error"] = str(exc)
    except Exception as exc:  # noqa: BLE001 - deliberately broad: any failure is a recorded result
        record["status"] = "ERROR"
        record["actual_result"] = None
        record["error"] = f"{type(exc).__name__}: {exc}"
        record["traceback"] = traceback.format_exc()
    record["execution_time_seconds"] = round(time.perf_counter() - start, 4)
    return record


def estimate_live_calls(spec: dict) -> str:
    return spec.get("estimated_live_llm_calls", "unknown")


def print_plan(case_ids: list[str], specs: dict) -> None:
    print("\nPlanned execution:")
    for case_id in case_ids:
        mode, _ = CASES[case_id]
        spec = specs[case_id]
        suffix = f"  [{estimate_live_calls(spec)}]" if mode == "live" else ""
        print(f"  {case_id}  ({mode})  {spec['name']}{suffix}")
    print()


def run_cases(case_ids: list[str], specs: dict, existing_results: dict) -> dict:
    """Execute cases and merge results into the cumulative results dict.

    A case's *previous* record (if any) is preserved under `history` rather
    than being silently overwritten -- so a documented failure -> fix ->
    regression story (e.g. TC-10's schema bug) stays visible in the
    machine-readable results, not just in prose docs.
    """
    runs = existing_results.setdefault("runs", {})
    for case_id in case_ids:
        spec = specs[case_id]
        print(f"Running {case_id}: {spec['name']} ...", end=" ", flush=True)
        record = execute_case(case_id, spec)

        previous = runs.get(case_id)
        if previous is not None:
            history = previous.pop("history", [])
            if previous["status"] != "PASS":
                # Only retain history for a genuine failure -> pass story.
                # A routine PASS -> PASS re-run has nothing to preserve.
                history.append(previous)
            if history:
                record["history"] = history

        runs[case_id] = record
        print(f"{record['status']} ({record['execution_time_seconds']}s)")
        if record["status"] != "PASS":
            print(f"    -> {record['error']}")
    existing_results["last_updated"] = datetime.now(timezone.utc).isoformat()
    return existing_results


def write_results_json(results: dict) -> None:
    with open(RESULTS_JSON_PATH, "w") as f:
        json.dump(results, f, indent=2, sort_keys=False)


def write_results_md(results: dict, specs: dict) -> None:
    runs = results.get("runs", {})
    lines = [
        "# HireLens AI — Evaluation Results",
        "",
        f"Last updated: {results.get('last_updated', 'never')}",
        "",
        "Generated by `evaluation/run_evaluation.py`. Every row below reflects an "
        "actual, measured execution — nothing here is estimated or fabricated. "
        "A test case not yet run is listed separately under 'Prepared but not "
        "yet executed'.",
        "",
        "## Executed results",
        "",
        "| Test ID | Scenario | Mode | Expected Behavior (summary) | Actual Result | Status | Time (s) |",
        "|---------|----------|------|------------------------------|----------------|--------|----------|",
    ]

    executed_ids = []
    for case_id in sorted(runs.keys()):
        record = runs[case_id]
        spec = specs.get(case_id, {})
        executed_ids.append(case_id)
        expected_summary = (spec.get("expected_behavior", "") or "")[:80]
        if len(spec.get("expected_behavior", "")) > 80:
            expected_summary += "..."
        if record["status"] == "PASS":
            actual = _summarize_actual_result(record.get("actual_result"))
        else:
            actual = record.get("error", "")[:120]
        lines.append(
            f"| {case_id} | {spec.get('name', record['name'])} | {record['mode']} | "
            f"{expected_summary} | {actual} | {record['status']} | {record['execution_time_seconds']} |"
        )

    history_lines = []
    for case_id in sorted(runs.keys()):
        record = runs[case_id]
        for previous in record.get("history", []):
            prev_actual = (
                _summarize_actual_result(previous.get("actual_result"))
                if previous["status"] == "PASS"
                else (previous.get("error", "") or "")[:150]
            )
            history_lines.append(
                f"| {case_id} | {previous['status']} | {previous.get('execution_time_seconds')} | "
                f"{prev_actual} | **{record['status']}** | {record.get('execution_time_seconds')} |"
            )
    if history_lines:
        lines += [
            "",
            "## Regression history (failure → fix → re-run)",
            "",
            "Cases that failed at least once before passing. The original failure is "
            "preserved here, not deleted -- see `failure_analysis.md` for the full "
            "root-cause writeup.",
            "",
            "| Test ID | Previous Status | Previous Time (s) | Previous Result/Error | Current Status | Current Time (s) |",
            "|---------|-----------------|--------------------|--------------------------|----------------|-------------------|",
            *history_lines,
        ]

    lines += [
        "",
        "## Prepared but not yet executed",
        "",
        "| Test ID | Scenario | Mode | Reason |",
        "|---------|----------|------|--------|",
    ]
    for case_id, spec in specs.items():
        if case_id in executed_ids:
            continue
        mode, _ = CASES[case_id]
        reason = (
            "Requires `--mode live --yes` (real Groq API calls; see README.md)"
            if mode == "live"
            else "Not yet selected in a run of `run_evaluation.py`"
        )
        lines.append(f"| {case_id} | {spec['name']} | {mode} | {reason} |")

    lines.append("")
    RESULTS_MD_PATH.write_text("\n".join(lines))


def _summarize_actual_result(actual_result) -> str:
    if actual_result is None:
        return ""
    if isinstance(actual_result, dict):
        parts = []
        for key in ("overall_score", "alignment_label", "status", "llm_model", "groq_model"):
            if key in actual_result:
                parts.append(f"{key}={actual_result[key]}")
        if parts:
            return ", ".join(parts)
        return json.dumps(actual_result)[:100]
    return str(actual_result)[:100]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--mode",
        choices=["deterministic", "mocked", "live", "all"],
        default=None,
        help="Which category of cases to run. Default: deterministic + mocked (no API cost).",
    )
    parser.add_argument("--case", help="Run a single test case by ID (e.g. TC-06), ignoring --mode's filtering.")
    parser.add_argument("--list", action="store_true", help="List all test cases and exit. No execution.")
    parser.add_argument(
        "--yes",
        action="store_true",
        help="Required to actually execute live-mode cases (real Groq API calls). "
        "Without it, live mode only prints the execution plan and exits.",
    )
    args = parser.parse_args()

    specs = load_test_case_specs()

    if args.list:
        for case_id, spec in specs.items():
            mode, _ = CASES[case_id]
            print(f"{case_id}  [{mode}]  {spec['name']}")
            print(f"    condition: {spec['condition_tested']}")
        return 0

    if args.case:
        if args.case not in specs:
            print(f"Unknown test case ID: {args.case}", file=sys.stderr)
            return 2
        case_ids = [args.case]
    elif args.mode in (None, "deterministic", "mocked", "all"):
        selected_modes = {"deterministic", "mocked"} if args.mode in (None, "all") else {args.mode}
        if args.mode == "all":
            selected_modes = {"deterministic", "mocked", "live"}
        case_ids = [cid for cid in specs if CASES[cid][0] in selected_modes]
    else:  # args.mode == "live"
        case_ids = [cid for cid in specs if CASES[cid][0] == "live"]

    live_case_ids = [cid for cid in case_ids if CASES[cid][0] == "live"]

    if live_case_ids and not args.yes:
        print(
            "The following case(s) require real Groq API calls and will NOT be "
            "run without --yes:"
        )
        print_plan(live_case_ids, specs)
        print(
            "Re-run with `--yes` (and, if you only want the live cases, "
            "`--mode live --yes`) to actually execute them.\n"
        )
        case_ids = [cid for cid in case_ids if cid not in live_case_ids]
        if not case_ids:
            return 0

    print_plan(case_ids, specs)

    existing_results = load_existing_results()
    results = run_cases(case_ids, specs, existing_results)
    write_results_json(results)
    write_results_md(results, specs)

    print(f"\nResults written to {RESULTS_JSON_PATH.relative_to(REPO_ROOT)} and {RESULTS_MD_PATH.relative_to(REPO_ROOT)}")

    failures = [cid for cid in case_ids if results["runs"][cid]["status"] != "PASS"]
    if failures:
        print(f"\n{len(failures)} case(s) did not pass: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
