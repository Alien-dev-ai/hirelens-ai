from __future__ import annotations

import logging
import os
import tempfile
from pathlib import Path

import streamlit as st

from src.services.llm_service import GroqLLMService
from src.services.hirelens_pipeline import (
    HireLensPipeline,
    HireLensPipelineError,
)

# Application entry point: configure the standard logging module once so the
# pipeline's stage-boundary logs (src/services/hirelens_pipeline.py) are
# actually visible in the console. A no-op on Streamlit's script reruns,
# since basicConfig() only takes effect the first time (no handlers exist
# yet on the root logger). No CV/JD content is ever logged — see that
# module's "SECURITY NOTE" docstring.
logging.basicConfig(level=logging.INFO)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="HireLens AI",
    page_icon="🔍",
    layout="wide",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    .main {
        background-color: #0E1117;
    }

    .hero-title {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0;
    }

    .hero-subtitle {
        font-size: 1.1rem;
        color: #9CA3AF;
        margin-bottom: 2rem;
    }

    .section-title {
        font-size: 1.4rem;
        font-weight: 600;
        margin-top: 1rem;
    }

    .status-found {
        color: #22C55E;
        font-weight: 600;
    }

    .status-missing {
        color: #F59E0B;
        font-weight: 600;
    }

    .status-verify {
        color: #60A5FA;
        font-weight: 600;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# HELPER FUNCTIONS
# ============================================================

@st.cache_resource
def get_pipeline():
    """
    Create and cache the HireLens pipeline.

    The same Groq-backed pipeline is reused during the
    Streamlit session instead of being recreated unnecessarily.
    """

    llm_service = GroqLLMService()

    pipeline = HireLensPipeline.create(
        llm_service=llm_service
    )

    return pipeline


def save_uploaded_file(uploaded_file) -> str:
    """
    Save a Streamlit uploaded file temporarily.

    Returns:
        Path to the temporary file.
    """

    suffix = Path(uploaded_file.name).suffix

    with tempfile.NamedTemporaryFile(
        delete=False,
        suffix=suffix,
    ) as temp_file:

        temp_file.write(uploaded_file.getbuffer())

        return temp_file.name


def status_badge(status: str) -> str:
    """
    Convert evidence status into a readable label.
    """

    labels = {
        "evidence_found": "✅ Evidence Found",
        "no_evidence_found": "⚠️ No Evidence Found",
        "needs_verification": "🔎 Needs Verification",
    }

    return labels.get(status, status)


_IMPORTANCE_LABELS = {
    "required": "Required qualifications",
    "preferred": "Preferred qualifications",
    "unclear": "Requirements with unclear importance",
}

_IMPORTANCE_ORDER = ["required", "preferred", "unclear"]

_STATUS_ICON = {
    "evidence_found": "✅",
    "no_evidence_found": "❔",
    "needs_verification": "🔎",
}

_ALIGNMENT_BAND_STYLE = {
    "Strong Alignment": "success",
    "Moderate Alignment": "info",
    "Limited Alignment": "warning",
    "Minimal Alignment": "warning",
}


def render_requirement_breakdown(job_requirements, evidence_matches) -> None:
    """
    Render a compact, at-a-glance checklist of requirements grouped by
    importance, so a recruiter isn't forced to open every requirement's
    detailed expander just to see the overall shape of the match.

    This is a display of the same EvidenceMatch statuses shown in detail
    further down the page — a different view of the same data, not a new
    judgment.
    """

    matches_by_requirement_id = {
        match.requirement_id: match for match in evidence_matches
    }

    st.markdown("### 📋 Requirement Breakdown")

    for level in _IMPORTANCE_ORDER:

        level_requirements = [
            requirement
            for requirement in job_requirements.requirements
            if requirement.importance == level
        ]

        if not level_requirements:
            continue

        st.markdown(f"**{_IMPORTANCE_LABELS.get(level, level)}**")

        for requirement in level_requirements:

            match = matches_by_requirement_id.get(requirement.id)
            icon = _STATUS_ICON.get(match.status, "❔") if match else "❔"

            st.write(f"{icon} {requirement.requirement}")


def build_evidence_coverage_summary(job_requirements, evidence_matches) -> list[dict]:
    """
    Build a neutral, factual rollup of evidence coverage per importance level.

    This is a plain grouping of the EvidenceMatch statuses that the pipeline
    already produced — evidence_found / no_evidence_found / needs_verification
    — by whether the job description marked the requirement as required,
    preferred, or unclear, keeping the specific requirement text for each
    status so the summary can name them. It adds no judgment of its own: it
    does not score, rank, or state whether a candidate is suitable for the
    role.
    """

    matches_by_requirement_id = {
        match.requirement_id: match for match in evidence_matches
    }

    groups_by_importance = {
        level: {
            "evidence_found": [],
            "no_evidence_found": [],
            "needs_verification": [],
        }
        for level in _IMPORTANCE_ORDER
    }

    for requirement in job_requirements.requirements:

        match = matches_by_requirement_id.get(requirement.id)

        if match is None:
            continue

        level = requirement.importance
        group = groups_by_importance.get(level)

        if group is None or match.status not in group:
            continue

        group[match.status].append(requirement.requirement)

    return [
        {"level": level, **groups_by_importance[level]}
        for level in _IMPORTANCE_ORDER
        if any(groups_by_importance[level].values())
    ]


def _join_requirement_names(names: list[str]) -> str:
    """Join requirement names into a readable comma/"and"-separated list."""

    if len(names) == 1:
        return f"'{names[0]}'"

    quoted = [f"'{name}'" for name in names]

    return ", ".join(quoted[:-1]) + f", and {quoted[-1]}"


def render_evidence_coverage_summary(summary_rows: list[dict]) -> None:
    """
    Render the evidence coverage summary as neutral, readable sentences.

    Names the specific requirements behind each status so the summary is
    more than a bare count, while deliberately avoiding any suitability,
    fit, or hiring language — it only ever reports what was and wasn't
    explicitly found in the submitted documents, never what the candidate
    does or doesn't "have."
    """

    if not summary_rows:
        return

    st.markdown("### 📊 Evidence Coverage Summary")

    for row in summary_rows:

        label = _IMPORTANCE_LABELS.get(row["level"], row["level"])
        sentences = []

        found = row["evidence_found"]
        missing = row["no_evidence_found"]
        unverified = row["needs_verification"]

        if found:
            sentences.append(
                f"Explicit evidence was found in the submitted documents "
                f"for {_join_requirement_names(found)}."
            )

        if missing:
            sentences.append(
                f"No explicit evidence was found for "
                f"{_join_requirement_names(missing)}. This does not mean "
                f"the candidate lacks these — only that they were not "
                f"explicitly stated in the documents reviewed. The "
                f"interview questions below offer a neutral way to ask "
                f"about them directly."
            )

        if unverified:
            sentences.append(
                f"The evidence found for {_join_requirement_names(unverified)} "
                f"was partial or ambiguous and may be worth verifying "
                f"directly with the candidate."
            )

        st.markdown(f"**{label}**")
        st.write(" ".join(sentences))

    st.caption(
        "This reflects only what was explicitly found in the submitted "
        "documents. It is not an assessment of the candidate's suitability "
        "— hiring decisions remain with the human recruiter."
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="hero-title">🔍 HireLens AI</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="hero-subtitle">
    Evidence-grounded candidate analysis, alignment scoring, and interview
    preparation. HireLens synthesizes what is explicitly present in
    submitted documents into decision support for a recruiter — it does
    not make the hiring decision itself.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("📄 Candidate Input")

    uploaded_cv = st.file_uploader(
        "Upload Candidate CV",
        type=["pdf", "docx"],
        help="Upload a PDF or DOCX candidate resume.",
    )

    st.divider()

    st.caption(
        "HireLens provides evidence-grounded analysis and a deterministic "
        "alignment score to support a recruiter's judgment. It never makes "
        "the hiring decision itself."
    )


# ============================================================
# MAIN INPUT AREA
# ============================================================

st.subheader("Job Description")

job_description = st.text_area(
    "Paste the job description below",
    height=300,
    placeholder="""
Example:

Backend Engineer

Requirements:
- Strong experience with Python
- Experience building REST APIs using FastAPI
- Experience with PostgreSQL
- Experience using Docker

Preferred:
- AWS
- Kubernetes
""",
)


analyze_button = st.button(
    "🔍 Analyze Candidate",
    type="primary",
    use_container_width=True,
)


# ============================================================
# VALIDATION
# ============================================================

if analyze_button:

    if uploaded_cv is None:
        st.error("Please upload a candidate CV first.")
        st.stop()

    if not job_description.strip():
        st.error("Please provide a job description.")
        st.stop()

    temp_file_path = None

    try:

        # ----------------------------------------------------
        # SAVE CV
        # ----------------------------------------------------

        with st.spinner("Preparing candidate document..."):

            temp_file_path = save_uploaded_file(
                uploaded_cv
            )

        # ----------------------------------------------------
        # INITIALIZE PIPELINE
        # ----------------------------------------------------

        with st.spinner("Initializing HireLens AI..."):

            pipeline = get_pipeline()

        # ----------------------------------------------------
        # RUN PIPELINE
        # ----------------------------------------------------

        with st.status(
            "Analyzing candidate...",
            expanded=True,
        ) as status:

            st.write("📄 Extracting candidate document...")
            st.write("🧠 Analyzing candidate profile...")
            st.write("📋 Analyzing job requirements...")
            st.write("🔎 Matching evidence...")
            st.write("🧮 Computing alignment score...")
            st.write("🧭 Synthesizing recruiter analysis...")
            st.write("💬 Generating interview questions...")

            dossier = pipeline.analyze(
                cv_file_path=temp_file_path,
                job_description=job_description,
            )

            status.update(
                label="Analysis complete!",
                state="complete",
                expanded=False,
            )

        # ====================================================
        # RESULTS
        # ====================================================

        st.success("Candidate analysis completed successfully!")

        # ----------------------------------------------------
        # CANDIDATE OVERVIEW (score + high-level summary, up front)
        # ----------------------------------------------------

        st.divider()

        st.header("🧭 Candidate Overview")

        alignment = dossier.alignment_score

        overview_col1, overview_col2 = st.columns([2, 1])

        with overview_col1:
            st.subheader(dossier.candidate_profile.full_name or "Candidate")

            if dossier.job_requirements.role_title:
                st.caption(f"Evaluated against: {dossier.job_requirements.role_title}")

        with overview_col2:
            st.metric("Overall Match Score", f"{alignment.overall_score} / 100")

            band_style = _ALIGNMENT_BAND_STYLE.get(alignment.alignment_label, "info")
            getattr(st, band_style)(alignment.alignment_label)

        st.caption(
            "This score reflects how strongly the candidate's documented "
            "evidence covers the job's stated requirements. It is decision "
            "support, not a hiring decision — the recruiter makes the final "
            "call."
        )

        st.markdown("### 🧭 Recruiter Analysis")
        st.write(dossier.recruiter_analysis.summary)

        strengths_col, gaps_col = st.columns(2)

        with strengths_col:
            st.markdown("### ✅ Key Strengths")
            if dossier.recruiter_analysis.strengths:
                for strength in dossier.recruiter_analysis.strengths:
                    st.write(f"- {strength}")
            else:
                st.info("No specific strengths were highlighted.")

        with gaps_col:
            st.markdown("### 🔍 Key Gaps / Missing Evidence")
            if dossier.recruiter_analysis.gaps:
                for gap in dossier.recruiter_analysis.gaps:
                    st.write(f"- {gap}")
            else:
                st.info("No gaps were highlighted.")

        st.divider()

        render_requirement_breakdown(dossier.job_requirements, dossier.evidence_matches)

        st.divider()

        st.markdown("### 📋 Recruiter Verification Checklist")

        if dossier.recruiter_analysis.validation_areas:
            for index, area in enumerate(dossier.recruiter_analysis.validation_areas, start=1):
                st.write(f"{index}. {area}")
        else:
            st.info("No specific validation areas were identified.")

        st.caption(
            "The alignment score and recruiter analysis above are "
            "decision-support only — they never claim certainty, an "
            "objective hiring probability, or a hiring decision, and they "
            "never consider protected characteristics. The human recruiter "
            "always makes the final call."
        )

        # ----------------------------------------------------
        # CANDIDATE PROFILE DETAILS
        # ----------------------------------------------------

        st.divider()

        st.header("👤 Candidate Profile")

        candidate = dossier.candidate_profile

        col1, col2 = st.columns(2)

        with col1:
            st.subheader(candidate.full_name or "Candidate")

            if candidate.email:
                st.write(f"📧 {candidate.email}")

            if candidate.phone:
                st.write(f"📱 {candidate.phone}")

        with col2:

            if candidate.location:
                st.write(f"📍 {candidate.location}")

            st.write(
                f"💼 {len(candidate.experiences)} experience record(s)"
            )

            st.write(
                f"🛠️ {len(candidate.skills)} identified skill(s)"
            )

        if candidate.summary:

            st.markdown("### Professional Summary")

            st.write(candidate.summary)

        # ----------------------------------------------------
        # SKILLS
        # ----------------------------------------------------

        st.markdown("### Skills")

        if candidate.skills:

            skills_text = " • ".join(candidate.skills)

            st.info(skills_text)

        else:

            st.info("No skills explicitly identified.")

        # ====================================================
        # JOB REQUIREMENTS
        # ====================================================

        st.divider()

        st.header("📋 Job Requirements")

        requirements = dossier.job_requirements

        st.subheader(requirements.role_title)

        for requirement in requirements.requirements:

            importance = requirement.importance.upper()

            with st.expander(
                f"{requirement.id} — {requirement.requirement}"
            ):

                st.write(
                    f"**Category:** {requirement.category}"
                )

                st.write(
                    f"**Importance:** {importance}"
                )

        # ====================================================
        # EVIDENCE MATCHING
        # ====================================================

        st.divider()

        st.header("🔎 Evidence Analysis")

        evidence_coverage_summary = build_evidence_coverage_summary(
            dossier.job_requirements,
            dossier.evidence_matches,
        )

        render_evidence_coverage_summary(evidence_coverage_summary)

        st.divider()

        for match in dossier.evidence_matches:

            if match.status == "evidence_found":

                st.success(
                    f"**{match.requirement}**\n\n"
                    f"{status_badge(match.status)}"
                )

            elif match.status == "no_evidence_found":

                st.warning(
                    f"**{match.requirement}**\n\n"
                    f"{status_badge(match.status)}"
                )

            else:

                st.info(
                    f"**{match.requirement}**\n\n"
                    f"{status_badge(match.status)}"
                )

            st.write(match.explanation)

            if match.evidence:

                st.markdown("**Supporting Evidence:**")

                for evidence in match.evidence:

                    st.code(evidence)

            st.caption(
                f"Confidence: {match.confidence:.0%}"
            )

            st.divider()

        # ====================================================
        # INTERVIEW QUESTIONS
        # ====================================================

        st.header("💬 Interview Questions")

        for index, question in enumerate(
            dossier.interview_questions,
            start=1,
        ):

            with st.expander(
                f"{index}. {question.focus_requirement}"
            ):

                st.markdown(
                    f"### {question.question}"
                )

                st.write(
                    f"**Reason:** {question.reason}"
                )

                st.write(
                    f"**Priority:** {question.priority.upper()}"
                )

        # ====================================================
        # WARNINGS
        # ====================================================

        if dossier.warnings:

            st.divider()

            st.header("⚠️ Analysis Notes")

            for warning in dossier.warnings:

                st.warning(warning)

        # ====================================================
        # RAW JSON DOWNLOAD
        # ====================================================

        st.divider()

        st.header("📥 Export")

        dossier_json = dossier.model_dump_json(
            indent=2
        )

        st.download_button(
            label="Download Analysis as JSON",
            data=dossier_json,
            file_name="hirelens_analysis.json",
            mime="application/json",
            use_container_width=True,
        )


    except HireLensPipelineError as exc:

        if exc.is_quota_exceeded:

            st.warning(
                "⏳ The Groq API quota or rate limit has been reached. "
                "Please wait a bit and try again later."
            )

        else:

            # str(exc) is safe to show directly: it is built only from
            # pipeline/service stage descriptions and the Groq API's own
            # error text — never candidate CV or job description content.
            st.error(
                f"HireLens could not complete the analysis: {exc}"
            )

        with st.expander("Technical details"):
            st.exception(exc)

    except Exception as exc:

        st.error(
            f"An unexpected error occurred: {exc}"
        )

        with st.expander("Technical details"):
            st.exception(exc)

    finally:

        # ----------------------------------------------------
        # CLEAN UP TEMP FILE
        # ----------------------------------------------------

        if temp_file_path:

            try:

                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)

            except OSError:
                pass