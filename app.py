from __future__ import annotations

import os
import tempfile
from pathlib import Path

import streamlit as st

from src.services.llm_service import GeminiLLMService
from src.services.hirelens_pipeline import (
    HireLensPipeline,
    HireLensPipelineError,
)


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

    The same Gemini-backed pipeline is reused during the
    Streamlit session instead of being recreated unnecessarily.
    """

    llm_service = GeminiLLMService()

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
    Evidence-based candidate analysis and interview preparation.
    HireLens analyzes what is explicitly present in submitted documents —
    it does not make hiring decisions.
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
        "HireLens provides evidence-based analysis. "
        "No automatic hiring recommendation is generated."
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
        # CANDIDATE OVERVIEW
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
                "⏳ The Gemini API quota or rate limit has been reached. "
                "Please wait a bit and try again later."
            )

        else:

            # str(exc) is safe to show directly: it is built only from
            # pipeline/service stage descriptions and the Gemini API's own
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