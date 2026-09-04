# HireLens AI

HireLens AI is a recruiter-facing candidate analysis and **decision-support**
tool. It analyzes a candidate's CV and a job description and produces a
structured, evidence-grounded dossier: a normalized candidate profile,
structured job requirements, requirement-by-requirement evidence matches, a
deterministic candidate-to-role alignment score, an LLM-synthesized recruiter
analysis (summary, strengths, gaps, and interview validation areas), and
suggested interview questions.

HireLens is decision support, not a hiring decision:

- The **alignment score** is computed entirely by deterministic application
  logic (see `src/services/alignment_scorer.py`) from the evidence already
  gathered — the LLM is never asked to invent or adjust it.
- "No evidence found" (or a gap in the recruiter analysis) means no explicit
  evidence was found in the submitted documents — **never** that the
  candidate lacks the qualification. Gaps are always phrased as "not
  evidenced in the submitted CV," never as a confirmed absence.
- Nothing in HireLens considers or infers protected characteristics (race,
  ethnicity, religion, gender, age, disability, marital status, nationality,
  appearance, health, or similar).
- Nothing in HireLens claims certainty, an objective probability of hiring
  success, or states a hire/reject outcome. The human recruiter always makes
  the final call.

The pipeline:

```
Document extraction
  → Candidate analysis
  → Job description analysis
  → Evidence matching
  → Deterministic alignment scoring
  → LLM recruiter analysis synthesis
  → Interview question generation
```

LLM calls (candidate analysis, job description analysis, recruiter analysis
synthesis, and — where deterministic matching/logic is inconclusive —
evidence matching and interview question generation) are powered by
[Groq](https://groq.com/). Alignment scoring makes no LLM calls at all.

## Installation

Requires Python 3.10+.

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

## Environment variables

Copy `.env.example` to `.env` and fill in your own Groq API key:

```bash
cp .env.example .env
```

```
# Groq API configuration
GROQ_API_KEY=your-groq-api-key-here

# Optional model configuration
GROQ_MODEL=llama-3.3-70b-versatile
```

- `GROQ_API_KEY` — **required**. Get one from the
  [Groq Console](https://console.groq.com/keys).
- `GROQ_MODEL` — optional. Defaults to `llama-3.3-70b-versatile` if unset.

Never commit your actual `.env` file or a real API key — `.env` is already
listed in `.gitignore`.

## Getting started

1. Install dependencies and configure `.env` as above.
2. Run the automated test suite (no API key needed — it uses fakes/mocks,
   never a real Groq call):

   ```bash
   ./venv/bin/python -m pytest -q
   ```

3. Optionally, verify your Groq credentials work with a real API call:

   ```bash
   python test_groq.py
   ```

   A few other manual, real-API smoke-test scripts are also available for
   exercising individual pipeline stages end to end (`test_candidate_real.py`,
   `test_jd_real.py`, `test_pipeline_real.py`). They are excluded from
   automated test discovery (see `pytest.ini`) since they make real API calls
   and require a valid `GROQ_API_KEY`.

## Running the Streamlit application

```bash
streamlit run app.py
```

Then, in the browser tab that opens: upload a candidate CV (PDF or DOCX) in
the sidebar, paste a job description, and click **Analyze Candidate**.
