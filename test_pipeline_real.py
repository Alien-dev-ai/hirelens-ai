from src.services.llm_service import GroqLLMService
from src.services.hirelens_pipeline import HireLensPipeline


def main():
    print("=" * 70)
    print("HIRELENS AI — REAL END-TO-END PIPELINE TEST")
    print("=" * 70)

    cv_file_path = "data/samples/john_doe_cv.docx"

    job_description = """
Backend Engineer

We are looking for a Backend Engineer with strong experience in Python.

Requirements:
- Strong experience with Python
- Experience building REST APIs using FastAPI
- Experience working with PostgreSQL databases
- Experience using Git for version control
- Experience with Docker and containerized applications

Preferred Qualifications:
- Experience with AWS cloud services
- Experience with Kubernetes
- Bachelor's degree in Computer Science or a related field

Responsibilities:
- Build and maintain backend services and APIs.
- Design and maintain database integrations.
- Collaborate with frontend engineers.
- Participate in code reviews.
"""

    try:
        print("\n[1/5] Initializing Groq...")
        llm_service = GroqLLMService()

        print("[2/5] Initializing HireLens pipeline...")
        pipeline = HireLensPipeline.create(llm_service)

        print("[3/7] Extracting and analyzing candidate CV...")
        print("[4/7] Analyzing job description...")
        print("[5/7] Matching evidence...")
        print("[6/7] Computing alignment score and recruiter analysis...")
        print("[7/7] Generating interview questions...")

        dossier = pipeline.analyze(
            cv_file_path=cv_file_path,
            job_description=job_description,
        )

        print("\n" + "=" * 70)
        print("HIRELENS ANALYSIS COMPLETE")
        print("=" * 70)

        print(dossier.model_dump_json(indent=2))

    except Exception as exc:
        print("\n" + "=" * 70)
        print("PIPELINE ERROR")
        print("=" * 70)
        print(f"{type(exc).__name__}: {exc}")

        # Helpful during development
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()