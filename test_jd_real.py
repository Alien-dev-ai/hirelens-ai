from src.services.llm_service import GroqLLMService
from src.services.jd_analyzer import JobDescriptionAnalyzer


jd_text = """
Backend Engineer

We are looking for a Backend Engineer to join our engineering team.

Required Qualifications:

- Strong experience with Python.
- Experience building REST APIs using FastAPI.
- Experience working with PostgreSQL databases.
- Experience using Git for version control.
- Experience with Docker and containerized applications.

Preferred Qualifications:

- Experience with AWS cloud services.
- Experience with Kubernetes.

Responsibilities:

- Build and maintain backend services and APIs.
- Design and maintain database integrations.
- Collaborate with frontend engineers.
- Participate in code reviews.

A Bachelor's degree in Computer Science or a related field is preferred.
"""


def main():
    print("\nInitializing Groq...")

    llm_service = GroqLLMService()

    analyzer = JobDescriptionAnalyzer(llm_service)

    print("Analyzing job description...\n")

    requirements = analyzer.analyze(jd_text)

    print("=" * 60)
    print("STRUCTURED JOB REQUIREMENTS")
    print("=" * 60)

    print(requirements.model_dump_json(indent=2))


if __name__ == "__main__":
    main()