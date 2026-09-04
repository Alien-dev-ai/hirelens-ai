from src.services.llm_service import GeminiLLMService
from src.services.candidate_analyzer import CandidateAnalyzer


cv_text = """
JOHN DOE

Software Engineer

Email: john.doe@example.com
Phone: +1 555 123 4567
Location: Lagos, Nigeria

SUMMARY

Software Engineer with experience building backend applications and APIs.

SKILLS

Python, FastAPI, PostgreSQL, Git, Docker

EXPERIENCE

Software Engineer
Tech Solutions Ltd
January 2022 - Present

- Developed REST APIs using Python and FastAPI.
- Worked with PostgreSQL databases.
- Built containerized applications using Docker.

Junior Developer
Startup Labs
June 2020 - December 2021

- Assisted with backend development.
- Used Git for version control.

EDUCATION

BSc Computer Science
University of Lagos

PROJECTS

Task Management API

Built a REST API for managing tasks using FastAPI and PostgreSQL.
"""


def main():
    print("\nInitializing Gemini...")

    llm_service = GeminiLLMService()

    analyzer = CandidateAnalyzer(llm_service)

    print("Analyzing candidate CV...\n")

    profile = analyzer.analyze(cv_text)

    print("=" * 60)
    print("STRUCTURED CANDIDATE PROFILE")
    print("=" * 60)

    print(profile.model_dump_json(indent=2))


if __name__ == "__main__":
    main()