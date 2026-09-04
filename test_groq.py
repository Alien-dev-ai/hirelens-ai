"""Manual smoke test: verify a real Groq API connection works.

Not part of the automated test suite (excluded via pytest.ini's
`testpaths = tests`) — it makes a real network call and needs a real
GROQ_API_KEY. Run directly: `python test_groq.py`.
"""

import os

from dotenv import load_dotenv
from groq import Groq

load_dotenv()

api_key = os.getenv("GROQ_API_KEY")

if not api_key:
    raise ValueError(
        "GROQ_API_KEY was not found. Check your .env file."
    )

client = Groq(api_key=api_key)

response = client.chat.completions.create(
    model=os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile"),
    messages=[
        {
            "role": "user",
            "content": "Reply with exactly: HireLens Groq connection successful",
        }
    ],
)

print(response.choices[0].message.content)
