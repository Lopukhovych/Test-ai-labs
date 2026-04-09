# simple_eval.py
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
import json

load_dotenv()
client = OpenAI()

class CorrectnessScore(BaseModel):
    score: int  # 1-5
    reason: str
    matches_expected: bool

def check_correctness(question: str, answer: str, expected: str) -> CorrectnessScore:
    """Check if answer matches expected answer."""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You evaluate AI answers."},
            {"role": "user", "content": f"""
Compare the answer to the expected answer.

Question: {question}
Expected Answer: {expected}
Actual Answer: {answer}

Return JSON:
{{
    "score": 1-5 (1=completely wrong, 5=perfect match),
    "reason": "brief explanation",
    "matches_expected": true/false
}}"""}
        ],
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return CorrectnessScore(**data)

# Test
if __name__ == "__main__":
    result = check_correctness(
        question="How many vacation days after 2 years?",
        answer="Employees receive twenty days of paid time off after working for 2 years.",
        expected="20 days"
    )

    print(f"Score: {result.score}/5")
    print(f"Matches: {result.matches_expected}")
    print(f"Reason: {result.reason}")
