import json

from dotenv import load_dotenv
from openai import OpenAI
from pydantic import BaseModel

load_dotenv()
client = OpenAI()


class ReasonedAnswer(BaseModel):
    reasoning: str
    answer: str
    confidence: float


def answer_with_reasoning(question: str) -> ReasonedAnswer:
    """Get an answer with step-by-step reasoning."""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": """Think through the problem step by step before answering.

Return JSON:
{
    "reasoning": "Step 1: ... Step 2: ... Step 3: ...",
    "answer": "Your final answer",
    "confidence": 0.0 to 1.0
}"""
            },
            {"role": "user", "content": question}
        ],
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return ReasonedAnswer(**data)


# Test
questions = [
    "If a train leaves at 9am going 60mph, and another leaves at 10am going 80mph, when do they meet?",
    "Should I use Python or JavaScript for a machine learning project?",
    "A bat and ball cost $1.10 total. The bat costs $1 more than the ball. How much is the ball?"
]

for q in questions:
    result = answer_with_reasoning(q)
    print(f"\nQ: {q}")
    print(f"Reasoning: {result.reasoning}")
    print(f"Answer: {result.answer}")
    print(f"Confidence: {result.confidence}")
