# faithfulness_eval.py
from openai import OpenAI
from dotenv import load_dotenv
from pydantic import BaseModel
from typing import List
import json

load_dotenv()
client = OpenAI()

class FaithfulnessResult(BaseModel):
    score: float  # 0-1
    claims: List[str]
    supported_claims: List[str]
    unsupported_claims: List[str]

def evaluate_faithfulness(context: str, answer: str) -> FaithfulnessResult:
    """Check if all claims in the answer are supported by context."""

    # Step 1: Extract claims from answer
    claims_response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "Extract factual claims from text."},
            {"role": "user", "content": f"""
Extract all factual claims from this answer:

"{answer}"

Return JSON: {{"claims": ["claim 1", "claim 2", ...]}}"""}
        ],
        response_format={"type": "json_object"}
    )

    claims_data = json.loads(claims_response.choices[0].message.content)
    claims = claims_data.get("claims", [])

    if not claims:
        return FaithfulnessResult(
            score=1.0, claims=[], supported_claims=[], unsupported_claims=[]
        )

    # Step 2: Check each claim against context
    supported = []
    unsupported = []

    for claim in claims:
        verify_response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "user", "content": f"""
Is this claim supported by the context?

Context: {context}

Claim: {claim}

Return JSON: {{"supported": true/false}}"""}
            ],
            response_format={"type": "json_object"}
        )

        verify_data = json.loads(verify_response.choices[0].message.content)

        if verify_data.get("supported"):
            supported.append(claim)
        else:
            unsupported.append(claim)

    score = len(supported) / len(claims) if claims else 1.0

    return FaithfulnessResult(
        score=score,
        claims=claims,
        supported_claims=supported,
        unsupported_claims=unsupported
    )

# Test
if __name__ == "__main__":
    context = """
    Employees receive 20 days of paid time off after 2 years of employment.
    Vacation must be requested 2 weeks in advance through the HR portal.
    """

    # Good answer (faithful)
    good_answer = "After 2 years, you get 20 vacation days. Request them 2 weeks ahead."

    # Bad answer (has hallucination)
    bad_answer = "You get 20 vacation days, and you can also carry over up to 30 days."

    print("=== Good Answer ===")
    result = evaluate_faithfulness(context, good_answer)
    print(f"Score: {result.score:.2f}")
    print(f"Supported: {result.supported_claims}")
    print(f"Unsupported: {result.unsupported_claims}")

    print("\n=== Bad Answer ===")
    result = evaluate_faithfulness(context, bad_answer)
    print(f"Score: {result.score:.2f}")
    print(f"Supported: {result.supported_claims}")
    print(f"Unsupported: {result.unsupported_claims}")
