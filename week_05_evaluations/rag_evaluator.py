# rag_evaluator.py
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List
import json

load_dotenv()
client = OpenAI()

@dataclass
class EvalResult:
    question: str
    answer: str
    retrieval_score: float  # Did we find right docs?
    faithfulness_score: float  # Answer matches context?
    correctness_score: float  # Answer is correct?
    overall_score: float

def evaluate_rag_response(
        question: str,
        answer: str,
        context: str,
        expected_answer: str,
        expected_doc: str,
        retrieved_docs: List[str]
) -> EvalResult:
    """Comprehensive RAG evaluation."""

    # 1. Retrieval score: Did we find the expected document?
    retrieval_score = 1.0 if expected_doc in retrieved_docs else 0.0

    # 2. Faithfulness: Is answer grounded in context?
    faith_response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": f"""
Score how well the answer is supported by the context (0-1):

Context: {context}
Answer: {answer}

Return JSON: {{"score": 0.0-1.0, "reason": "..."}}"""}
        ],
        response_format={"type": "json_object"}
    )
    faithfulness_score = json.loads(faith_response.choices[0].message.content)["score"]

    # 3. Correctness: Does answer match expected?
    correct_response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": f"""
Score how well the answer matches the expected answer (0-1):

Question: {question}
Expected: {expected_answer}
Actual: {answer}

Return JSON: {{"score": 0.0-1.0}}"""}
        ],
        response_format={"type": "json_object"}
    )
    correctness_score = json.loads(correct_response.choices[0].message.content)["score"]

    # Overall score (weighted average)
    overall = (retrieval_score * 0.3 + faithfulness_score * 0.4 + correctness_score * 0.3)

    return EvalResult(
        question=question,
        answer=answer,
        retrieval_score=retrieval_score,
        faithfulness_score=faithfulness_score,
        correctness_score=correctness_score,
        overall_score=overall
    )

# Test
if __name__ == "__main__":
    result = evaluate_rag_response(
        question="How many vacation days after 2 years?",
        answer="After working for 2 years, employees receive 20 days of PTO.",
        context="Employees with 2-5 years tenure receive 20 days per year.",
        expected_answer="20 days",
        expected_doc="vacation_policy.txt",
        retrieved_docs=["vacation_policy.txt", "benefits.txt"]
    )

    print(f"Question: {result.question}")
    print(f"Answer: {result.answer}")
    print(f"\nScores:")
    print(f"  Retrieval:    {result.retrieval_score:.2f}")
    print(f"  Faithfulness: {result.faithfulness_score:.2f}")
    print(f"  Correctness:  {result.correctness_score:.2f}")
    print(f"  Overall:      {result.overall_score:.2f}")
