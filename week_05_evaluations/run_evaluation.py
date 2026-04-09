# run_evaluation.py
from rag_evaluator import evaluate_rag_response, EvalResult
from typing import List
import json

def run_evaluation_suite(rag_system, test_file: str = "test_data.json") -> List[EvalResult]:
    """Run all test cases through the RAG system and evaluate."""

    # Load test cases
    with open(test_file) as f:
        test_cases = json.load(f)

    results = []

    for test in test_cases:
        print(f"Testing: {test['question'][:50]}...")

        # Get RAG response (mock for now)
        # response = rag_system.ask(test['question'])

        # Mock response for demo
        mock_answer = f"Based on the {test['relevant_doc']}, {test['expected_answer']}."
        mock_context = f"Document content about {test['expected_answer']}."
        mock_docs = [test['relevant_doc']]

        result = evaluate_rag_response(
            question=test['question'],
            answer=mock_answer,
            context=mock_context,
            expected_answer=test['expected_answer'],
            expected_doc=test['relevant_doc'],
            retrieved_docs=mock_docs
        )

        results.append(result)

    return results

def print_summary(results: List[EvalResult]):
    """Print evaluation summary."""

    if not results:
        print("No results")
        return

    avg_retrieval = sum(r.retrieval_score for r in results) / len(results)
    avg_faith = sum(r.faithfulness_score for r in results) / len(results)
    avg_correct = sum(r.correctness_score for r in results) / len(results)
    avg_overall = sum(r.overall_score for r in results) / len(results)

    print("\n" + "="*50)
    print("EVALUATION SUMMARY")
    print("="*50)
    print(f"Total test cases: {len(results)}")
    print(f"\nAverage Scores:")
    print(f"  Retrieval:    {avg_retrieval:.2%}")
    print(f"  Faithfulness: {avg_faith:.2%}")
    print(f"  Correctness:  {avg_correct:.2%}")
    print(f"  Overall:      {avg_overall:.2%}")

    # Find failures
    failures = [r for r in results if r.overall_score < 0.7]
    if failures:
        print(f"\n⚠️  {len(failures)} test(s) below 70%:")
        for r in failures:
            print(f"  - {r.question[:50]}... ({r.overall_score:.2%})")

# Run
if __name__ == "__main__":
    # Create test data first
    exec(open("test_dataset.py").read())

    results = run_evaluation_suite(None)
    print_summary(results)
