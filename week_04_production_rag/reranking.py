# reranking.py
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

def rerank_results(
        query: str,
        results: list[dict],
        top_k: int = 3
) -> list[dict]:
    """Re-rank search results using LLM scoring."""

    # Format results for scoring
    results_text = "\n".join([
        f"{i+1}. {r['text'][:300]}..."
        for i, r in enumerate(results)
    ])

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[{
            "role": "user",
            "content": f"""Score each search result 1-10 for relevance to the query.

Query: "{query}"

Results:
{results_text}

Return JSON: {{"scores": [score1, score2, ...], "reasoning": ["reason1", ...]}}"""
        }],
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    scores = data["scores"]

    # Add scores and sort
    for i, result in enumerate(results):
        result["rerank_score"] = scores[i] if i < len(scores) else 0

    ranked = sorted(results, key=lambda x: x["rerank_score"], reverse=True)
    # return ranked[:top_k]
    return ranked

# Test with mock results
if __name__ == "__main__":
    query = "How to reset password"
    results = [
        {"text": "Our company picnic will be held on Saturday.", "score": 0.85},
        {"text": "To reset your password, go to Settings > Security > Reset Password.", "score": 0.82},
        {"text": "Password requirements: 8 characters, 1 uppercase, 1 number.", "score": 0.80},
    ]

    reranked = rerank_results(query, results)
    print(f"Query: {query}\n")
    for r in reranked:
        print(f"Score {r['rerank_score']}/10: {r['text'][:60]}...")
