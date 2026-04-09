# semantic_search.py
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from typing import List
from dataclasses import dataclass

load_dotenv()
client = OpenAI()

@dataclass
class Document:
    content: str
    embedding: List[float] = None

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class SemanticSearch:
    """A simple semantic search engine."""

    def __init__(self):
        self.documents: List[Document] = []

    def add_document(self, content: str):
        """Add a document and compute its embedding."""
        doc = Document(
            content=content,
            embedding=get_embedding(content)
        )
        self.documents.append(doc)
        print(f"Added: {content[:50]}...")

    def search(self, query: str, top_k: int = 3) -> List[tuple]:
        """Find most similar documents to query."""
        query_embedding = get_embedding(query)

        # Calculate similarity to all documents
        results = []
        for doc in self.documents:
            sim = cosine_similarity(query_embedding, doc.embedding)
            results.append((doc.content, sim))

        # Sort by similarity
        results.sort(key=lambda x: x[1], reverse=True)
        return results[:top_k]

# Test
if __name__ == "__main__":
    search = SemanticSearch()

    # Add documents
    documents = [
        "Employees receive 20 days of paid time off per year",
        "The company 401k plan matches up to 4% of salary",
        "Remote work is allowed on Mondays and Fridays",
        "Health insurance covers medical, dental, and vision",
        "Annual performance reviews happen in December",
        "The office is located in downtown San Francisco",
        "New employees receive a $500 equipment allowance"
    ]

    for doc in documents:
        search.add_document(doc)

    # Search
    queries = [
        "vacation policy",  # Should find PTO
        "working from home",  # Should find remote work
        "retirement benefits"  # Should find 401k
    ]

    for query in queries:
        print(f"\n=== Query: '{query}' ===")
        results = search.search(query, top_k=2)
        for content, score in results:
            print(f"  [{score:.4f}] {content}")
