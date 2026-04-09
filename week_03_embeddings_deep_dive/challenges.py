# semantic_rag.py
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from typing import List, Tuple
from dataclasses import dataclass

load_dotenv()

@dataclass
class Document:
    content: str
    filename: str
    embedding: List[float]

client = OpenAI()
def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-large",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(a: List[float], b: List[float]) -> float:
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

class SemanticRAG:
    """RAG with semantic search instead of keyword search."""

    def __init__(self, docs_folder: str):
        self.documents = []
        self._load_documents(docs_folder)

    def _load_documents(self, folder: str):
        """Load and embed all documents."""
        folder_path = Path(folder)

        for file in folder_path.glob("*.txt"):
            content = file.read_text()
            embedding = get_embedding(content)

            doc = Document(
                content=content,
                filename=file.name,
                embedding=embedding
            )
            self.documents.append(doc)
            print(f"Embedded: {file.name}")

    def search(self, query: str, threshold: float) -> List[Tuple[str, float]]:
        """Find most relevant documents using semantic search."""
        query_embedding = get_embedding(query)

        # Calculate similarities
        scored = []
        for doc in self.documents:
            score = cosine_similarity(query_embedding, doc.embedding)
            scored.append((doc, score))

        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        # print(f'scored: {[score for _, score in scored]}')
        return [(doc.filename, score) for doc, score in scored if score >= threshold]

    # def ask(self, question: str) -> str:
    #     """Ask a question, get answer from documents."""

        # Semantic search
        # relevant_docs = self.search(question, top_k=2)

        # Build context
        # context = "\n\n".join([
        #     f"[{doc.filename}]:\n{doc.content}"
        #     for doc in relevant_docs
        # ])
        #
        # # Generate answer
        # response = client.chat.completions.create(
        #     model="gpt-5-mini",
        #     messages=[
        #         {"role": "system", "content": "Answer based on the provided context only."},
        #         {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
        #     ]
        # )
        #
        # return response.choices[0].message.content

# Test
if __name__ == "__main__":
    eng_emb = get_embedding("I love programming")
    sp_emb = get_embedding("Me encanta programar")
    print(f"Multilanguage similarity: {cosine_similarity(eng_emb, sp_emb)}")  # How close?

#     rag = SemanticRAG("docs")
#
#     # Test queries that should match documents
#     test_queries = [
#         "vacation policy",      # Should match benefits/vacation docs
#         "remote work",          # Should match remote work policy
#         "401k retirement",      # Should match benefits
#         "AI software",          # Should match company mission
#         "pizza preferences"     # Should have low similarity (no match)
#     ]
#
#     print("=" * 70)
#     print("THRESHOLD TESTING: Precision vs Recall Trade-off")
#     print("=" * 70)
#
#     for query in test_queries:
#         print(f"\n📌 Query: '{query}'")
#         print("-" * 70)
#
#         # Test different thresholds
#         for threshold in [0.6, 0.7, 0.8, 0.9]:
#             results = rag.search(query=query, threshold=threshold)
#
#             if results:
#                 print(f"Threshold {threshold}: Found {len(results)} document(s)")
#                 for filename, score in results:
#                     print(f"file; {filename}, score: {score}")
#                     # Print first 80 chars of each result
#                     # preview = doc.content.replace("\n", " ")[:80]
#                     # print(f"    • {doc.filename}: \"{preview}...\"")
#             # else:
#             #     print(f"\n  Threshold {threshold}: No results (too strict)")
#
#         print()
#
#     print("\n" + "=" * 70)
#     print("OBSERVATIONS:")
#     print("=" * 70)
#     print("""
# Lower threshold (0.6):   More results, some irrelevant (low precision)
# Higher threshold (0.9):  Fewer results, all relevant (high precision)
#
# Find the sweet spot that balances:
#   - Precision: Are all returned results relevant?
#   - Recall: Are all relevant results being returned?
# """)
