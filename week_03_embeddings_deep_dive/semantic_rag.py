# semantic_rag.py
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
import numpy as np
from typing import List
from dataclasses import dataclass

load_dotenv()
client = OpenAI()

@dataclass
class Document:
    content: str
    filename: str
    embedding: List[float]

def get_embedding(text: str) -> List[float]:
    response = client.embeddings.create(
        model="text-embedding-3-small",
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

    def search(self, query: str, top_k: int = 2) -> List[Document]:
        """Find most relevant documents using semantic search."""
        query_embedding = get_embedding(query)

        # Calculate similarities
        scored = []
        for doc in self.documents:
            score = cosine_similarity(query_embedding, doc.embedding)
            scored.append((doc, score))

        # Sort by similarity
        scored.sort(key=lambda x: x[1], reverse=True)
        print(f'scored: {[score for _, score in scored]}')
        return [doc for doc, score in scored[:top_k]]

    def ask(self, question: str) -> str:
        """Ask a question, get answer from documents."""

        # Semantic search
        relevant_docs = self.search(question, top_k=2)

        # Build context
        context = "\n\n".join([
            f"[{doc.filename}]:\n{doc.content}"
            for doc in relevant_docs
        ])

        # Generate answer
        response = client.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Answer based on the provided context only."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )

        return response.choices[0].message.content

# Test
if __name__ == "__main__":
    rag = SemanticRAG("docs")

    # These queries now work with synonyms!
    questions = [
        "What's the PTO policy?",  # Finds "vacation" docs
        "Can I work remotely?",     # Finds "work from home" docs
        "What retirement plans exist?"  # Finds "401k" docs
    ]

    for q in questions:
        print(f"\nQ: {q}")
        answer = rag.ask(q)
        print(f"A: {answer}")
