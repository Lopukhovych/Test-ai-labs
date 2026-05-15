# production_rag.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from dotenv import load_dotenv
from pathlib import Path
from typing import List
from dataclasses import dataclass
import uuid

load_dotenv()

@dataclass
class SearchResult:
    text: str
    score: float
    filename: str
    chunk_index: int

@dataclass
class RAGResponse:
    answer: str
    sources: List[SearchResult]

class ProductionRAG:
    """Production-ready RAG system."""

    def __init__(self, collection_name: str = "rag_docs"):
        self.qdrant = QdrantClient(":memory:")
        self.openai = OpenAI()
        self.collection = collection_name

        self.qdrant.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=1536, distance=Distance.COSINE)
        )

    def _chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple chunking."""
        chunks = []
        for i in range(0, len(text), chunk_size - 100):
            chunks.append(text[i:i + chunk_size])
        return chunks

    def index_folder(self, folder: str, chunk_size: int = 500):
        """Index all text files in a folder."""
        folder_path = Path(folder)
        all_points = []

        for file in folder_path.glob("*.txt"):
            print(f'file: {file}')
            content = file.read_text()
            chunks = self._chunk_text(content, chunk_size)

            # Get embeddings for all chunks
            response = self.openai.embeddings.create(
                model="text-embedding-3-small",
                input=chunks
            )

            for i, chunk in enumerate(chunks):
                all_points.append(PointStruct(
                    id=str(uuid.uuid4()),
                    vector=response.data[i].embedding,
                    payload={
                        "text": chunk,
                        "filename": file.name,
                        "chunk_index": i
                    }
                ))

            print(f"Indexed: {file.name} ({len(chunks)} chunks)")

        self.qdrant.upsert(collection_name=self.collection, points=all_points)

    def ask(self, question: str, top_k: int = 3) -> RAGResponse:
        """Ask a question, get answer with citations."""

        # Search
        query_emb = self.openai.embeddings.create(
            model="text-embedding-3-small", input=question
        ).data[0].embedding

        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_emb,  # type: ignore[arg-type]
            limit=top_k
        ).points
        for i, hit in enumerate(results):
            print(f'hit #{i}:  {hit}')

        search_results = [
            SearchResult(
                text=hit.payload["text"],
                score=hit.score,
                filename=hit.payload["filename"],
                chunk_index=hit.payload["chunk_index"]
            )
            for hit in results
        ]

        # Build context
        context = "\n\n".join([
            f"[{r.filename}]: {r.text}" for r in search_results
        ])

        # Generate answer
        response = self.openai.chat.completions.create(
            model="gpt-5-mini",
            messages=[
                {"role": "system", "content": "Answer based on context. Cite sources."},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}
            ]
        )

        return RAGResponse(
            answer=response.choices[0].message.content,
            sources=search_results
        )

# Test
if __name__ == "__main__":
    rag = ProductionRAG()
    rag.index_folder("../week_03_embeddings_deep_dive/docs")

    response = rag.ask("What's the vacation policy?")
    print(f"A: {response.answer}")
    print(f"\nSources: {[s.filename for s in response.sources]}")
