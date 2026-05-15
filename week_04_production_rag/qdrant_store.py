# qdrant_store.py
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import uuid

load_dotenv()

class VectorStore:
    """Simple vector store using Qdrant."""

    def __init__(self, collection_name: str = "documents"):
        # Use in-memory Qdrant (no server needed)
        self.client = QdrantClient(":memory:")
        self.openai = OpenAI()
        self.collection_name = collection_name

        # Create collection
        self.client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(
                size=1536,  # OpenAI embedding size
                distance=Distance.COSINE
            )
        )
        print(f"Created collection: {collection_name}")

    def _get_embedding(self, text: str) -> List[float]:
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=text
        )
        return response.data[0].embedding

    def add_documents(self, documents: List[dict]):
        """Add multiple documents efficiently."""
        points = []

        # Get all embeddings in batch
        texts = [doc["text"] for doc in documents]
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        for i, doc in enumerate(documents):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=response.data[i].embedding,
                payload=doc
            ))

        self.client.upsert(
            collection_name=self.collection_name,
            points=points
        )
        print(f"Added {len(documents)} documents")

    def search(self, query: str, top_k: int = 3) -> List[dict]:
        """Search for similar documents."""
        query_embedding = self._get_embedding(query)

        response = self.client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,  # type: ignore[arg-type]
            limit=top_k
        )
        print(f'results: {response.points}')

        return [
            {
                "text": hit.payload["text"],
                "score": hit.score,
                **{k: v for k, v in hit.payload.items() if k != "text"}
            }
            for hit in response.points
        ]

# Test
if __name__ == "__main__":
    store = VectorStore()

    docs = [
        {"text": "Python is great for machine learning", "source": "article1"},
        {"text": "JavaScript is used for web development", "source": "article2"},
        {"text": "TensorFlow is a popular ML framework", "source": "article3"},
    ]

    store.add_documents(docs)

    results = store.search("deep learning frameworks")
    print("\nSearch: 'deep learning frameworks'")
    for r in results:
        print(f"  [{r['score']:.4f}] {r['text']}")
