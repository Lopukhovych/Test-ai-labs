# pinecone_store.py
from pinecone import Pinecone, ServerlessSpec
from openai import OpenAI
from dotenv import load_dotenv
import os
import uuid

load_dotenv()


class PineconeVectorStore:
    """Vector store using Pinecone managed service."""

    def __init__(self, index_name: str = "documents"):
        self.pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        self.openai = OpenAI()
        self.index_name = index_name

        # Create index if it doesn't exist
        existing = [i.name for i in self.pc.list_indexes()]
        if index_name not in existing:
            self.pc.create_index(
                name=index_name,
                dimension=1536,  # text-embedding-3-small output size
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            print(f"Created index: {index_name}")

        self.index = self.pc.Index(index_name)

    def add_documents(self, documents: list[dict]):
        """Embed and upsert documents into Pinecone."""
        texts = [doc["text"] for doc in documents]
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )

        vectors = []
        for i, doc in enumerate(documents):
            vectors.append({
                "id": str(uuid.uuid4()),
                "values": response.data[i].embedding,
                "metadata": doc  # stored as-is, searchable via filters
            })

        self.index.upsert(vectors=vectors)
        print(f"Upserted {len(vectors)} vectors")

    def search(self, query: str, top_k: int = 3) -> list[dict]:
        """Query Pinecone for similar documents."""
        query_emb = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        results = self.index.query(
            vector=query_emb,
            top_k=top_k,
            include_metadata=True
        )

        return [
            {
                "text": match.metadata["text"],
                "score": match.score,
                **{k: v for k, v in match.metadata.items() if k != "text"}
            }
            for match in results.matches
        ]

    def search_with_filter(self, query: str, filters: dict, top_k: int = 3) -> list[dict]:
        """Pinecone supports metadata filtering at query time."""
        query_emb = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=query
        ).data[0].embedding

        results = self.index.query(
            vector=query_emb,
            top_k=top_k,
            include_metadata=True,
            filter=filters  # e.g. {"source": {"$eq": "article1"}}
        )

        return [
            {"text": m.metadata["text"], "score": m.score}
            for m in results.matches
        ]

    def delete_index(self):
        """Clean up — delete the index to avoid storage costs."""
        self.pc.delete_index(self.index_name)
        print(f"Deleted index: {self.index_name}")


# Test
if __name__ == "__main__":
    store = PineconeVectorStore(index_name="week4-demo")

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

    # Metadata filter example
    filtered = store.search_with_filter(
        query="machine learning",
        filters={"source": {"$eq": "article1"}}
    )
    print("\nFiltered search (article1 only):")
    for r in filtered:
        print(f"  [{r['score']:.4f}] {r['text']}")

    store.delete_index()  # clean up free-tier quota
