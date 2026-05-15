"""
metadata_search.py — Document update detection and versioned re-indexing.

Key concepts:
  DocumentRegistry   — persists document state (hash, version, chunk IDs) to JSON
  VersionedDocumentStore — wraps Qdrant, detects changes, removes stale chunks,
                           re-indexes, and optionally archives old versions
"""

import hashlib
import json
import uuid
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from qdrant_client import QdrantClient
from embedding_cache import EmbeddingCache
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchValue,
    PointIdsList,
    PointStruct,
    VectorParams,
)

load_dotenv()


# ---------------------------------------------------------------------------
# Registry — what we know about each document across versions
# ---------------------------------------------------------------------------

@dataclass
class VersionSnapshot:
    """Records chunk IDs for one historical version of a document."""
    version: int
    content_hash: str
    chunk_ids: list[str]
    indexed_at: str


@dataclass
class DocRecord:
    """Full state for one document (current version + optional history)."""
    doc_id: str
    current_version: int
    current_hash: str
    current_chunk_ids: list[str]
    indexed_at: str
    metadata: dict = field(default_factory=dict)
    # Each entry is a VersionSnapshot serialised as a dict
    version_history: list[dict] = field(default_factory=list)


class DocumentRegistry:
    """
    Persists document records to a JSON file so change detection survives
    process restarts.
    """

    def __init__(self, registry_path: str = "doc_registry.json"):
        self.path = Path(registry_path)
        self._records: dict[str, DocRecord] = {}
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if self.path.exists():
            raw = json.loads(self.path.read_text())
            self._records = {k: DocRecord(**v) for k, v in raw.items()}

    def _save(self):
        self.path.write_text(
            json.dumps({k: asdict(v) for k, v in self._records.items()}, indent=2)
        )

    # ------------------------------------------------------------------
    # CRUD
    # ------------------------------------------------------------------

    def get(self, doc_id: str) -> Optional[DocRecord]:
        return self._records.get(doc_id)

    def put(self, record: DocRecord):
        self._records[record.doc_id] = record
        self._save()

    def delete(self, doc_id: str):
        self._records.pop(doc_id, None)
        self._save()

    def all(self) -> list[DocRecord]:
        return list(self._records.values())

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def hash_content(text: str) -> str:
        """SHA-256 of the document text — used as the change fingerprint."""
        return hashlib.sha256(text.encode()).hexdigest()


# ---------------------------------------------------------------------------
# Versioned vector store
# ---------------------------------------------------------------------------

class VersionedDocumentStore:
    """
    Wraps Qdrant with change-aware indexing:

      index_document(doc_id, text) →
        "skipped"  if content unchanged
        "indexed"  if new document
        "updated"  if content changed (old chunks deleted, new ones inserted)

    When keep_old_versions=True the old chunks stay in Qdrant with
    is_current=False so you can still query them with search(..., current_only=False).
    """

    def __init__(
        self,
        collection_name: str = "versioned_docs",
        registry_path: str = "doc_registry.json",
        keep_old_versions: bool = False,
        qdrant_location: str = ":memory:",
        embedding_cache: EmbeddingCache | None = None,
    ):
        self.collection = collection_name
        self.keep_old_versions = keep_old_versions
        self.registry = DocumentRegistry(registry_path)
        self.openai = OpenAI()
        self._cache = embedding_cache  # None = no caching

        self.qdrant = QdrantClient(qdrant_location)
        existing = {c.name for c in self.qdrant.get_collections().collections}
        if collection_name not in existing:
            self.qdrant.create_collection(
                collection_name=collection_name,
                vectors_config=VectorParams(size=1536, distance=Distance.COSINE),
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _chunk_text(text: str, chunk_size: int = 500, overlap: int = 100) -> list[str]:
        """Split text into overlapping chunks, preferring sentence boundaries."""
        if len(text) <= chunk_size:
            return [text]

        chunks: list[str] = []
        start = 0
        while start < len(text):
            end = min(start + chunk_size, len(text))
            if end < len(text):
                lookback = int(chunk_size * 0.2)
                for i in range(end, max(end - lookback, start), -1):
                    if text[i] in ".!?\n":
                        end = i + 1
                        break
            chunk = text[start:end].strip()
            if chunk:
                chunks.append(chunk)
            if end >= len(text):
                break
            start = end - overlap
        return chunks

    @staticmethod
    def _chunk_id(doc_id: str, version: int, chunk_idx: int) -> str:
        """
        Deterministic UUID from (doc_id, version, chunk_idx).
        Lets us reconstruct IDs for any version without querying Qdrant.
        """
        key = f"{doc_id}::v{version}::chunk{chunk_idx}"
        return str(uuid.uuid5(uuid.NAMESPACE_DNS, key))

    def _embed(self, texts: list[str]) -> list[list[float]]:
        if self._cache is not None:
            return self._cache.get_embeddings(texts)
        response = self.openai.embeddings.create(
            model="text-embedding-3-small",
            input=texts,
        )
        return [d.embedding for d in response.data]

    def _delete_chunks(self, chunk_ids: list[str]):
        if chunk_ids:
            self.qdrant.delete(
                collection_name=self.collection,
                points_selector=PointIdsList(points=chunk_ids),
            )

    def _mark_not_current(self, chunk_ids: list[str]):
        """Mark old chunks as archived without removing them."""
        if chunk_ids:
            self.qdrant.set_payload(
                collection_name=self.collection,
                payload={"is_current": False},
                points=PointIdsList(points=chunk_ids),
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def index_document(
        self,
        doc_id: str,
        text: str,
        metadata: dict | None = None,
        chunk_size: int = 500,
    ) -> str:
        """
        Index or re-index a document.

        Returns 'skipped', 'indexed', or 'updated'.
        """
        meta: dict = metadata or {}
        new_hash = DocumentRegistry.hash_content(text)
        record = self.registry.get(doc_id)

        # --- Change detection ---
        if record and record.current_hash == new_hash:
            print(f"  [SKIPPED] '{doc_id}' — content unchanged (v{record.current_version})")
            return "skipped"

        new_version = (record.current_version + 1) if record else 1
        status = "updated" if record else "indexed"

        # --- Handle old chunks ---
        if record:
            if self.keep_old_versions:
                # Archive: keep in Qdrant but flag as non-current
                self._mark_not_current(record.current_chunk_ids)
                print(f"  Archived {len(record.current_chunk_ids)} chunks from v{record.current_version}")
            else:
                # Purge: delete old chunks entirely
                self._delete_chunks(record.current_chunk_ids)
                print(f"  Removed {len(record.current_chunk_ids)} stale chunks from v{record.current_version}")

        # --- Chunk and embed ---
        chunks = self._chunk_text(text, chunk_size=chunk_size)
        embeddings = self._embed(chunks)

        new_chunk_ids: list[str] = []
        points: list[PointStruct] = []
        for i, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cid = self._chunk_id(doc_id, new_version, i)
            new_chunk_ids.append(cid)
            points.append(PointStruct(
                id=cid,
                vector=emb,
                payload={
                    "text": chunk,
                    "doc_id": doc_id,
                    "version": new_version,
                    "chunk_index": i,
                    "is_current": True,
                    **meta,
                },
            ))

        self.qdrant.upsert(collection_name=self.collection, points=points)

        # --- Update registry ---
        history = record.version_history[:] if record else []
        if record and self.keep_old_versions:
            history.append(asdict(VersionSnapshot(
                version=record.current_version,
                content_hash=record.current_hash,
                chunk_ids=record.current_chunk_ids,
                indexed_at=record.indexed_at,
            )))

        self.registry.put(DocRecord(
            doc_id=doc_id,
            current_version=new_version,
            current_hash=new_hash,
            current_chunk_ids=new_chunk_ids,
            indexed_at=datetime.now(timezone.utc).isoformat(),
            metadata=meta,
            version_history=history,
        ))

        print(f"  [{status.upper()}] '{doc_id}' → v{new_version} ({len(chunks)} chunks)")
        return status

    def delete_document(self, doc_id: str, version: int | None = None):
        """
        Delete a document entirely, or one specific historical version.

        - version=None  → delete current + all history, remove from registry
        - version=N     → delete that version's chunks only (history must exist)
        """
        record = self.registry.get(doc_id)
        if not record:
            print(f"  No record for '{doc_id}'")
            return

        if version is None:
            # Wipe everything
            all_ids = list(record.current_chunk_ids)
            for snap in record.version_history:
                all_ids.extend(snap["chunk_ids"])
            self._delete_chunks(all_ids)
            self.registry.delete(doc_id)
            print(f"  Deleted '{doc_id}' (all versions, {len(all_ids)} chunks)")

        elif version == record.current_version:
            # Demote the latest archived version back to current, or just remove
            self._delete_chunks(record.current_chunk_ids)
            if record.version_history:
                prev = record.version_history[-1]
                record.current_version = prev["version"]
                record.current_hash = prev["content_hash"]
                record.current_chunk_ids = prev["chunk_ids"]
                record.indexed_at = prev["indexed_at"]
                record.version_history = record.version_history[:-1]
                # Re-mark the restored version as current in Qdrant
                if record.current_chunk_ids:
                    self.qdrant.set_payload(
                        collection_name=self.collection,
                        payload={"is_current": True},
                        points=PointIdsList(points=record.current_chunk_ids),
                    )
                self.registry.put(record)
                print(f"  Deleted current v{version}, rolled back to v{record.current_version}")
            else:
                self.registry.delete(doc_id)
                print(f"  Deleted '{doc_id}' v{version} (no history to restore)")

        else:
            # Delete a specific old version from history
            match = next((s for s in record.version_history if s["version"] == version), None)
            if not match:
                print(f"  No v{version} found in history for '{doc_id}'")
                return
            self._delete_chunks(match["chunk_ids"])
            record.version_history = [s for s in record.version_history if s["version"] != version]
            self.registry.put(record)
            print(f"  Deleted historical v{version} for '{doc_id}' ({len(match['chunk_ids'])} chunks)")

    def search(
        self,
        query: str,
        top_k: int = 5,
        current_only: bool = True,
        doc_id_filter: str | None = None,
        version_filter: int | None = None,
    ) -> list[dict]:
        """
        Semantic search with optional metadata filters.

        current_only=True  → excludes archived chunks (default)
        doc_id_filter      → restrict to a specific document
        version_filter     → restrict to a specific version
        """
        query_emb = self._embed([query])[0]

        must: list[FieldCondition] = []
        if current_only and self.keep_old_versions:
            must.append(FieldCondition(key="is_current", match=MatchValue(value=True)))
        if doc_id_filter:
            must.append(FieldCondition(key="doc_id", match=MatchValue(value=doc_id_filter)))
        if version_filter is not None:
            must.append(FieldCondition(key="version", match=MatchValue(value=version_filter)))

        results = self.qdrant.query_points(
            collection_name=self.collection,
            query=query_emb,  # type: ignore[arg-type]
            query_filter=Filter(must=must) if must else None,
            limit=top_k,
        ).points

        return [
            {
                "text": payload["text"],
                "score": round(hit.score, 4),
                "doc_id": payload["doc_id"],
                "version": payload["version"],
                "chunk_index": payload["chunk_index"],
                "is_current": payload.get("is_current", True),
            }
            for hit in results
            if (payload := hit.payload or {})
        ]

    def document_status(self, doc_id: str) -> Optional[dict]:
        """Summary of a document's current state and version history."""
        record = self.registry.get(doc_id)
        if not record:
            return None
        return {
            "doc_id": record.doc_id,
            "current_version": record.current_version,
            "content_hash": record.current_hash[:12] + "...",
            "current_chunks": len(record.current_chunk_ids),
            "indexed_at": record.indexed_at,
            "historical_versions": [s["version"] for s in record.version_history],
            "metadata": record.metadata,
        }

    def list_documents(self) -> list[dict]:
        """Status of every tracked document."""
        result: list[dict] = []
        for rec in self.registry.all():
            status = self.document_status(rec.doc_id)
            if status is not None:
                result.append(status)
        return result


# ---------------------------------------------------------------------------
# Demo
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    store = VersionedDocumentStore(
        collection_name="demo",
        registry_path="doc_registry.json",
        keep_old_versions=True,   # flip to False to purge instead of archive
        qdrant_location=":memory:",
    )

    # --- v1: index original documents ---
    # print("\n=== Initial indexing ===")
    # status1 = store.index_document(
    #     "python-guide",
    #     "Python is a high-level language known for readability and simplicity. "
    #     "It supports procedural, object-oriented, and functional programming.",
    #     metadata={"source": "guide", "author": "team"},
    # )
    # print(f'status1: {status1}')
    # status2 = store.index_document(
    #     "ml-intro",
    #     "Machine learning allows computers to learn from data without being explicitly programmed. "
    #     "Common frameworks include TensorFlow and PyTorch.",
    # )
    # print(f'status2: {status2}')
    #
    # # --- No change: should be skipped ---
    # print("\n=== Re-index unchanged document ===")
    # status3 = store.index_document(
    #     "python-guide",
    #     "Python is a high-level language known for readability and simplicity. "
    #     "It supports procedural, object-oriented, and functional programming.",
    # )
    # print(f'status3: {status3}')
    #
    # # --- v2: update one document ---
    # print("\n=== Update python-guide ===")
    # status4 = store.index_document(
    #     "python-guide",
    #     "Python is a high-level, dynamically typed language. "
    #     "It is widely used in data science, web development, and automation. "
    #     "The ecosystem includes powerful libraries like NumPy, Pandas, and FastAPI.",
    #     metadata={"source": "guide", "author": "team", "revision": 2},
    # )
    # print(f'status4: {status4}')

    # --- Status ---
    print("\n=== Document status ===")
    for doc in store.list_documents():
        print(f"  {doc}")

    # --- Search current version only ---
    print("\n=== Search (current only) ===")
    for r in store.search("data science libraries", top_k=3):
        print(f"  [{r['score']}] v{r['version']} {r['doc_id']}: {r['text'][:80]}")

    # --- Search all versions (current + archived) ---
    print("\n=== Search (all versions) ===")
    for r in store.search("readability and simplicity", top_k=3, current_only=False):
        marker = "current" if r["is_current"] else "archived"
        print(f"  [{r['score']}] v{r['version']} ({marker}) {r['doc_id']}: {r['text'][:80]}")

    # --- Delete a specific historical version ---
    print("\n=== Delete historical v1 of python-guide ===")
    store.delete_document("python-guide", version=1)
    print(f"  Status after: {store.document_status('python-guide')}")

    # --- Full delete ---
    # print("\n=== Delete ml-intro entirely ===")
    # store.delete_document("ml-intro")
    # print(f"  Remaining docs: {[d['doc_id'] for d in store.list_documents()]}")
