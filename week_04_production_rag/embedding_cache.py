"""
embedding_cache.py — Persistent, batch-aware embedding cache with savings metrics.

Design:
  - SHA-256 key (stable across processes, unlike Python's hash())
  - Batch-aware: only cache-miss texts are sent to OpenAI in a single batch call
  - Disk persistence via JSON so the cache survives restarts
  - Stats object tracks hits, misses, API calls, tokens and cost saved
"""

import hashlib
import json
import time
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# text-embedding-3-small pricing (April 2026)
_PRICE_PER_TOKEN = 0.02 / 1_000_000   # $0.02 per 1M tokens
_CHARS_PER_TOKEN = 4                   # rough approximation


# ---------------------------------------------------------------------------
# Stats
# ---------------------------------------------------------------------------

@dataclass
class CacheStats:
    """Running totals across all get_embeddings() calls."""
    texts_requested: int = 0   # individual texts requested
    cache_hits: int = 0        # texts served from cache
    cache_misses: int = 0      # texts that hit the API
    api_calls: int = 0         # actual batched OpenAI calls made
    api_latency_ms: float = 0  # real time spent waiting for OpenAI
    tokens_saved: int = 0      # estimated tokens NOT sent to API

    @property
    def hit_rate(self) -> float:
        return self.cache_hits / self.texts_requested if self.texts_requested else 0.0

    @property
    def cost_saved(self) -> float:
        """Estimated USD saved by not re-embedding cached texts."""
        return self.tokens_saved * _PRICE_PER_TOKEN

    @property
    def latency_saved_ms(self) -> float:
        """
        Estimated latency saved. We scale actual API latency by the ratio of
        cache hits to total API texts — a hit would have cost roughly the same
        time per text as a miss did.
        """
        if self.cache_misses == 0:
            return 0.0
        ms_per_text = self.api_latency_ms / self.cache_misses
        return ms_per_text * self.cache_hits

    def summary(self) -> str:
        total = self.cache_hits + self.cache_misses
        return (
            f"  Hit rate:      {self.hit_rate:.1%}  "
            f"({self.cache_hits} hits, {self.cache_misses} misses, {total} total texts)\n"
            f"  API calls:     {self.api_calls}  "
            f"(vs {self.texts_requested} get_embeddings() calls)\n"
            f"  Tokens saved:  ~{self.tokens_saved:,}\n"
            f"  Cost saved:    ~${self.cost_saved:.6f}\n"
            f"  Latency saved: ~{self.latency_saved_ms:.0f} ms"
        )

    def reset(self):
        """Reset counters — useful for before/after comparisons."""
        self.texts_requested = 0
        self.cache_hits = 0
        self.cache_misses = 0
        self.api_calls = 0
        self.api_latency_ms = 0
        self.tokens_saved = 0


# ---------------------------------------------------------------------------
# Cache
# ---------------------------------------------------------------------------

class EmbeddingCache:
    """
    Drop-in replacement for a bare OpenAI embeddings.create() call.

    Usage:
        cache = EmbeddingCache()
        embeddings = cache.get_embeddings(["text a", "text b"])
        embedding  = cache.get_embedding("text a")   # convenience wrapper
    """

    def __init__(
        self,
        model: str = "text-embedding-3-small",
        cache_path: str | None = "embedding_cache.json",
    ):
        self.model = model
        self.openai = OpenAI()
        self.stats = CacheStats()
        self._cache: dict[str, list[float]] = {}
        self._path = Path(cache_path) if cache_path else None
        self._load()

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    def _load(self):
        if self._path and self._path.exists():
            data = json.loads(self._path.read_text())
            self._cache = data.get("embeddings", {})
            print(f"[cache] Loaded {len(self._cache)} cached embeddings from {self._path}")

    def _save(self):
        if self._path:
            # compact JSON — embeddings are large, no need for pretty-printing
            self._path.write_text(
                json.dumps({"embeddings": self._cache}, separators=(",", ":"))
            )

    # ------------------------------------------------------------------
    # Key
    # ------------------------------------------------------------------

    @staticmethod
    def _key(text: str) -> str:
        """SHA-256 of the text — deterministic and collision-resistant."""
        return hashlib.sha256(text.encode()).hexdigest()

    # ------------------------------------------------------------------
    # Core API
    # ------------------------------------------------------------------

    def get_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Return embeddings for each text in order.

        Cache hits are returned immediately; only misses are batched into a
        single OpenAI API call.
        """
        self.stats.texts_requested += len(texts)

        keys = [self._key(t) for t in texts]
        results: list[list[float] | None] = [None] * len(texts)

        miss_indices: list[int] = []
        miss_texts: list[str] = []

        for i, (key, text) in enumerate(zip(keys, texts)):
            if key in self._cache:
                results[i] = self._cache[key]
                self.stats.cache_hits += 1
                self.stats.tokens_saved += len(text) // _CHARS_PER_TOKEN
            else:
                miss_indices.append(i)
                miss_texts.append(text)
                self.stats.cache_misses += 1

        # One batched API call for all misses
        if miss_texts:
            t0 = time.perf_counter()
            response = self.openai.embeddings.create(
                model=self.model,
                input=miss_texts,
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            self.stats.api_calls += 1
            self.stats.api_latency_ms += elapsed_ms

            for j, data in enumerate(response.data):
                idx = miss_indices[j]
                emb = data.embedding
                results[idx] = emb
                self._cache[keys[idx]] = emb

            self._save()

        return results  # type: ignore[return-value]  # every slot is filled above

    def get_embedding(self, text: str) -> list[float]:
        """Convenience wrapper for a single text."""
        return self.get_embeddings([text])[0]

    # ------------------------------------------------------------------
    # Cache management
    # ------------------------------------------------------------------

    def invalidate(self, text: str):
        """Remove a single entry — call after a document is updated."""
        key = self._key(text)
        if key in self._cache:
            del self._cache[key]
            self._save()

    def invalidate_many(self, texts: list[str]):
        for text in texts:
            key = self._key(text)
            self._cache.pop(key, None)
        self._save()

    def clear(self):
        """Wipe the entire cache."""
        self._cache.clear()
        self._save()
        print("[cache] Cleared.")

    @property
    def size(self) -> int:
        return len(self._cache)


# ---------------------------------------------------------------------------
# Integration helper for VersionedDocumentStore
# ---------------------------------------------------------------------------

def patch_store_with_cache(store, cache: EmbeddingCache):
    """
    Replace the _embed method on a VersionedDocumentStore instance so it
    uses the cache transparently — no subclassing needed.

        store = VersionedDocumentStore(...)
        cache = EmbeddingCache()
        patch_store_with_cache(store, cache)
    """
    def _cached_embed(texts: list[str]) -> list[list[float]]:
        return cache.get_embeddings(texts)

    import types
    store._embed = types.MethodType(lambda self, texts: _cached_embed(texts), store)


# ---------------------------------------------------------------------------
# Demo — before / after comparison
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    from metadata_search import VersionedDocumentStore

    # Chunk boundary falls around char ~403 (at the period after "prediction).").
    # Chunk 0 = chars   0–403  →  identical in UPDATED_ML  → cache HIT
    # Chunk 1 = chars 303–end  →  differs after char 403   → cache MISS
    _ML_CHUNK0 = (                                          # ~403 chars, shared by both versions
        "Machine learning is a subset of artificial intelligence that enables systems to learn "
        "from data and improve their performance over time without being explicitly programmed. "
        "The field is broadly divided into three paradigms. "
        "Supervised learning trains models on labeled input-output pairs; common tasks include "
        "classification (spam detection, image recognition) and regression (house price prediction). "
    )
    _ML_ORIGINAL_TAIL = (                                   # changed in the update
        "Unsupervised learning finds hidden structure in unlabeled data through clustering "
        "and dimensionality reduction techniques such as k-means and PCA. "
        "Reinforcement learning trains agents to take actions in an environment to maximize "
        "cumulative reward; it powers game-playing systems like AlphaGo and robotics control. "
        "Popular frameworks include scikit-learn for classical ML, TensorFlow and PyTorch for "
        "deep learning, and XGBoost for gradient boosting. Model evaluation uses metrics "
        "such as accuracy, precision, recall, F1-score, and AUC-ROC depending on the task."
    )
    _ML_UPDATED_TAIL = (                                    # new tail — different chunks
        "Unsupervised learning finds patterns in unlabeled data via clustering and PCA. "
        "Reinforcement learning maximizes cumulative reward through trial-and-error; "
        "it underpins AlphaGo, RLHF fine-tuning in ChatGPT, and autonomous vehicle systems. "
        "Key libraries include scikit-learn, PyTorch, TensorFlow, and Hugging Face Transformers. "
        "Models are evaluated with accuracy, F1-score, RMSE, and AUC-ROC."
    )

    DOCUMENTS = {
        "python-intro": (
            "Python is a high-level, dynamically typed language created by Guido van Rossum. "
            "It emphasizes readability and simplicity. Python supports multiple paradigms: "
            "procedural, object-oriented, and functional. The standard library is vast."
        ),
        "ml-basics": _ML_CHUNK0 + _ML_ORIGINAL_TAIL,
        "rag-overview": (
            "Retrieval-Augmented Generation combines a retrieval step with a language model. "
            "Documents are embedded and stored in a vector database. At query time, "
            "relevant chunks are retrieved and injected into the prompt as context."
        ),
        "vector-dbs": (
            "Vector databases store high-dimensional embeddings and support approximate nearest "
            "neighbor search. Popular options include Pinecone, Qdrant, Weaviate, and Chroma. "
            "They enable semantic search at scale without scanning every document."
        ),
        "transformers": (
            "Transformer models use self-attention to process sequences in parallel. "
            "BERT is encoder-only, GPT is decoder-only, T5 is encoder-decoder. "
            "Pre-training on large corpora gives strong transfer learning performance."
        ),
    }

    # Only the tail changes — chunk 0 is identical so it will be a cache hit
    UPDATED_ML = _ML_CHUNK0 + _ML_UPDATED_TAIL

    cache = EmbeddingCache(cache_path="embedding_cache.json")
    cache.clear()   # always start clean so Round 1 is a true cold start

    # Verify ml-basics splits into 2+ chunks before running the demo
    ml_chunks = VersionedDocumentStore._chunk_text(DOCUMENTS["ml-basics"])
    print(f"\nml-basics: {len(DOCUMENTS['ml-basics'])} chars → {len(ml_chunks)} chunks")
    for i, c in enumerate(ml_chunks):
        print(f"  chunk {i}: {len(c)} chars  '{c[:60]}...'")

    # ------------------------------------------------------------------
    # ROUND 1 — cold start, cache is empty (or pre-populated from disk)
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROUND 1 — First indexing (cold start)")
    print("=" * 60)

    cache.stats.reset()
    store = VersionedDocumentStore(
        collection_name="cache-demo",
        registry_path="cache_demo_registry.json",
        keep_old_versions=True,
        qdrant_location=":memory:",
    )
    patch_store_with_cache(store, cache)

    for doc_id, text in DOCUMENTS.items():
        store.index_document(doc_id, text)

    print(f"\nRound 1 stats (cache had {cache.size - cache.stats.cache_misses} entries before):")
    print(cache.stats.summary())

    # ------------------------------------------------------------------
    # ROUND 2 — same documents, same content → all chunks hit cache
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROUND 2 — Re-index identical content (warm cache)")
    print("=" * 60)

    cache.stats.reset()
    store2 = VersionedDocumentStore(
        collection_name="cache-demo-2",
        registry_path="cache_demo_registry2.json",
        keep_old_versions=True,
        qdrant_location=":memory:",
    )
    patch_store_with_cache(store2, cache)

    for doc_id, text in DOCUMENTS.items():
        store2.index_document(doc_id, text)

    print(f"\nRound 2 stats (warm cache, {cache.size} entries):")
    print(cache.stats.summary())

    # ------------------------------------------------------------------
    # ROUND 3 — one document updated → only changed chunks miss cache
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROUND 3 — One document updated (partial cache hit)")
    print("=" * 60)

    cache.stats.reset()
    store2.index_document("ml-basics", UPDATED_ML)  # only this doc changes

    print("\nRound 3 stats (one document updated):")
    print(cache.stats.summary())

    # ------------------------------------------------------------------
    # ROUND 4 — simulate a server restart: new store, same cache on disk
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("ROUND 4 — Simulate restart (cache reloaded from disk)")
    print("=" * 60)

    cache2 = EmbeddingCache(cache_path="embedding_cache.json")
    cache2.stats.reset()

    store3 = VersionedDocumentStore(
        collection_name="cache-demo-3",
        registry_path="cache_demo_registry3.json",
        qdrant_location=":memory:",
    )
    patch_store_with_cache(store3, cache2)

    for doc_id, text in DOCUMENTS.items():
        store3.index_document(doc_id, text)

    print(f"\nRound 4 stats (new process, cache from disk):")
    print(cache2.stats.summary())

    # ------------------------------------------------------------------
    # Summary table
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"  Cache entries on disk: {cache2.size}")
    total_texts = len(DOCUMENTS) * 2 + 1   # rounds 1+2+update
    print(f"  Total texts embedded across rounds 1-3: ~{total_texts}")
    print(f"  Without cache: every round would call the API")
    print(f"  With cache:    only novel texts call the API")

    # Clean up demo registry files
    import os
    for f in ["cache_demo_registry.json", "cache_demo_registry2.json", "cache_demo_registry3.json"]:
        if os.path.exists(f):
            os.remove(f)
