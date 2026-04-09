# batch_embeddings.py
from openai import OpenAI
from dotenv import load_dotenv
from typing import List
import time

load_dotenv()
client = OpenAI()

def get_embeddings_batch(texts: List[str]) -> List[List[float]]:
    """Get embeddings for multiple texts in one API call."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=texts
    )
    return [item.embedding for item in response.data]

# Compare single vs batch
documents = [
    "Python is a programming language",
    "Machine learning uses algorithms",
    "Neural networks are inspired by brains",
    "Data science analyzes large datasets",
    "Natural language processing handles text"
]

# Batch method (faster!)
print("Batch method:")
start = time.time()
embeddings = get_embeddings_batch(documents)
batch_time = time.time() - start
print(f"  Time: {batch_time:.3f}s for {len(documents)} documents")
print(f"  Got {len(embeddings)} embeddings of dimension {len(embeddings[0])}")

# Single method (slower)
print("\nSingle method:")
start = time.time()
single_embeddings = []
for doc in documents:
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc
    )
    single_embeddings.append(response.data[0].embedding)
single_time = time.time() - start
print(f"  Time: {single_time:.3f}s for {len(documents)} documents")

print(f"\nBatch is {single_time/batch_time:.1f}x faster!")
