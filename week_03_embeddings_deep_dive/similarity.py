# similarity.py
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np

load_dotenv()
client = OpenAI()

def get_embedding(text: str) -> list[float]:
    """Get embedding for text."""
    response = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return response.data[0].embedding

def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Calculate cosine similarity between two vectors."""
    a = np.array(vec1)
    b = np.array(vec2)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Test with different texts
texts = [
    "I love programming in Python",
    "Python is my favorite language for coding",
    "I enjoy writing code in Python",
    "The weather is beautiful today",
    "I hate programming"
]

# Get embedding for the first text
base_text = texts[0]
base_embedding = get_embedding(base_text)

print(f"Comparing everything to: '{base_text}'\n")

for text in texts[1:]:
    embedding = get_embedding(text)
    similarity = cosine_similarity(base_embedding, embedding)
    print(f"Similarity: {similarity:.4f} | '{text}'")
