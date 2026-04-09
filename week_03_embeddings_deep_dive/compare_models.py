# compare_models.py
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
import time

load_dotenv()
client = OpenAI()

def get_embedding(text: str, model: str) -> list:
    response = client.embeddings.create(model=model, input=text)
    return response.data[0].embedding

def cosine_similarity(a, b):
    a, b = np.array(a), np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))

# Test pairs
test_pairs = [
    ("I love pizza", "Pizza is my favorite food"),
    ("I love pizza", "I hate pizza"),
    ("The stock market crashed", "Financial markets declined sharply"),
    ("A cat sleeps on the couch", "The weather is sunny today")
]

models = ["text-embedding-3-small", "text-embedding-3-large"]

for model in models:
    print(f"\n{'='*50}")
    print(f"Model: {model}")
    print("="*50)

    start = time.time()

    for text1, text2 in test_pairs:
        emb1 = get_embedding(text1, model)
        emb2 = get_embedding(text2, model)
        sim = cosine_similarity(emb1, emb2)
        print(f"{sim:.4f} | '{text1}' vs '{text2}'")

    elapsed = time.time() - start
    print(f"\nTime: {elapsed:.2f}s | Dimensions: {len(emb1)}")
