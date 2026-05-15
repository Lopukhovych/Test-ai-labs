# chunker.py
from typing import List
from dataclasses import dataclass

@dataclass
class Chunk:
    text: str
    start_index: int
    end_index: int
    metadata: dict = None

def chunk_text(
        text: str,
        chunk_size: int = 500,
        chunk_overlap: int = 100
) -> List[Chunk]:
    """Split text into overlapping chunks."""

    if len(text) <= chunk_size:
        return [Chunk(text=text, start_index=0, end_index=len(text))]

    chunks = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        # Try to end at a sentence boundary
        if end < len(text):
            # Look for period, newline in last 20% of chunk
            lookback = int(chunk_size * 0.2)
            for i in range(end, end - lookback, -1):
                if text[i] in '.!?\n':
                    end = i + 1
                    break

        chunk_text_str = text[start:end].strip()
        if chunk_text_str:
            chunks.append(Chunk(
                text=chunk_text_str,
                start_index=start,
                end_index=end
            ))

        if end >= len(text):
            break

        start = end - chunk_overlap

    return chunks

# Test
if __name__ == "__main__":
    test_text = """
    Python is a high-level programming language known for its readability. 
    It was created by Guido van Rossum and first released in 1991. Python supports 
    multiple programming paradigms, including procedural, object-oriented, and functional.
    
    The language emphasizes code readability with its notable use of significant indentation. 
    Its language constructs aim to help programmers write clear, logical code.
    """

    chunks = chunk_text(test_text, chunk_size=200, chunk_overlap=50)

    print(f"Original length: {len(test_text)} chars")
    print(f"Number of chunks: {len(chunks)}\n")

    for i, chunk in enumerate(chunks):
        print(f"Chunk {i+1}: '{chunk.text[:60]}...'\n")
