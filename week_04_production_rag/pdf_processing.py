# pdf_processing.py
from pypdf import PdfReader
from typing import List
from dataclasses import dataclass
import re

@dataclass
class DocumentChunk:
    text: str
    source: str
    page: int
    chunk_index: int

def extract_pdf(path: str) -> List[dict]:
    """Extract text from PDF with page tracking."""
    reader = PdfReader(path)
    pages = []

    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        # Clean up common PDF extraction issues
        text = re.sub(r'\s+', ' ', text)
        text = text.strip()

        if text:
            pages.append({
                "text": text,
                "page": i + 1,
                "source": path
            })

    return pages

def chunk_pdf(path: str, chunk_size: int = 500, overlap: int = 100) -> List[DocumentChunk]:
    """Extract and chunk a PDF file."""
    pages = extract_pdf(path)
    chunks = []
    chunk_idx = 0

    for page_data in pages:
        text = page_data["text"]
        start = 0

        while start < len(text):
            end = min(start + chunk_size, len(text))
            chunk_text = text[start:end].strip()

            if chunk_text:
                chunks.append(DocumentChunk(
                    text=chunk_text,
                    source=path,
                    page=page_data["page"],
                    chunk_index=chunk_idx
                ))
                chunk_idx += 1

            if end >= len(text):
                break
            start = end - overlap

    return chunks

# Test
if __name__ == "__main__":
    # uv add pypdf
    chunks = chunk_pdf("../week_04_production_rag/test_docs/ai_learning_program_short.pdf")
    print(f"Extracted {len(chunks)} chunks")
    for chunk in chunks:
        print(f"Page {chunk.page}, index {chunk.chunk_index}: {chunk.text[:80]}...")
