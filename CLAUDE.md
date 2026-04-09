# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a structured AI/ML learning project organized by topic/week. The codebase demonstrates practical applications of the OpenAI API, focusing on:
- **Week 01** (`week_01_fundamentals`): Core LLM interactions (chat, structured outputs, prompt engineering, retry logic)
- **Week 02** (`week_02_rag_introduction`): RAG introduction (minimal content currently)
- **Week 03** (`week_03_embeddings_deep_dive`): Embeddings and semantic search (text-embedding-3-small, RAG pattern, similarity matching)
- **Week 05** (`week_05_evaluations`): LLM evaluation frameworks (faithfulness, regression testing, RAG evaluation)

## Common Development Commands

```bash
# Run a specific script
python week_03_embeddings_deep_dive/first_embedding.py

# Run any Python file
python path/to/file.py

# Install dependencies
uv sync

# Run evaluations
python week_05_evaluations/run_evaluation.py
```

## Architecture & Patterns

### Environment Setup
- All scripts use `from dotenv import load_dotenv` and `load_dotenv()` at module start
- OpenAI API key is stored in `.env` file (never commit this)
- OpenAI client is initialized at module level: `client = OpenAI()`

### Common Patterns

**Chatbot Pattern** (week_01/chatbot_class.py):
```python
class Chatbot:
    def __init__(self, system_prompt):
        self.client = OpenAI()
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, user_message) -> str:
        # Append user message, call API, append response, return
```
- Maintains message history automatically
- System prompt stays constant across conversation
- Use `clear_history()` to reset while keeping system prompt

**Embeddings Pattern** (week_03):
- Use `text-embedding-3-small` model for embeddings
- Store embeddings in dataclasses with the original content
- Calculate similarity using cosine: `np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))`
- Document structure: `@dataclass` with `content`, `embedding`, optionally `filename`

**Test Code**:
- Test/example code lives at the bottom of each file under `if __name__ == "__main__"`
- This allows files to be both importable modules and executable scripts

### Data Structures
- Use `@dataclass` from `dataclasses` for simple data containers (Document, etc.)
- Use `typing` hints for function signatures and class attributes
- Type-annotated return types: `List[float]`, `tuple`, etc.

### Dependencies
- `numpy`: Vector operations, similarity calculations
- `openai`: Chat completions, embeddings API
- `python-dotenv`: Load environment variables
- `pydantic`: Data validation (week_01_fundamentals examples, not in pyproject.toml — install separately if needed)

## Key Implementation Details

**API Model Names**:
- Chat completions: `gpt-4-mini` (note: examples may use older names, check current availability)
- Embeddings: `text-embedding-3-small`

**Semantic Search & RAG**:
- Week 03 implements a simple RAG pattern: load documents → embed → search by similarity
- Documents are embedded once at load time, queries are embedded at search time
- Top-k results are returned sorted by similarity score
- No database—documents are kept in memory in Python objects

**Common Imports Across Files**:
```python
from openai import OpenAI
from dotenv import load_dotenv
import numpy as np
from typing import List
from dataclasses import dataclass
```

## File Organization

```
week_01/  - LLM interactions & prompting
  ├── first_call.py              # Basic API call
  ├── chatbot_class.py           # Reusable Chatbot class
  ├── cli_chat.py                # Interactive CLI chat
  ├── chain_of_thought.py        # CoT prompting
  ├── few_shot.py                # Few-shot examples
  ├── structured_chatbot.py      # Pydantic validation
  ├── json_basics.py             # JSON mode output
  ├── pydantic_validation.py     # Structured outputs
  ├── retry_logic.py             # Handling API failures
  ├── save_conversations.py      # Persist conversation history
  ├── summarize.py               # Text summarization
  ├── personalities.py           # Custom system prompts
  └── human_in_loop.py           # Human feedback in loop

week_03/  - Embeddings & semantic search
  ├── first_embedding.py         # Basic embedding call
  ├── similarity.py              # Cosine similarity examples
  ├── compare_models.py          # Compare embedding models
  ├── semantic_search.py         # Simple semantic search
  ├── semantic_rag.py            # RAG from documents
  ├── batch_embeddings.py        # Efficient batch embedding
  ├── challenges.py              # Practice exercises
  └── docs/                      # Sample documents for RAG
```

## Notes for Future Development

- **API costs**: Embeddings and chat completions incur costs. Be mindful of repeated calls during development.
- **Rate limiting**: OpenAI API has rate limits. Week 03's batch_embeddings.py implements efficient batching.
- **Example code**: All scripts can be run directly and contain working examples; modify the test code at the bottom to experiment.
- **Import paths**: Files in week_01 can be imported by week_03 (e.g., `from week_01.chatbot_class import Chatbot`) since week_01 contains reusable classes.
- **Environment**: Ensure `.env` is in the root directory and contains a valid `OPENAI_API_KEY`.