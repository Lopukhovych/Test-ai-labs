# chain_workflow.py
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-5-mini")

class ChainState(TypedDict):
    topic: str
    outline: str
    draft: str
    final: str

def create_outline(state: ChainState) -> dict:
    """Step 1: Create an outline."""
    response = llm.invoke(f"Create a 3-point outline for a blog post about: {state['topic']}")
    return {"outline": response.content}

def write_draft(state: ChainState) -> dict:
    """Step 2: Write a draft from the outline."""
    response = llm.invoke(
        f"Write a short blog post using this outline:\n{state['outline']}\n\nKeep it under 200 words."
    )
    return {"draft": response.content}

def polish_draft(state: ChainState) -> dict:
    """Step 3: Polish the draft."""
    response = llm.invoke(
        f"Improve this draft — fix grammar, improve flow, make it more engaging:\n{state['draft']}"
    )
    return {"final": response.content}

graph = StateGraph(ChainState)
graph.add_node("outline", create_outline)
graph.add_node("draft", write_draft)
graph.add_node("polish", polish_draft)
graph.add_edge(START, "outline")
graph.add_edge("outline", "draft")
graph.add_edge("draft", "polish")
graph.add_edge("polish", END)

app = graph.compile()

if __name__ == "__main__":
    result = app.invoke({"topic": "Why Python is great for AI"})
    print("=== FINAL POST ===")
    print(result["final"])
