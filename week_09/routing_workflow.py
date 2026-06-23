# routing_workflow.py
from PIL.Image import Image
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
import json
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-5-mini")

class SupportState(TypedDict):
    message: str
    category: str
    response: str

def classify(state: SupportState) -> dict:
    """Classify the incoming message."""
    result = llm.invoke(
        f"""Classify this support message into one category: billing, technical, general

Message: {state['message']}

Respond with JSON: {{"category": "billing|technical|general"}}"""
    )
    data = json.loads(result.content)
    return {"category": data["category"]}

def handle_billing(state: SupportState) -> dict:
    response = llm.invoke(
        f"You are a billing specialist. Answer this: {state['message']}"
    )
    return {"response": response.content}

def handle_technical(state: SupportState) -> dict:
    response = llm.invoke(
        f"You are a technical support engineer. Answer this: {state['message']}"
    )
    return {"response": response.content}

def handle_general(state: SupportState) -> dict:
    response = llm.invoke(f"Answer this customer message helpfully: {state['message']}")
    return {"response": response.content}

def route(state: SupportState) -> Literal["billing", "technical", "general"]:
    return state["category"]

graph = StateGraph(SupportState)
graph.add_node("classify", classify)
graph.add_node("billing", handle_billing)
graph.add_node("technical", handle_technical)
graph.add_node("general", handle_general)

graph.add_edge(START, "classify")
graph.add_conditional_edges("classify", route)
graph.add_edge("billing", END)
graph.add_edge("technical", END)
graph.add_edge("general", END)

app = graph.compile()
# png_data = app.get_graph().draw_mermaid_png()
# with open("routing_workflow.png", "wb") as f:
#     f.write(png_data)

if __name__ == "__main__":
    messages = [
        # "My invoice is wrong, I was charged twice",
        # "The app crashes when I try to upload files",
        "What are your business hours?",
    ]
    for msg in messages:
        result = app.invoke({"message": msg})
        print(f"\n[{result['category'].upper()}] {msg}")
        print(f"→ {result['response'][:200]}...")
