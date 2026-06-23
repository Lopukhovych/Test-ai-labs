# orchestrator_worker.py
from langgraph.graph import StateGraph, START, END
from langgraph.types import Send
from langchain_openai import ChatOpenAI
from typing import TypedDict, Annotated
import operator
import json
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-5-mini")

class MainState(TypedDict):
    topic: str
    subtasks: list[str]
    results: Annotated[list[str], operator.add]  # Merge results from workers

class WorkerState(TypedDict):
    subtask: str
    result: str

def orchestrate(state: MainState) -> dict:
    """Break topic into subtasks."""
    response = llm.invoke(
        f"""Break this topic into 3 research subtasks:
Topic: {state['topic']}
Return JSON: {{"subtasks": ["task1", "task2", "task3"]}}"""
    )
    data = json.loads(response.content)
    return {"subtasks": data["subtasks"]}

def dispatch_workers(state: MainState):
    """Dynamically create a worker for each subtask."""
    return [Send("worker", {"subtask": task}) for task in state["subtasks"]]

def worker(state: WorkerState) -> dict:
    """Each worker handles one subtask."""
    response = llm.invoke(f"Research this subtask briefly (2-3 sentences): {state['subtask']}")
    return {"results": [f"[{state['subtask'][:30]}] {response.content}"]}

def synthesize(state: MainState) -> dict:
    """Combine all worker results."""
    combined = "\n\n".join(state["results"])
    response = llm.invoke(
        f"Synthesize these research findings into a coherent summary:\n{combined}"
    )
    return {"results": [f"SYNTHESIS:\n{response.content}"]}

graph = StateGraph(MainState)
graph.add_node("orchestrate", orchestrate)
graph.add_node("worker", worker)
graph.add_node("synthesize", synthesize)

graph.add_edge(START, "orchestrate")
graph.add_conditional_edges("orchestrate", dispatch_workers, ["worker"])
graph.add_edge("worker", "synthesize")
graph.add_edge("synthesize", END)

app = graph.compile()

# png_data = app.get_graph().draw_mermaid_png()
# with open("orchestrator_worker.png", "wb") as f:
#     f.write(png_data)

if __name__ == "__main__":
    result = app.invoke({"topic": "Impact of AI on software engineering jobs"})
    for r in result["results"]:
        print(r)
        print()
