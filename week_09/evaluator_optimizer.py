# evaluator_optimizer.py
from langgraph.graph import StateGraph, START, END
from langchain_openai import ChatOpenAI
from typing import TypedDict, Literal
import json
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-5-mini")

class OptimizeState(TypedDict):
    task: str
    draft: str
    feedback: str
    score: int
    iterations: int

def generate(state: OptimizeState) -> dict:
    """Generate or improve the draft."""
    if state.get("feedback"):
        prompt = f"""Improve this draft based on the feedback:

Draft: {state['draft']}
Feedback: {state['feedback']}

Write an improved version."""
    else:
        prompt = f"Write a response to: {state['task']}"

    response = llm.invoke(prompt)
    return {"draft": response.content, "iterations": state.get("iterations", 0) + 1}

def evaluate(state: OptimizeState) -> dict:
    """Score the draft 1-10."""
    result = llm.invoke(
        f"""Evaluate this response to the task: "{state['task']}"

Response: {state['draft']}

Rate it 1-10 and give feedback for improvement.
Return JSON: {{"score": 7, "feedback": "..."}}"""
    )
    data = json.loads(result.content)
    return {"score": data["score"], "feedback": data["feedback"]}

def should_continue(state: OptimizeState) -> Literal["generate", "__end__"]:
    """Continue improving if score < 8 and we have budget."""
    if state["score"] >= 8 or state.get("iterations", 0) >= 3:
        return "__end__"
    return "generate"

graph = StateGraph(OptimizeState)
graph.add_node("generate", generate)
graph.add_node("evaluate", evaluate)
graph.add_edge(START, "generate")
graph.add_edge("generate", "evaluate")
graph.add_conditional_edges("evaluate", should_continue)
graph.add_edge("evaluate", END)

app = graph.compile()
# png_data = app.get_graph().draw_mermaid_png()
# with open("evaluator_optimizer.png", "wb") as f:
#     f.write(png_data)

if __name__ == "__main__":
    result = app.invoke({"task": "Explain why recursion is powerful in programming"})
    print(f"Final score: {result['score']}/10 after {result['iterations']} iteration(s)")
    print(f"\n{result['draft']}")
