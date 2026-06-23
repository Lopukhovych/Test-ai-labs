# langgraph_agent.py
from langgraph.graph import StateGraph, START, END, MessagesState
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, ToolMessage
from langchain_core.tools import tool
from dotenv import load_dotenv

load_dotenv()

# --- Define tools using @tool decorator ---

@tool
def search(query: str) -> str:
    """Search the web for information on a topic."""
    # Mock — replace with real API in production
    return f"Search results for '{query}': Found comprehensive information on this topic."

@tool
def calculate(expression: str) -> str:
    """Evaluate a mathematical expression like '2 * 3 + 4'."""
    try:
        return str(eval(expression, {"__builtins__": {}}, {}))
    except Exception as e:
        return f"Error: {e}"

@tool
def get_weather(city: str) -> str:
    """Get current weather for a city."""
    return f"Weather in {city}: 20°C, partly cloudy."

tools = [search, calculate, get_weather]
tools_by_name = {t.name: t for t in tools}

# --- LLM with tools bound ---
llm = ChatOpenAI(model="gpt-5-mini").bind_tools(tools)

# --- Nodes ---
def agent_node(state: MessagesState) -> dict:
    """Call the LLM to decide next action."""
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def tool_node(state: MessagesState) -> dict:
    """Execute all tool calls from the last AI message."""
    last_message = state["messages"][-1]
    results = []

    for tool_call in last_message.tool_calls:
        tool = tools_by_name[tool_call["name"]]
        result = tool.invoke(tool_call["args"])
        results.append(
            ToolMessage(content=str(result), tool_call_id=tool_call["id"])
        )

    return {"messages": results}

def should_use_tools(state: MessagesState) -> str:
    """Route: use tools or return final answer."""
    last = state["messages"][-1]
    if hasattr(last, "tool_calls") and last.tool_calls:
        return "tools"
    return "__end__"

# --- Build graph ---
graph = StateGraph(MessagesState)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "agent")
graph.add_conditional_edges("agent", should_use_tools)
graph.add_edge("tools", "agent")  # Loop back after tool execution
graph.add_edge("agent", END)  # Loop back after tool execution

# --- Compile with checkpointer ---
checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# png_data = app.get_graph().draw_mermaid_png()
# with open("langgraph_agent.png", "wb") as f:
#     f.write(png_data)

# --- Multi-turn conversation using checkpointing ---
if __name__ == "__main__":
    config = {"configurable": {"thread_id": "demo-session-1"}}

    turns = [
        "What is 25 * 4 + 100?",
        "What's the weather in London?",
        "Based on our conversation, what two things did we discuss?",
    ]

    for question in turns:
        print(f"\nUser: {question}")
        result = app.invoke(
            {"messages": [HumanMessage(content=question)]},
            config=config
        )
        final_message = result["messages"][-1]
        print(f"Agent: {final_message.content}")
