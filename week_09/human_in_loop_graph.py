# human_in_loop_graph.py
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from langchain_openai import ChatOpenAI
from typing import TypedDict
from dotenv import load_dotenv

load_dotenv()
llm = ChatOpenAI(model="gpt-4o-mini")

class EmailState(TypedDict):
    request: str
    draft: str
    approved: bool
    final: str

def draft_email(state: EmailState) -> dict:
    """AI drafts an email."""
    response = llm.invoke(
        f"Draft a professional email for this request: {state['request']}"
    )
    return {"draft": response.content}

def human_review(state: EmailState) -> dict:
    """Pause for human approval."""
    feedback = interrupt({
        "draft": state["draft"],
        "action": "Please review this email draft. Return 'approve' or feedback text."
    })

    if feedback == "approve":
        return {"approved": True, "final": state["draft"]}
    else:
        # Human rejected — store feedback, mark not approved
        return {"approved": False, "final": feedback}

def send_email(state: EmailState) -> dict:
    """Send the approved email."""
    print(f"\n📧 SENDING EMAIL:\n{state['final']}")
    return {}

def show_feedback(state: EmailState) -> dict:
    """Show rejection feedback when email is not approved."""
    print(f"\n❌ EMAIL NOT SENT. Feedback received:\n{state['final']}")
    print("Please revise and resubmit.")
    return {}

def route_after_review(state: EmailState) -> str:
    return "send" if state.get("approved") else "feedback"

graph = StateGraph(EmailState)
graph.add_node("draft", draft_email)
graph.add_node("review", human_review)
graph.add_node("send", send_email)
graph.add_node("feedback", show_feedback)

graph.add_edge(START, "draft")
graph.add_edge("draft", "review")
graph.add_conditional_edges("review", route_after_review, {"send": "send", "feedback": "feedback"})
graph.add_edge("send", END)
graph.add_edge("feedback", END)

checkpointer = MemorySaver()
app = graph.compile(checkpointer=checkpointer)

# png_data = app.get_graph().draw_mermaid_png()
# with open("human_in_loop_graph.png", "wb") as f:
#     f.write(png_data)

if __name__ == "__main__":
    request = "Email the team about tomorrow's 10am standup being cancelled"

    # --- Case 1: approved=True → email is sent ---
    print("=" * 50)
    print("CASE 1: Approved")
    print("=" * 50)
    config1 = {"configurable": {"thread_id": "email-approve"}}

    result = app.invoke({"request": request}, config=config1)
    print("Draft ready for review:")
    print(result["draft"])

    app.update_state(config1, values={"approved": True})
    app.invoke(Command(resume="approve"), config=config1)

    # --- Case 2: approved=False → feedback message shown ---
    print("\n" + "=" * 50)
    print("CASE 2: Rejected with feedback")
    print("=" * 50)
    config2 = {"configurable": {"thread_id": "email-reject"}}

    result = app.invoke({"request": request}, config=config2)
    print("Draft ready for review:")
    print(result["draft"])

    app.update_state(config2, values={"approved": False})
    app.invoke(Command(resume="The tone is too casual. Please use a more formal style."), config=config2)
