from dotenv import load_dotenv
from strands import Agent, tool
from strands.models.openai import OpenAIModel

load_dotenv()

model = OpenAIModel(model_id="gpt-4o-mini")

@tool
def researcher(query: str) -> str:
    """Research a topic and return detailed findings."""
    agent = Agent(model=model, system_prompt="You are a research expert. Find accurate, detailed information.")
    return str(agent(query))

@tool
def technical_writer(content: str) -> str:
    """Turn research findings into a polished 300-word developer summary."""
    agent = Agent(model=model, system_prompt="You are a technical writer. Be concise and clear.")
    return str(agent(f"Summarize for developers: {content}"))

orchestrator = Agent(
    model=model,
    system_prompt="""Route tasks to specialists:
1. Use 'researcher' to gather information
2. Use 'technical_writer' to produce the final output""",
    tools=[researcher, technical_writer]
)

result = orchestrator("Research and summarize AI agent frameworks in 2025.")
