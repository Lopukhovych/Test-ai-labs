# research_team.py
from strands import Agent, tool
from strands.models import OpenAIModel
from dotenv import load_dotenv

load_dotenv()

model = OpenAIModel(model_id="gpt-4o-mini")

# --- Specialist sub-agents defined as @tool functions ---

@tool
def researcher(query: str) -> str:
    """Research a topic thoroughly and return detailed findings.

    Args:
        query: The research question or topic to investigate
    Returns:
        Detailed research findings with key facts
    """
    research_agent = Agent(
        model=model,
        system_prompt="""You are a Senior Research Analyst.
Research topics thoroughly and provide accurate, well-organized findings.
Always include key facts, statistics, and relevant context."""
    )
    response = research_agent(query)
    return str(response)


@tool
def fact_checker(claim: str) -> str:
    """Verify a claim and flag anything inaccurate or unsubstantiated.

    Args:
        claim: A statement or set of findings to fact-check
    Returns:
        Verification result with accuracy notes
    """
    checker_agent = Agent(
        model=model,
        system_prompt="""You are a meticulous Fact Checker.
Analyze claims critically. Flag anything that seems inaccurate,
unsupported, or that needs clarification. Be specific."""
    )
    response = checker_agent(f"Fact-check this: {claim}")
    return str(response)


@tool
def technical_writer(content: str) -> str:
    """Transform research into a clear, engaging 300-word technical summary.

    Args:
        content: Research findings to summarize
    Returns:
        A polished 300-word summary for a developer audience
    """
    writer_agent = Agent(
        model=model,
        system_prompt="""You are a Technical Writer.
Transform complex research into clear, concise content for developers.
Aim for ~300 words. Use plain language; avoid jargon."""
    )
    response = writer_agent(f"Write a 300-word developer summary of: {content}")
    return str(response)


# --- Orchestrator: routes tasks to the right specialist ---

orchestrator = Agent(
    model=model,
    system_prompt="""You are a research team orchestrator.
For any research request, follow this pipeline:
1. Use 'researcher' to gather detailed findings
2. Use 'fact_checker' to verify the findings
3. Use 'technical_writer' to produce the final summary
Always complete all three steps before responding.""",
    tools=[researcher, fact_checker, technical_writer]
)

if __name__ == "__main__":
    result = orchestrator(
        "Research the current state of AI agent frameworks (LangGraph, Strands Agents, AutoGen) "
        "and their key use cases in 2025."
    )
    print("\n" + "="*60)
    print("FINAL OUTPUT:")
    print("="*60)
    print(result)
