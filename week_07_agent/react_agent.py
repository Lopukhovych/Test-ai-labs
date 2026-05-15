# react_agent.py
from openai import OpenAI
from dotenv import load_dotenv
import json
import re

load_dotenv()
client = OpenAI()

# Tools
def search_web(query: str) -> str:
    """Mock web search."""
    return f"Search results for '{query}': Found relevant information about {query}."

def calculate(expression: str) -> str:
    """Calculate math expression."""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"

def get_weather(city: str) -> str:
    """Mock weather."""
    return f"Weather in {city}: 20°C, partly cloudy"

tools = {
    "search_web": search_web,
    "calculate": calculate,
    "get_weather": get_weather
}

REACT_PROMPT = """You are a helpful assistant that uses tools to answer questions.

Available tools:
- search_web(query): Search the web for information
- calculate(expression): Do math calculations
- get_weather(city): Get weather for a city

Always use this format:
THOUGHT: [your reasoning about what to do]
ACTION: tool_name(arguments)

When you have enough information:
THOUGHT: [summary of what you learned]
ANSWER: [your final answer]

Begin!

Question: {question}
"""

def parse_response(text: str) -> tuple:
    """Parse agent response into thought, action, or answer."""

    # Check for answer
    if "ANSWER:" in text:
        thought = re.search(r"THOUGHT:(.*?)(?:ANSWER:|$)", text, re.DOTALL)
        answer = re.search(r"ANSWER:(.*?)$", text, re.DOTALL)
        return "answer", {
            "thought": thought.group(1).strip() if thought else "",
            "answer": answer.group(1).strip() if answer else text
        }

    # Check for action
    if "ACTION:" in text:
        thought = re.search(r"THOUGHT:(.*?)ACTION:", text, re.DOTALL)
        action = re.search(r"ACTION:\s*(\w+)\((.*?)\)", text)

        if action:
            return "action", {
                "thought": thought.group(1).strip() if thought else "",
                "tool": action.group(1),
                "args": action.group(2).strip('"\'')
            }

    return "unknown", {"text": text}

def run_agent(question: str, max_steps: int = 5) -> str:
    """Run the ReAct agent."""

    prompt = REACT_PROMPT.format(question=question)
    messages = [{"role": "user", "content": prompt}]

    for step in range(max_steps):
        print(f"\n--- Step {step + 1} ---")

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            temperature=0
        )

        text = response.choices[0].message.content
        print(text)

        result_type, data = parse_response(text)

        if result_type == "answer":
            return data["answer"]

        if result_type == "action":
            # Execute tool
            tool_name = data["tool"]
            tool_args = data["args"]

            if tool_name in tools:
                observation = tools[tool_name](tool_args)
            else:
                observation = f"Unknown tool: {tool_name}"

            print(f"OBSERVATION: {observation}")

            # Add to messages
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": f"OBSERVATION: {observation}"})
        else:
            messages.append({"role": "assistant", "content": text})
            messages.append({"role": "user", "content": "Continue reasoning. Use THOUGHT/ACTION or ANSWER format."})

    return "Max steps reached without answer"

# Test
if __name__ == "__main__":
    questions = [
        "What is 25 * 17?",
        "What's the weather like in Tokyo?",
        "Calculate 15% of 200 and tell me if it's more than 25"
    ]

    for q in questions:
        print(f"\n{'='*50}")
        print(f"QUESTION: {q}")
        print("="*50)
        answer = run_agent(q)
        print(f"\n✅ FINAL: {answer}")
