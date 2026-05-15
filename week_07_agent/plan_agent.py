# plan_agent.py
from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

class PlanAndExecuteAgent:
    """Agent that creates a plan first, then executes it."""

    def __init__(self):
        self.client = OpenAI()
        self.tools = {
            "search": lambda q: f"Results for '{q}': relevant information found",
            "write": lambda text: f"Written: {text[:50]}...",
            "calculate": lambda e: str(eval(e)),
        }

    def create_plan(self, goal: str) -> list:
        """Have LLM create a plan."""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Create a step-by-step plan to achieve this goal:

GOAL: {goal}

Available tools: search(query), write(text), calculate(expression)

Return JSON: {{"steps": ["step 1", "step 2", ...]}}
Each step should be a specific action with the tool to use."""
            }],
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)
        return data.get("steps", [])

    def execute_step(self, step: str, context: list) -> str:
        """Execute a single step."""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Execute this step:

STEP: {step}

Previous context:
{chr(10).join(context[-3:])}

Available tools: search(query), write(text), calculate(expression)

Return JSON: {{"tool": "tool_name", "argument": "...", "reasoning": "..."}}"""
            }],
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)

        tool = data.get("tool", "")
        arg = data.get("argument", "")

        if tool in self.tools:
            return self.tools[tool](arg)
        return f"Executed: {step}"

    def run(self, goal: str) -> str:
        # Phase 1: Plan
        print("📋 Creating plan...")
        plan = self.create_plan(goal)

        print(f"Plan ({len(plan)} steps):")
        for i, step in enumerate(plan, 1):
            print(f"  {i}. {step}")

        # Phase 2: Execute
        print("\n🚀 Executing plan...")
        context = []

        for i, step in enumerate(plan, 1):
            print(f"\nStep {i}: {step}")
            result = self.execute_step(step, context)
            print(f"  → {result}")
            context.append(f"{step}: {result}")

        # Phase 3: Summarize
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Summarize what was accomplished:

GOAL: {goal}

EXECUTION:
{chr(10).join(context)}

Provide a brief summary of results."""
            }]
        )

        return response.choices[0].message.content

# Test
if __name__ == "__main__":
    agent = PlanAndExecuteAgent()
    result = agent.run("Research Python's history and calculate how old it is")
    print(f"\n✅ Summary:\n{result}")
