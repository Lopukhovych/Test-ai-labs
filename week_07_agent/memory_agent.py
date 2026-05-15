# memory_agent.py
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass, field
from typing import List, Dict
import json

load_dotenv()
client = OpenAI()

@dataclass
class AgentMemory:
    """Agent's working memory."""
    goal: str = ""
    steps_taken: List[Dict] = field(default_factory=list)
    observations: List[str] = field(default_factory=list)
    current_plan: List[str] = field(default_factory=list)

    def add_step(self, thought: str, action: str, result: str):
        self.steps_taken.append({
            "thought": thought,
            "action": action,
            "result": result
        })
        self.observations.append(result)

    def summary(self) -> str:
        return f"""
GOAL: {self.goal}

COMPLETED STEPS:
{chr(10).join([f"- {s['action']}: {s['result'][:100]}" for s in self.steps_taken])}

OBSERVATIONS SO FAR:
{chr(10).join(self.observations[-3:])}
"""

class MemoryAgent:
    def __init__(self):
        self.client = OpenAI()
        self.memory = AgentMemory()
        self.tools = {
            "search": lambda q: f"Found info about: {q}",
            "calculate": lambda e: str(eval(e)),
            "note": lambda n: f"Noted: {n}"
        }

    def run(self, goal: str, max_steps: int = 5) -> str:
        self.memory.goal = goal

        for step in range(max_steps):
            # Build prompt with memory
            prompt = f"""
{self.memory.summary()}

Available tools: search(query), calculate(expr), note(text)

What should you do next to achieve the goal?
Respond with JSON: {{"thought": "...", "action": "tool(arg)", "done": false}}
Or if complete: {{"thought": "...", "answer": "...", "done": true}}
"""

            response = self.client.chat.completions.create(
                model="gpt-5-mini",
                messages=[{"role": "user", "content": prompt}],
                response_format={"type": "json_object"}
            )

            data = json.loads(response.choices[0].message.content)
            print(f"Step {step + 1}: {data}")

            if data.get("done"):
                return data.get("answer", "Completed")

            # Execute action
            action = data.get("action", "")
            import re
            match = re.match(r"(\w+)\((.*?)\)", action)

            if match:
                tool, arg = match.groups()
                result = self.tools.get(tool, lambda x: "Unknown tool")(arg.strip('"\''))
                self.memory.add_step(data.get("thought", ""), action, result)
            else:
                self.memory.add_step(data.get("thought", ""), "none", "No action taken")

        return "Goal not completed in max steps"

# Test
if __name__ == "__main__":
    agent = MemoryAgent()
    result = agent.run("Calculate 20% of 250 and add 50 to it")
    print(f"\nFinal: {result}")
