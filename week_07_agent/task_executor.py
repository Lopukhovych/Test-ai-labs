# task_executor.py
from openai import OpenAI
from dotenv import load_dotenv
from dataclasses import dataclass
from typing import List, Optional
import json

load_dotenv()

def extract_json(text: str) -> dict:
    """Extract first JSON object from text that may contain extra content."""
    start = text.find("{")
    end = text.rfind("}") + 1
    if start == -1 or end == 0:
        raise ValueError(f"No JSON object found in: {text!r}")
    return json.loads(text[start:end])

@dataclass
class TaskResult:
    task: str
    success: bool
    result: str
    subtasks: List['TaskResult'] = None

class TaskExecutor:
    """Execute complex multi-step tasks."""

    def __init__(self):
        self.client = OpenAI()
        self.tools = {
            "search": lambda q: f"Found: {q}",
            "write": lambda t: f"Created: {t[:30]}...",
            "calculate": lambda e: str(eval(e)),
            "list": lambda items: f"Listed {items}",
        }

    def decompose_task(self, task: str) -> List[str]:
        """Break down a complex task."""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Break this task into simple subtasks:

TASK: {task}

Tools available: search, write, calculate, list

Return JSON: {{"subtasks": ["task1", "task2", ...]}}
Each subtask should be simple and actionable."""
            }],
            response_format={"type": "json_object"}
        )

        data = extract_json(response.choices[0].message.content)
        return data.get("subtasks", [task])

    def execute_simple_task(self, task: str) -> str:
        """Execute a simple atomic task."""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Execute this task:

TASK: {task}

Available tools: search(query), write(text), calculate(expr), list(items)

Return JSON: {{"tool": "name", "arg": "argument", "result": "what happened"}}"""
            }],
            response_format={"type": "json_object"}
        )

        data = extract_json(response.choices[0].message.content)

        tool = data.get("tool", "")
        if tool in self.tools:
            return self.tools[tool](data.get("arg", ""))
        return data.get("result", "Completed")

    def execute(self, task: str) -> TaskResult:
        """Execute a potentially complex task."""

        # Try to decompose
        subtasks = self.decompose_task(task)

        if len(subtasks) <= 1:
            # Simple task - execute directly
            result = self.execute_simple_task(task)
            return TaskResult(task=task, success=True, result=result)

        # Complex task - execute subtasks
        print(f"📋 Decomposed into {len(subtasks)} subtasks")

        subtask_results = []
        all_success = True

        for i, subtask in enumerate(subtasks, 1):
            print(f"  {i}. {subtask}")
            result = self.execute_simple_task(subtask)
            print(f"     → {result}")

            subtask_results.append(TaskResult(
                task=subtask,
                success=True,
                result=result
            ))

        # Combine results
        combined = "\n".join([f"- {r.result}" for r in subtask_results])

        return TaskResult(
            task=task,
            success=all_success,
            result=combined,
            subtasks=subtask_results
        )

# Test
if __name__ == "__main__":
    executor = TaskExecutor()

    task = "Research Python's creation date, calculate its age, and list 3 major features"
    print(f"🎯 Task: {task}\n")

    result = executor.execute(task)

    print(f"\n✅ Final Result:\n{result.result}")
