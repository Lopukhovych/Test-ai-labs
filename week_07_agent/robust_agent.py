# robust_agent.py
from openai import OpenAI
from dotenv import load_dotenv
import json
import traceback

load_dotenv()
client = OpenAI()

class RobustAgent:
    """Agent with error handling and retries."""

    def __init__(self):
        self.client = OpenAI()
        self.max_retries = 3

    def execute_with_retry(self, task: str) -> dict:
        """Execute a task with retries on failure."""

        errors = []

        for attempt in range(self.max_retries):
            try:
                result = self._attempt_task(task, errors)
                return {"success": True, "result": result, "attempts": attempt + 1}

            except Exception as e:
                error_msg = f"Attempt {attempt + 1}: {str(e)}"
                errors.append(error_msg)
                print(f"⚠️ {error_msg}")

                if attempt == self.max_retries - 1:
                    return {
                        "success": False,
                        "error": str(e),
                        "attempts": self.max_retries,
                        "all_errors": errors
                    }

    def _attempt_task(self, task: str, previous_errors: list) -> str:
        """Single attempt at a task."""

        error_context = ""
        if previous_errors:
            error_context = f"""
Previous attempts failed:
{chr(10).join(previous_errors)}

Learn from these errors and try a different approach.
"""

        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=[{
                "role": "user",
                "content": f"""Complete this task:

TASK: {task}

{error_context}

Return JSON: {{
    "approach": "how you'll do it",
    "result": "the actual result"
}}"""
            }],
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)

        # Simulate occasional failures for demo
        import random
        if random.random() < 0.3 and len(previous_errors) < 2:
            raise Exception("Simulated random failure")

        return data.get("result", "No result")

    def run(self, goal: str) -> str:
        """Run the agent on a goal."""

        result = self.execute_with_retry(goal)

        if result["success"]:
            return f"✅ Success (attempts: {result['attempts']}): {result['result']}"
        else:
            return f"❌ Failed after {result['attempts']} attempts: {result['error']}"

# Test
if __name__ == "__main__":
    agent = RobustAgent()
    print(agent.run("What is 2 + 2?"))
