from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class Chatbot:
    def __init__(self, system_prompt: str = "You are a helpful assistant."):
        self.client = OpenAI()
        self.system_prompt = system_prompt
        self.messages = [{"role": "system", "content": system_prompt}]

    def chat(self, user_message: str) -> str:
        """Send a message and get a response."""
        # Add user message
        self.messages.append({"role": "user", "content": user_message})

        # Get AI response
        response = self.client.chat.completions.create(
            model="gpt-5-mini",
            messages=self.messages
        )

        ai_message = response.choices[0].message.content

        # Add to history
        self.messages.append({"role": "assistant", "content": ai_message})

        return ai_message

    def clear_history(self):
        """Reset conversation, keeping system prompt."""
        self.messages = [{"role": "system", "content": self.system_prompt}]

    def get_history(self) -> list:
        """Return conversation history."""
        return self.messages[1:]  # Exclude system prompt
