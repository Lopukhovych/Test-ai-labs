from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

class ChatResponse(BaseModel):
    message: str
    mood: str  # helpful, curious, apologetic, enthusiastic
    follow_up_questions: List[str] = []
    topic: str
    confidence: float

class StructuredChatbot:
    def __init__(self, system_prompt: str):
        self.system_prompt = system_prompt
        self.messages = []

    def chat(self, user_message: str) -> ChatResponse:
        """Get a structured response from the chatbot."""

        self.messages.append({"role": "user", "content": user_message})

        system = f"""{self.system_prompt}

Always respond with JSON:
{{
    "message": "your response to the user",
    "mood": "helpful/curious/apologetic/enthusiastic",
    "follow_up_questions": ["suggested question 1", "suggested question 2"],
    "topic": "the main topic discussed",
    "confidence": 0.0 to 1.0 (how confident you are)
}}"""

        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": system},
                *self.messages
            ],
            response_format={"type": "json_object"}
        )

        data = json.loads(response.choices[0].message.content)
        result = ChatResponse(**data)

        # Store just the message in history
        self.messages.append({"role": "assistant", "content": result.message})

        return result

# Interactive test
if __name__ == "__main__":
    bot = StructuredChatbot("You are a helpful coding assistant.")

    print("Structured Chatbot! Type 'quit' to exit.\n")

    while True:
        user_input = input("You: ")
        if user_input.lower() == 'quit':
            break

        response = bot.chat(user_input)

        print(f"AI [{response.mood}]: {response.message}")
        if response.follow_up_questions:
            print(f"   You could ask: {response.follow_up_questions[0]}")
        print()
