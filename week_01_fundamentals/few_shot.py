from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def classify_intent(message: str) -> str:
    """Classify user intent using few-shot prompting."""

    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": """Classify the user's intent. Examples:

Message: "I want to cancel my subscription"
Intent: cancellation

Message: "How do I reset my password?"
Intent: support

Message: "What plans do you offer?"
Intent: pricing

Message: "Your product is amazing!"
Intent: feedback

Message: "I can't login to my account"
Intent: support

Respond with only the intent category."""
            },
            {"role": "user", "content": message}
        ]
    )

    return response.choices[0].message.content.strip().lower()

# Test various messages
messages = [
    "I want a refund",
    "How much does the pro plan cost?",
    "I love using your app!",
    "My payment didn't go through",
    "Please delete my account"
]

for msg in messages:
    intent = classify_intent(msg)
    print(f"'{msg}' → {intent}")
