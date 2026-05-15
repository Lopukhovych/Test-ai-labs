# content_moderation.py
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

def moderate_content(text: str) -> dict:
    """Check content for policy violations."""

    response = client.moderations.create(input=text)
    result = response.results[0]

    return {
        "flagged": result.flagged,
        "categories": {
            cat: flagged for cat, flagged in result.categories.model_dump().items() if flagged
        },
        "scores": {
            cat: score for cat, score in result.category_scores.model_dump().items()
            if score > 0.1
        }
    }

def safe_chat(user_message: str) -> str:
    """Chat with content moderation."""

    # Check input
    input_mod = moderate_content(user_message)
    if input_mod["flagged"]:
        return f"I can't process that request. Categories: {list(input_mod['categories'].keys())}"

    # Get response
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": user_message}
        ]
    )

    answer = response.choices[0].message.content

    # Check output
    output_mod = moderate_content(answer)
    if output_mod["flagged"]:
        return "I generated a response but it was flagged by moderation."

    return answer

# Test
if __name__ == "__main__":
    print(safe_chat("What's the weather like today?"))
