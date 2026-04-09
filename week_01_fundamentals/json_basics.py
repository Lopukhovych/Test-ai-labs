from openai import OpenAI
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

def analyze_product(description: str) -> dict:
    """Get structured analysis of a product."""
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": """Analyze the product and return JSON with:
{
    "name": "product name",
    "category": "category",
    "sentiment": "positive/negative/neutral",
    "key_features": ["feature1", "feature2", "feature3"]
}
Return ONLY valid JSON, no other text."""
            },
            {"role": "user", "content": description}
        ],
        response_format={"type": "json_object"}  # Enforce JSON
    )

    return json.loads(response.choices[0].message.content)

# Test
product = """
The new iPhone 15 Pro has an amazing camera system with 48MP resolution.
The titanium design makes it lighter than ever. Great battery life too!
"""

result = analyze_product(product)
print(json.dumps(result, indent=2))
