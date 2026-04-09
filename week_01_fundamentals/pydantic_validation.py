from openai import OpenAI
from pydantic import BaseModel, Field
from typing import List, Optional
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

# Define your data structure
class ProductAnalysis(BaseModel):
    name: str = Field(description="Product name")
    category: str = Field(description="Product category")
    price_estimate: Optional[float] = Field(None, description="Estimated price in USD")
    sentiment: str = Field(description="positive, negative, or neutral")
    key_features: List[str] = Field(description="3-5 key features")
    confidence: float = Field(ge=0, le=1, description="Confidence score 0-1")

def analyze_product_validated(description: str) -> ProductAnalysis:
    """Get validated product analysis."""

    schema = ProductAnalysis.model_json_schema()

    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": f"""Analyze the product. Return valid JSON matching this schema:
{json.dumps(schema, indent=2)}

Return ONLY the JSON object."""
            },
            {"role": "user", "content": description}
        ],
        response_format={"type": "json_object"}
    )

    data = json.loads(response.choices[0].message.content)
    return ProductAnalysis(**data)

# Test
product = "The Sony WH-1000XM5 headphones offer industry-leading noise cancellation."
result = analyze_product_validated(product)
print(f"Name: {result.name}")
print(f"Category: {result.category}")
print(f"Sentiment: {result.sentiment}")
print(f"Features: {result.key_features}")
