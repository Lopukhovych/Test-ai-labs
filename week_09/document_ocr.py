# document_ocr.py
from openai import OpenAI
from pydantic import BaseModel
from typing import List, Optional
import json
from dotenv import load_dotenv

load_dotenv()
client = OpenAI()

class InvoiceData(BaseModel):
    invoice_number: Optional[str]
    date: Optional[str]
    vendor: Optional[str]
    total_amount: Optional[float]
    line_items: List[str]

def extract_invoice_data(pdf_path: str) -> InvoiceData:
    """Extract structured data from an invoice PDF."""
    with open(pdf_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": """Extract invoice data and return JSON:
{
    "invoice_number": "...",
    "date": "...",
    "vendor": "...",
    "total_amount": 0.00,
    "line_items": ["item 1", "item 2"]
}"""},
                    {"type": "file", "file": {"file_id": uploaded.id}}
                ]
            }],
            response_format={"type": "json_object"}
        )
    finally:
        client.files.delete(uploaded.id)

    data = json.loads(response.choices[0].message.content)
    print(f'data: {data}')
    return InvoiceData(**data)

def analyze_chart(image_path: str) -> dict:
    """Extract insights from a chart or graph image."""
    with open(image_path, "rb") as f:
        uploaded = client.files.create(file=f, purpose="user_data")

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": """Analyze this chart. Return JSON:
{
    "chart_type": "bar|line|pie|scatter|...",
    "title": "...",
    "key_insights": ["insight1", "insight2"],
    "trend": "increasing|decreasing|stable",
    "notable_values": {"max": "...", "min": "..."}
}"""},
                    {"type": "file", "file": {"file_id": uploaded.id}}
                ]
            }],
            response_format={"type": "json_object"}
        )
    finally:
        client.files.delete(uploaded.id)

    return json.loads(response.choices[0].message.content)


if __name__ == "__main__":
    # invoice_data = extract_invoice_data("/Users/volod34/PycharmProjects/my-ai-project/week_09/invoice.pdf")
    invoice_data = analyze_chart("/Users/volod34/PycharmProjects/my-ai-project/week_09/invoice.pdf")
    print(f'invoice_data: {invoice_data}')
