from openai import OpenAI
from pydantic import BaseModel, ValidationError
from typing import List
from dotenv import load_dotenv
import json
import time

load_dotenv()
client = OpenAI()

class EmailExtraction(BaseModel):
    sender: str
    subject: str
    is_urgent: bool
    action_items: List[str]
    sentiment: str  # positive, negative, neutral

def extract_with_retry(email_text: str, max_retries: int = 3) -> EmailExtraction:
    """Extract email data with retry logic."""

    last_error = None

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="gpt-5-mini",
                messages=[
                    {
                        "role": "system",
                        "content": """Extract data from the email. Return JSON:
{
    "sender": "name or email",
    "subject": "email subject/topic",
    "is_urgent": true or false,
    "action_items": ["item1", "item2"],
    "sentiment": "positive/negative/neutral"
}"""
                    },
                    {"role": "user", "content": email_text}
                ],
                response_format={"type": "json_object"}
            )

            data = json.loads(response.choices[0].message.content)
            result = EmailExtraction(**data)
            return result

        except (json.JSONDecodeError, ValidationError) as e:
            last_error = e
            print(f"Attempt {attempt + 1} failed: {e}")
            time.sleep(0.5)  # Brief pause before retry

    raise Exception(f"Failed after {max_retries} attempts: {last_error}")

# Test
email = """
From: boss@company.com
Subject: URGENT: Q4 Report Needed

Hi team,

Please complete the Q4 report by tomorrow. Also schedule a review meeting.
This is top priority!

Thanks,
John
"""

result = extract_with_retry(email)
print(f"Sender: {result.sender}")
print(f"Urgent: {result.is_urgent}")
print(f"Actions: {result.action_items}")
