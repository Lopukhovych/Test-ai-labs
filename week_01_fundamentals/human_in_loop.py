from openai import OpenAI
from pydantic import BaseModel
from typing import Literal
from dotenv import load_dotenv
import json

load_dotenv()
client = OpenAI()

class EmailDraft(BaseModel):
    to: str
    subject: str
    body: str
    confidence: float
    risk_level: Literal["low", "medium", "high"]
    needs_review: bool

class ApprovalResult(BaseModel):
    approved: bool
    edited_body: str | None = None
    reason: str | None = None

def draft_email(request: str) -> EmailDraft:
    """AI drafts an email based on user request."""
    response = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {
                "role": "system",
                "content": """Draft a professional email based on the request.
Return JSON:
{
    "to": "recipient email",
    "subject": "email subject",
    "body": "email body",
    "confidence": 0.0-1.0,
    "risk_level": "low/medium/high",
    "needs_review": true if sensitive content or low confidence
}"""
            },
            {"role": "user", "content": request}
        ],
        response_format={"type": "json_object"}
    )
    return EmailDraft(**json.loads(response.choices[0].message.content))

def get_human_approval(draft: EmailDraft) -> ApprovalResult:
    """Present draft to human for approval."""
    print("\n" + "="*50)
    print("📧 EMAIL DRAFT FOR APPROVAL")
    print("="*50)
    print(f"To: {draft.to}")
    print(f"Subject: {draft.subject}")
    print(f"Risk Level: {draft.risk_level}")
    print(f"AI Confidence: {draft.confidence:.0%}")
    print("-"*50)
    print(draft.body)
    print("-"*50)

    choice = input("\n[A]pprove / [E]dit / [R]eject: ").strip().lower()

    if choice == 'a':
        return ApprovalResult(approved=True)
    elif choice == 'e':
        edited = input("Enter corrected body: ")
        return ApprovalResult(approved=True, edited_body=edited)
    else:
        reason = input("Rejection reason: ")
        return ApprovalResult(approved=False, reason=reason)

def send_email(draft: EmailDraft):
    """Simulate sending email."""
    print(f"\n✅ Email sent to {draft.to}!")

def email_with_approval(request: str):
    """Complete flow: draft → approval → send."""
    draft = draft_email(request)

    # Auto-approve low risk + high confidence
    if draft.risk_level == "low" and draft.confidence > 0.9 and not draft.needs_review:
        print("Auto-approved (low risk, high confidence)")
        send_email(draft)
        return

    # Otherwise, get human approval
    approval = get_human_approval(draft)

    if approval.approved:
        if approval.edited_body:
            draft.body = approval.edited_body
        send_email(draft)
    else:
        print(f"❌ Email rejected: {approval.reason}")

# Test
if __name__ == "__main__":
    email_with_approval("Write an email to john@company.com apologizing for the delayed shipment. Before send an email, I need to approve it")
