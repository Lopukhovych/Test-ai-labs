# pii_protection.py
import re
from typing import List
from dataclasses import dataclass

@dataclass
class PIIMatch:
    type: str
    value: str
    start: int
    end: int

def detect_pii(text: str) -> List[PIIMatch]:
    """Detect PII in text."""

    patterns = {
        "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
        "phone": r'\b(\+\d{1,2}\s?)?\(?\d{3}\)?[\s.-]?\d{3}[\s.-]?\d{4}\b',
        "ssn": r'\b\d{3}-\d{2}-\d{4}\b',
        "credit_card": r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',
        "ip_address": r'\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b',
    }

    matches = []
    for pii_type, pattern in patterns.items():
        for match in re.finditer(pattern, text, re.IGNORECASE):
            matches.append(PIIMatch(
                type=pii_type,
                value=match.group(),
                start=match.start(),
                end=match.end()
            ))

    return matches

def redact_pii(text: str) -> str:
    """Replace PII with redaction markers."""

    matches = detect_pii(text)

    # Sort by position (reverse) to replace from end
    matches.sort(key=lambda m: m.start, reverse=True)

    for match in matches:
        redacted = f"[{match.type.upper()}_REDACTED]"
        text = text[:match.start] + redacted + text[match.end:]

    return text

# Test
if __name__ == "__main__":
    test_text = """
    Contact John at john.doe@example.com or call 555-123-4567.
    His SSN is 123-45-6789 and credit card is 4532-1234-5678-9012.
    Server IP: 192.168.1.100
    """

    print("=== Original ===")
    print(test_text)

    print("=== PII Detected ===")
    for match in detect_pii(test_text):
        print(f"  {match.type}: {match.value}")

    print("=== Redacted ===")
    print(redact_pii(test_text))
