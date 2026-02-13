from __future__ import annotations


BILLING_KEYWORDS = {"refund", "charge", "charged", "billing", "invoice", "payment"}
DELIVERY_KEYWORDS = {"late", "delay", "delivery", "shipping", "shipment", "arrive"}
TECH_KEYWORDS = {"error", "bug", "issue", "broken", "not working", "crash", "failed"}


def _contains_any(text: str, keywords: set[str]) -> bool:
    return any(k in text for k in keywords)


def generate_reply(user_text: str, sentiment: str, confidence: float) -> str:
    t = user_text.lower()

    if _contains_any(t, BILLING_KEYWORDS):
        return (
            "I can help with billing. Please share your order ID or email used for purchase, "
            "and I will check refund or charge details."
        )
    if _contains_any(t, DELIVERY_KEYWORDS):
        return (
            "I can help with delivery status. Please share your order or tracking number so I can "
            "check delays and next steps."
        )
    if _contains_any(t, TECH_KEYWORDS):
        return (
            "I can help troubleshoot this. Tell me your device, app/browser version, and the exact "
            "error message so I can guide you."
        )

    if confidence < 0.45:
        return (
            "Thanks for the message. I want to make sure I understand correctly. "
            "Could you share a little more detail?"
        )

    if sentiment == "negative":
        return (
            "I am sorry this has been frustrating. I can help resolve it quickly. "
            "Please describe what happened and what outcome you want."
        )
    if sentiment == "positive":
        return "Happy to hear that. Is there anything else you want help with today?"
    return "Thanks for reaching out. Could you provide a bit more context so I can assist better?"
