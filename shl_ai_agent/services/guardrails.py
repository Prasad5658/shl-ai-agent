RISKY_INSTRUCTIONS = (
    "ignore previous",
    "ignore the catalog",
    "make up",
    "invent",
    "system prompt",
    "developer message",
    "secret",
)


def is_prompt_injection(text):
    lowered = text.lower()
    return any(marker in lowered for marker in RISKY_INSTRUCTIONS)


def clean_user_text(text):
    return " ".join(text.strip().split())


def guardrail_reply():
    return {
        "reply": (
            "I can help recommend SHL assessments, but I can only use the catalog "
            "and the hiring requirements you provide."
        ),
        "recommendations": [],
        "end_of_conversation": False,
    }
