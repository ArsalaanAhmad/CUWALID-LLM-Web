from __future__ import annotations


POLICY_KEYWORDS = [
    "what should we do",
    "tell me what to do",
    "what do we do",
    "what should i do",
    "what should our ministry do",
    "what policy should",
    "which policy should",
    "recommend policy",
    "policy advice",
    "recommendation",
    "recommendations",
    "decide for us",
    "give me a decision",
    "action plan",
    "what action",
    "what actions",
    "what should be done",
]

VAGUE_KEYWORDS = [
    "help",
    "what now",
    "now what",
    "anything",
    "idk",
    "i dont know",
    "i don't know",
    "not sure",
    "hmm",
    "uh",
    "ok",
    "hi",
    "hello",
]

IN_SCOPE_KEYWORDS = [
    "cuwalid",
    "forecast",
    "seasonal",
    "season",
    "ond",
    "mam",
    "jjas",
    "hydrology",
    "hydrological",
    "water",
    "flood",
    "groundwater",
    "surface_water",
    "surface water",
    "crop",
    "pasture",
    "modelling",
    "model",
    "workflow",
    "drip",
    "dryp",
    "location",
    "district",
    "region",
    "country",
]

OUT_OF_SCOPE_KEYWORDS = [
    "football",
    "soccer",
    "nba",
    "cricket",
    "tennis",
    "movie",
    "music",
    "celebrity",
    "gossip",
    "fashion",
    "recipe",
    "dating",
    "meme",
    "bitcoin",
    "crypto price",
    "stock market",
    "election",
    "president",
    "parliament",
    "party politics",
    "joke",
    "tell me a joke",
    "how are you",
    "who are you",
    "good morning",
    "good night",
]


def _normalize(message: str) -> str:
    return (message or "").strip().lower()


def is_policy_request(message: str) -> bool:
    """
    Detect requests for operational/policy decisions or prescriptive recommendations.
    """
    text = _normalize(message)
    if not text:
        return False

    return any(keyword in text for keyword in POLICY_KEYWORDS)


def refusal_response() -> dict:
    """
    Standard refusal payload for policy/decision requests.
    """
    return {
        "kind": "refuse",
        "reply": (
            "I can’t make operational or policy decisions. "
            "I can provide forecast information if you share country, location, "
            "variable, season, year, and language."
        ),
        "meta": {
            "reason": "policy_request_refused"
        }
    }


def is_nonsense_or_empty(message: str) -> bool:
    """
    Detect empty or extremely vague inputs that do not provide actionable detail.
    """
    text = _normalize(message)
    if not text:
        return True

    if len(text) <= 2:
        return True

    if text in VAGUE_KEYWORDS:
        return True

    return False


def nonsense_response() -> dict:
    """
    Clarification payload for empty/insufficient inputs.
    """
    return {
        "kind": "clarify",
        "reply": "Please provide more details such as location, forecast type, and time period.",
        "meta": {
            "reason": "insufficient_input"
        }
    }


def is_out_of_scope(message: str) -> bool:
    """
    Detect prompts unrelated to CUWALID forecasts/modelling.
    """
    text = _normalize(message)
    if not text:
        return False

    if any(keyword in text for keyword in IN_SCOPE_KEYWORDS):
        return False

    if any(keyword in text for keyword in OUT_OF_SCOPE_KEYWORDS):
        return True

    # Lightweight catch-all for general chit-chat not tied to domain.
    if text in {"hi", "hello", "hey", "yo", "sup", "thanks", "thank you"}:
        return True

    return False


def out_of_scope_response() -> dict:
    """
    Refusal payload for out-of-domain requests.
    """
    return {
        "kind": "refuse",
        "reply": (
            "I can only help with CUWALID forecasts and modelling workflows. "
            "Please ask about hydrology, seasonal forecasts, or system usage."
        ),
        "meta": {
            "reason": "out_of_scope"
        }
    }
