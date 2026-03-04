# just a stub for now to be used in main.py until MCP LLM extractor is ready.
#  The idea is to have a placeholder that simulates structured intent extraction from user messages,

from __future__ import annotations
from typing import Dict, Optional

COUNTRIES = ["kenya", "somalia", "ethiopia"]
VARIABLES = ["crop", "pasture", "surface_water", "groundwater", "flood"]
LANGS = {"english": "en", "en": "en", "swahili": "sw", "sw": "sw", "somali": "so", "so": "so"}

def extract_intent_stub(text: str) -> Dict[str, Optional[str]]:
    t = text.lower()

    country = next((c for c in COUNTRIES if c in t), None)
    variable = next((v for v in VARIABLES if v.replace("_", " ") in t or v in t), None)
    language = next((code for k, code in LANGS.items() if k in t), None)

    # season/year left None for now (force clarification)
    return {
        "country": country,
        "location": None,   # resolve via store search in /api/chat
        "variable": variable,
        "season": None,
        "year": None,
        "language": language or "en"
    }