# just a stub for now to be used in main.py until MCP LLM extractor is ready.
#  The idea is to have a placeholder that simulates structured intent extraction from user messages,

from __future__ import annotations
import re
from typing import Dict, Optional, Any

COUNTRIES = ["kenya", "somalia", "ethiopia"]
VARIABLES = ["crop", "pasture", "surface_water", "groundwater", "flood"]
LANGS = {"english": "en", "en": "en", "swahili": "sw", "sw": "sw", "somali": "so", "so": "so"}
SEASONS = ["OND", "MAM", "JJAS", "JJA", "SON", "DJF"]

def _extract_location_country(text: str) -> tuple[Optional[str], Optional[str]]:
    # Matches "for Marsabit, Kenya" / "in Garissa, Kenya"
    m = re.search(r"\b(?:for|in)\s+([a-z][a-z\s'\-]{1,50}?)\s*,\s*(kenya|somalia|ethiopia)\b", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    # Matches "for Marabit in Kenya"
    m = re.search(r"\b(?:for|in)\s+([a-z][a-z\s'\-]{1,50}?)\s+in\s+(kenya|somalia|ethiopia)\b", text)
    if m:
        return m.group(1).strip(), m.group(2).strip()

    return None, None


def extract_intent_stub(text: str) -> Dict[str, Optional[Any]]:
    t = text.lower()

    country = next((c for c in COUNTRIES if c in t), None)
    variable = next((v for v in VARIABLES if v.replace("_", " ") in t or v in t), None)
    language = next((code for k, code in LANGS.items() if k in t), None)

    location, country_from_phrase = _extract_location_country(t)
    if not country and country_from_phrase:
        country = country_from_phrase

    season = next((s for s in SEASONS if s.lower() in t), None)
    year_match = re.search(r"\b(19|20)\d{2}\b", t)
    year = year_match.group(0) if year_match else None

    return {
        "country": country,
        "location": location,
        "variable": variable,
        "season": season,
        "year": year,
        "language": language or "en"
    }