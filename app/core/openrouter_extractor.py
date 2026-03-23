# This file implements the OpenRouter extractor, which uses the OpenRouter API to extract structured information from user requests.

import os
import json
import re
import time
import logging
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

logger = logging.getLogger("cuwalid.extractor.openrouter")

MODEL_NAME = os.getenv("LLM_MODEL", "qwen/qwen3-235b-a22b-thinking-2507")

client = AsyncOpenAI(
    api_key=os.getenv("OPENROUTER_API_KEY"),
    base_url="https://openrouter.ai/api/v1"
)

EXTRACTION_SCHEMA = {
    "country": None,
    "location": None,
    "season": None,
    "year": None,
    "variable": None,
    "language": None
}

SYSTEM_PROMPT = f"""
You are a strict JSON extractor for seasonal hydrological forecast requests.

Task:
Extract only these fields from the user message:
- country
- location
- season
- year
- variable
- language

Output rules (mandatory):
1) Return exactly one JSON object.
2) Use only the six keys above.
3) If a value is missing/unclear, set it to null.
4) Do NOT include markdown, code fences, prose, or explanations.
5) Do NOT add any extra keys.

Hydrological forecast hints:
- Variables are usually one of: flood, crop, pasture, surface_water, groundwater.
- Location may be district/county/region/city/place.
- Season may be codes like OND, MAM, JJAS, JJA, SON, DJF, etc.
- Year should be numeric (e.g., 2026) when present.
- Language may be names or codes (for example English/en, Swahili/sw, Somali/so, Amharic/am, Oromo/or).

Required output shape:
{json.dumps(EXTRACTION_SCHEMA, indent=2)}
"""


def _strip_markdown_fences(text: str) -> str:
    """
    Remove surrounding markdown code fences if present.
    """
    s = (text or "").strip()
    if not s:
        return s

    # Remove opening fence like ```json or ```
    s = re.sub(r"^```(?:json)?\s*", "", s, flags=re.IGNORECASE)
    # Remove trailing fence
    s = re.sub(r"\s*```$", "", s)
    return s.strip()


def _extract_json_object_text(text: str) -> str:
    """
    Return the JSON object substring if extra tokens were returned.
    """
    s = _strip_markdown_fences(text)
    start = s.find("{")
    end = s.rfind("}")

    if start == -1 or end == -1 or end < start:
        return s
    return s[start:end + 1]


def _normalize_extracted_payload(payload: dict) -> dict:
    """
    Enforce stable schema and types for downstream deterministic logic.
    """
    normalized = dict(EXTRACTION_SCHEMA)
    if not isinstance(payload, dict):
        return normalized

    for key in normalized:
        value = payload.get(key)
        if isinstance(value, str):
            value = value.strip() or None

        if key == "year" and value is not None:
            try:
                value = int(value)
            except (TypeError, ValueError):
                value = None

        normalized[key] = value

    return normalized

async def extract_intent_via_openrouter(text: str) -> dict:
    start = time.perf_counter()

    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )

    raw = (response.choices[0].message.content or "").strip()
    candidate = _extract_json_object_text(raw)

    try:
        parsed = json.loads(candidate)
    except json.JSONDecodeError as e:
        # Keep extraction resilient: return null-filled schema on invalid model output.
        logger.warning(
            "invalid_json_from_model model=%s error=%s raw_preview=%s",
            MODEL_NAME,
            e,
            raw[:200],
        )
        return dict(EXTRACTION_SCHEMA)

    normalized = _normalize_extracted_payload(parsed)
    logger.info(
        "extractor_complete model=%s duration_ms=%.2f extracted_keys=%s",
        MODEL_NAME,
        (time.perf_counter() - start) * 1000,
        sorted(list(normalized.keys())),
    )
    return normalized