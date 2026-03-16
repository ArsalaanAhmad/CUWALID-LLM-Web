# This file implements the OpenRouter extractor, which uses the OpenRouter API to extract structured information from user requests.

import os
import json
from openai import AsyncOpenAI

from dotenv import load_dotenv
load_dotenv()

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
You are a structured extractor for a hydrological forecast system.

Extract the following fields from the user's request and return ONLY valid JSON:
- country
- location
- season
- year
- variable
- language

Allowed variables include:
flood, crop, pasture, surface_water, groundwater

If a field is missing or unclear, set it to null.

Return ONLY JSON.
Example format:
{json.dumps(EXTRACTION_SCHEMA, indent=2)}
"""

async def extract_intent_via_openrouter(text: str) -> dict:
    response = await client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": text}
        ]
    )

    raw = response.choices[0].message.content.strip()
    raw = raw.replace("```json", "").replace("```", "").strip()
    return json.loads(raw)