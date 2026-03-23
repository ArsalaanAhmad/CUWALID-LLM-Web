# This File is not currently in use because we're using OpenRouter.
# This file implements the MCP extractor, which connects to an MCP server using Server-Sent Events (SSE) to call an extraction tool.

import asyncio
import json
from mcp import ClientSession
from mcp.client.sse import sse_client

MCP_URL = os.getenv("MCP_URL", "http://127.0.0.1:8000/sse")

EXTRACTION_SCHEMA = {
    "country": "string or null",
    "location": "string or null",
    "season": "string or null",
    "year": "integer or null",
    "variable": "string or null",
    "language": "string or null"
}

async def extract_intent_via_mcp(text: str) -> dict:
    async with sse_client(MCP_URL) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()

            result = await session.call_tool(
                "extract_to_json",
                arguments={
                    "text": text,
                    "schema": EXTRACTION_SCHEMA,
                    "additional_instructions": (
                        "Extract hydrological forecast request information. "
                        "Allowed variables include flood, crop, pasture, surface_water, groundwater. "
                        "If information is missing, return null for that field. "
                        "Return only JSON."
                    )
                }
            )

            raw = result.content[0].text.strip()

            if raw.startswith("Error:"):
                raise ValueError(raw)

            return json.loads(raw)