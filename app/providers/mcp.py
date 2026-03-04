# This file implements the McpProvider, which connects to an MCP server using Server-Sent Events (SSE) to call 
# tools and retrieve responses.

# Currently Inactive - MCP integration is not fully set up yet, so this provider is not wired into the main application.
#  Once MCP is ready, we can switch the active provider in the configuration to use this implementation.

# By: Arsalaan Ahmad

from __future__ import annotations
import asyncio
from typing import Dict, Any, Optional

from app.providers.base import ChatProvider
from app.core.contract import ChatResult

# These imports must exist in your environment where MCP is installed
from mcp import ClientSession
from mcp.client.sse import sse_client


class McpProvider(ChatProvider):
    """
    Calls MCP tools over SSE. Keeps a single session alive.
    """
    def __init__(self, sse_url: str, tool_name: str, arg_name: str):
        self.sse_url = sse_url
        self.tool_name = tool_name
        self.arg_name = arg_name

        self._lock = asyncio.Lock()
        self._session: Optional[ClientSession] = None
        self._sse_cm = None

    async def _ensure_session(self) -> ClientSession:
        async with self._lock:
            if self._session is not None:
                return self._session

            if not self.sse_url:
                raise RuntimeError("MCP_SSE_URL is not set")

            self._sse_cm = sse_client(self.sse_url)
            read, write = await self._sse_cm.__aenter__()

            self._session = ClientSession(read, write)
            await self._session.__aenter__()
            await self._session.initialize()
            return self._session

    async def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> ChatResult:
        session = await self._ensure_session()
        result = await session.call_tool(self.tool_name, arguments={self.arg_name: message})

        # Parse content safely
        if not getattr(result, "content", None):
            return ChatResult("error", "MCP returned no content.")

        first = result.content[0]
        text = getattr(first, "text", None)
        if text is None:
            text = str(first)

        return ChatResult("success", text)

    async def close(self):
        async with self._lock:
            if self._session is not None:
                await self._session.__aexit__(None, None, None)
                self._session = None
            if self._sse_cm is not None:
                await self._sse_cm.__aexit__(None, None, None)
                self._sse_cm = None
