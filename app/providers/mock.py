## This file implements a MockProvider for testing the UI and web adapter contract without needing a real backend integration.
## By: Arsalaan Ahmad (12-02-2026)


from __future__ import annotations
from typing import Dict, Any, Optional
from app.providers.base import ChatProvider
from app.core.contract import ChatResult

class MockProvider(ChatProvider):
    async def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> ChatResult:
        m = message.strip()
        ml = m.lower()

        if not m:
            return ChatResult("clarify", "Please enter a question.")

        # Example guardrails (UI-level)
        if any(x in ml for x in ["should we", "tell me what to do", "decide", "recommend a policy"]):
            return ChatResult(
                "refuse",
                "I can’t make operational or policy decisions. If you share location + season, I can summarise forecast implications and key considerations."
            )

        if "this season" in ml and not any(x in ml for x in ["ethiopia", "somalia", "kenya", "region", "district", "county"]):
            return ChatResult(
                "clarify",
                "Which location and which season/time period are you asking about?"
            )

        return ChatResult(
            "success",
            "Mock response (UI validation): MCP integration isn’t connected yet. This placeholder confirms the web adapter + UI contract are working."
        )
