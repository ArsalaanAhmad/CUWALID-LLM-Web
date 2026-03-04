## This file defines the core configuration settings for the application, including which chat provider
#  to use and any relevant parameters for that provider.
## By: Arsalaan Ahmad 

import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    chat_provider: str = os.getenv("CHAT_PROVIDER", "mock")  # mock | mcp
    mcp_sse_url: str = os.getenv("MCP_SSE_URL", "")
    mcp_tool_name: str = os.getenv("MCP_TOOL_NAME", "ask_qwen")
    mcp_tool_arg: str = os.getenv("MCP_TOOL_ARG", "query")

settings = Settings()