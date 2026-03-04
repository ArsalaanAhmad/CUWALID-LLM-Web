## This file defines the abstract base class for chat providers. Any new provider (e.g., OpenAI, Azure, etc.) should
#  implement this interface to ensure compatibility with the rest of the application.
# By: Arsalaan Ahmad 

from __future__ import annotations
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from app.core.contract import ChatResult

class ChatProvider(ABC):
    @abstractmethod
    async def chat(self, message: str, context: Optional[Dict[str, Any]] = None) -> ChatResult:
        ...