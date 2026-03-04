## This file defines the core data structures for the contract between the frontend and backend.
# The ChatResult class encapsulates the response from the backend, including the kind of response, the reply text, any attachments, and optional metadata.
# By: Arsalaan Ahmad 

from __future__ import annotations
from dataclasses import dataclass, asdict
from typing import Literal, Optional, List, Dict, Any

Kind = Literal["success", "clarify", "refuse", "error"]  ## could be an Enum, but Literal is simpler for now

@dataclass
class Attachment:
    type: Literal["image", "map", "link"]
    url: str
    label: Optional[str] = None

@dataclass
class ChatResult:
    kind: Kind
    reply: str
    attachments: Optional[List[Attachment]] = None
    meta: Optional[Dict[str, Any]] = None

    def to_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # drop Nones to keep JSON clean
        return {k: v for k, v in d.items() if v is not None}
    
