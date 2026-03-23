from __future__ import annotations

"""
In-memory conversation/session state for the chat API.

Design goals:
- Keep request-level chat routing simple by centralizing per-conversation state.
- Support both forecast and modelling modes with the same state object.
- Stay deterministic and lightweight for local/dev use.

Non-goals:
- Durable persistence across process restarts.
- Distributed/multi-worker consistency guarantees.

If this service is scaled horizontally, replace this store with a shared backend
(for example Redis/Postgres) and keep the same public method contract where possible.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
import time
import logging


logger = logging.getLogger("cuwalid.session_store")


@dataclass
class ConversationState:
    """
    Lightweight in-memory conversation state.

    For forecast mode:
    - `intent` stores partially filled structured fields

    For modelling mode:
    - `history` can store recent messages for continuity

    Field notes:
    - conversation_id: caller-provided identifier for a single conversation thread.
    - mode: current route context (for example forecast or modelling).
    - tier: requested audience level for answer formatting.
    - intent: partial/complete structured slots extracted from user messages.
    - history: short rolling message history used only when helpful.
    - updated_at: unix timestamp used by stale-session pruning.
    """
    conversation_id: str
    mode: str = "forecast"
    tier: str = "general-public"
    intent: Dict[str, Any] = field(default_factory=dict)
    history: List[Dict[str, str]] = field(default_factory=list)
    updated_at: float = field(default_factory=time.time)


class SessionStore:
    """
    In-memory session / conversation store.

    Good enough for:
    - local development
    - demo deployments
    - low concurrency (~10 users)

    Not intended as a long-term persistent database.

    Concurrency note:
    - This implementation is in-process and non-transactional.
    - For multi-process/multi-instance deployments, use an external shared store.
    """

    def __init__(self):
        # conversation_id -> ConversationState
        self._store: Dict[str, ConversationState] = {}

    def get(self, conversation_id: str) -> ConversationState:
        """
        Return state for `conversation_id`, creating one lazily if needed.

        This keeps call sites simple: handlers can always call `get(...)`
        without checking existence first.
        """
        if conversation_id not in self._store:
            self._store[conversation_id] = ConversationState(conversation_id=conversation_id)
            logger.debug("session_created conversation_id=%s", conversation_id)
        return self._store[conversation_id]

    def update(
        self,
        conversation_id: str,
        *,
        mode: Optional[str] = None,
        tier: Optional[str] = None,
        intent: Optional[Dict[str, Any]] = None,
        append_history: Optional[Dict[str, str]] = None,
    ) -> ConversationState:
        """
        Patch selected fields and return updated state.

        Update semantics:
        - `mode` / `tier`: overwrite if provided.
        - `intent`: full replace (callers should merge before passing if needed).
        - `append_history`: append one message object to rolling history.

        History retention:
        - Keeps only the most recent 10 entries to avoid unbounded memory growth.
        """
        state = self.get(conversation_id)

        if mode is not None:
            state.mode = mode

        if tier is not None:
            state.tier = tier

        if intent is not None:
            state.intent = intent

        if append_history is not None:
            state.history.append(append_history)

            # Keep only the most recent 10 turns to stop memory growth
            if len(state.history) > 10:
                state.history = state.history[-10:]

        state.updated_at = time.time()
        logger.debug(
            "session_updated conversation_id=%s mode=%s tier=%s history_len=%s has_intent=%s",
            conversation_id,
            state.mode,
            state.tier,
            len(state.history),
            bool(state.intent),
        )
        return state

    def reset(self, conversation_id: str) -> None:
        """
        Delete a conversation entirely if present.

        Safe no-op when the key does not exist.
        """
        removed = self._store.pop(conversation_id, None)
        logger.info(
            "session_reset conversation_id=%s existed=%s",
            conversation_id,
            removed is not None,
        )

    def clear_intent(self, conversation_id: str) -> None:
        """
        Clear only structured forecast intent, but keep the conversation object.
        Useful after a successful forecast answer.

        This intentionally preserves mode/tier/history so the user can continue
        in the same session context.
        """
        state = self.get(conversation_id)
        state.intent = {}
        state.updated_at = time.time()
        logger.debug("session_intent_cleared conversation_id=%s", conversation_id)

    def prune_stale(self, max_age_seconds: int = 3600) -> None:
        """
        Remove stale conversations that haven't been touched recently.
        Default = 1 hour.

                Intended usage:
                - Call periodically from a lightweight scheduler or opportunistically
                    at request boundaries in low-traffic deployments.
        """
        now = time.time()
        stale_ids = [
            cid for cid, state in self._store.items()
            if now - state.updated_at > max_age_seconds
        ]
        for cid in stale_ids:
            self._store.pop(cid, None)

        if stale_ids:
            logger.info("session_pruned stale_count=%s", len(stale_ids))

    def debug_snapshot(self) -> Dict[str, Any]:
        """
        Lightweight debug view for /status or dev inspection.

        Returns a redacted shape (lengths/metadata) suitable for diagnostics
        without exposing more than necessary.
        """
        return {
            cid: {
                "mode": state.mode,
                "tier": state.tier,
                "intent": state.intent,
                "history_len": len(state.history),
                "updated_at": state.updated_at,
            }
            for cid, state in self._store.items()
        }