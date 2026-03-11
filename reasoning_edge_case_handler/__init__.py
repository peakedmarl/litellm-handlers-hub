 ```py /Users/oixande/litellm-handlers-hub/reasoning_edge_case_handler/__init__.py
"""
ReasoningEdgeCaseHandler - LiteLLM Proxy Custom Handler

Detects when reasoning models dump their answer into `reasoning_content`
but leave `content` empty, then triggers a follow-up "Go ahead!" nudge.

Uses persistent in-memory storage keyed by API token to track state
across requests and prevent duplicate nudges.

Storage Schema:
{
    "edge_case_detected": bool,      # Saw reasoning-only response
    "recovery_nudge_triggered": bool, # Already sent recovery nudge
    "last_detection_at": str,        # ISO timestamp for TTL
    "original_call_id": str          # Reference to original request
}

Session TTL: 5 minutes (300 seconds)
"""

import hashlib
import copy
from abc import ABC, abstractmethod
from datetime import datetime, timezone, timedelta
from typing import Any, AsyncGenerator, Dict, Optional

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponseStream
from litellm.proxy.proxy_server import UserAPIKeyAuth


# ==================== Storage Layer ====================

SESSION_TTL_SECONDS = 300  # 5 minutes


class BaseStorage(ABC):
    """Abstract base for storage backends."""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored state for a session."""
        pass

    @abstractmethod
    async def set(self, session_id: str, data: Dict[str, Any]) -> None:
        """Store state for a session with implicit TTL handling."""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete stored state for a session."""
        pass


class InMemoryStorage(BaseStorage):
    """In-memory storage with TTL-based expiration."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    def _is_expired(self, data: Dict[str, Any]) -> bool:
        """Check if stored data has exceeded TTL."""
        stored_at = data.get("last_detection_at")
        if not stored_at:
            return True

        try:
            stored_time = datetime.fromisoformat(stored_at)
            expiry_time = stored_time + timedelta(seconds=SESSION_TTL_SECONDS)
            return datetime.now(timezone.utc) > expiry_time
        except (ValueError, TypeError):
            return True

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        data = self._store.get(session_id)

        if data is None:
            return None

        # Auto-expire stale sessions
        if self._is_expired(data):
            self._store.pop(session_id, None)
            return None

        return data

    async def set(self, session_id: str, data: Dict[str, Any]) -> None:
        # Always update timestamp on write
        data["last_detection_at"] = datetime.now(timezone.utc).isoformat()
        self._store[session_id] = data

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class Storage:
    """Storage factory — in-memory only for this implementation."""

    def __init__(self):
        self._backend = InMemoryStorage()

    async def get_session(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Get session state, returning None if expired or missing."""
        return await self._backend.get(session_id)

    async def mark_edge_case_detected(
        self,
        session_id: str,
        original_call_id: str
    ) -> None:
        """Mark that we detected a reasoning-only edge case."""
        await self._backend.set(session_id, {
            "edge_case_detected": True,
            "recovery_nudge_triggered": False,
            "original_call_id": original_call_id
        })

    async def mark_recovery_nudge_triggered(self, session_id: str) -> None:
        """Mark that we've sent the recovery nudge."""
        current = await self._backend.get(session_id) or {}
        current["recovery_nudge_triggered"] = True
        await self._backend.set(session_id, current)

    async def should_skip_processing(self, session_id: str) -> bool:
        """
        Check if we should skip processing this request.
        Returns True if we already triggered a recovery nudge.
        """
        session = await self._backend.get(session_id)
        if not session:
            return False
        return session.get("recovery_nudge_triggered", False)

    async def clear_session(self, session_id: str) -> None:
        """Clear session state after a normal response."""
        await self._backend.delete(session_id)


# ==================== Handler Implementation ====================

class ReasoningEdgeCaseHandler(CustomLogger):
    """
    Detects reasoning-only responses and triggers recovery nudges.

    Uses persistent storage to track state across requests and
    prevent infinite loops or duplicate nudges.
    """

    def __init__(self):
        super().__init__()
        self.storage = Storage()

    def _get_session_id(self, user_api_key_dict: UserAPIKeyAuth) -> Optional[str]:
        """Generate stable session ID from API key token."""
        token = getattr(user_api_key_dict, "token", None)
        if not token:
            return None
        # Use first 16 chars of SHA256 hash for compact but unique IDs
        return hashlib.sha256(token.encode()).hexdigest()[:16]

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: Dict[str, Any],
        call_type: Any,
    ) -> Optional[Any]:
        """
        Pre-call hook — currently a no-op.

        All state management happens in the streaming hook where
        we have access to the full response content.
        """
        return data

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[ModelResponseStream, None],
        request_data: Dict[str, Any],
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Streaming hook — detects edge case and triggers recovery.

        Flow:
        1. Collect all chunks while yielding to client
        2. Analyze if reasoning_content present but content empty
        3. If edge case detected AND we haven't nudged yet → trigger follow-up
        4. If normal response (both present) → clear any stale state
        """
        call_id = request_data.get("litellm_call_id", "unknown")
        model = request_data.get("model", "unknown")
        session_id = self._get_session_id(user_api_key_dict)

        # Can't track state without session ID
        if not session_id:
            async for chunk in response:
                yield chunk
            return

        # Check if we already triggered a nudge for this session
        should_skip = await self.storage.should_skip_processing(session_id)
        if should_skip:
            async for chunk in response:
                yield chunk
            return

        # Collect chunks and analyze content
        chunks = []
        has_reasoning = False
        has_content = False

        async for chunk in response:
            chunks.append(chunk)

            if chunk.choices:
                delta = chunk.choices[0].delta

                # Check for reasoning content
                if getattr(delta, "reasoning_content", None):
                    has_reasoning = True

                # Check for regular content
                if getattr(delta, "content", None):
                    has_content = True

            yield chunk

        # Post-stream analysis
        content_empty = not has_content
        edge_case_detected = has_reasoning and content_empty

        if not edge_case_detected:
            # Normal response — clear any stale state
            await self.storage.clear_session(session_id)
            return

        # Edge case detected — check if we should trigger recovery
        session = await self.storage.get_session(session_id)
        already_nudged = session.get("recovery_nudge_triggered", False) if session else False

        if already_nudged:
            # We already nudged for this session, don't loop
            return

        # Mark detection and trigger recovery nudge
        await self.storage.mark_edge_case_detected(session_id, call_id)

        # Build follow-up request
        original_messages = request_data.get("messages", [])
        follow_up_messages = original_messages + [{"role": "user", "content": "Go ahead!"}]

        follow_up_params = {
            "model": model,
            "messages": follow_up_messages,
            "stream": False,
            "temperature": request_data.get("temperature", 0.7),
            "max_tokens": request_data.get("max_tokens"),
        }

        # Copy optional params if present
        for param in ["top_p", "presence_penalty", "frequency_penalty", "stop"]:
            if param in request_data:
                follow_up_params[param] = request_data[param]

        try:
            # Make follow-up call using direct litellm API
            follow_up_response = await litellm.acompletion(**follow_up_params)

            # Mark that we triggered the nudge to prevent duplicates
            await self.storage.mark_recovery_nudge_triggered(session_id)

            # Extract and stream the follow-up content
            if follow_up_response.choices:
                message = follow_up_response.choices[0].message
                content = getattr(message, "content", "")

                # Create synthetic chunk from last original chunk
                if chunks:
                    template_chunk = None
                    for c in reversed(chunks):
                        if getattr(c, "choices", None):
                            template_chunk = copy.deepcopy(c)
                            break

                    if template_chunk and template_chunk.choices:
                        from litellm.types.utils import Delta
                        template_chunk.choices[0].delta = Delta(content=content)
                        yield template_chunk

        except Exception:
            # If follow-up fails, we've already yielded original chunks
            # State remains marked as detected but not nudged — allows retry
            pass


# Singleton instance for LiteLLM proxy
handler_instance = ReasoningEdgeCaseHandler()
