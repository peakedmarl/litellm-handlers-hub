"""
DeepSeek V3.2 Context Handler - v5.0 (Reasoning Chain)

Implements reasoning chain persistence for multi-turn tool use conversations.
Maintains up to 10 turns of reasoning history, injecting full chain on tool-only turns.

Session Management:
- Token-based session ID (SHA256 hash of API key token)
- Reasoning chain stored as FIFO list (max 10 entries)
- Cleared on fresh user messages

Core Rules:
1. async_pre_call_hook:
   - New user input: Strip reasoning, clear chain
   - Tool-only turns: Inject full reasoning chain into last assistant message
2. async_post_call_success_hook: Append reasoning entry to chain (non-streaming)
3. async_post_call_streaming_iterator_hook: Append reasoning entry to chain (streaming)

Reasoning Chain Entry:
{
    "turn_index": int,           # Position in conversation
    "reasoning": str,            # Reasoning content
    "has_tool_calls": bool,      # Did this turn invoke tools?
    "stored_at": str             # ISO timestamp
}
"""

import json
import logging
import os
import hashlib
from abc import ABC, abstractmethod
from datetime import datetime, timezone
from typing import Any, AsyncGenerator, Dict, List, Optional

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import UserAPIKeyAuth
from litellm.types.utils import ModelResponse, ModelResponseStream

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants
MAX_CHAIN_LENGTH = 10
REASONING_SESSION_TTL = 3600  # 1 hour


# ==================== Storage Abstraction ====================

class BaseStorage(ABC):
    """Abstract base for reasoning storage backends."""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored data for a session."""
        pass

    @abstractmethod
    async def set(self, session_id: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        """Store data for a session with TTL."""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete stored data for a session."""
        pass


class InMemoryStorage(BaseStorage):
    """In-memory storage for development/testing."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(session_id)

    async def set(self, session_id: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        data["_ttl"] = ttl
        self._store[session_id] = data

    async def delete(self, session_id: str) -> None:
        self._store.pop(session_id, None)


class RedisStorage(BaseStorage):
    """Redis-backed storage for production."""

    def __init__(self, host: str, port: int, password: Optional[str] = None, db: int = 0):
        try:
            import redis.asyncio as redis
            self._redis = redis.Redis(
                host=host,
                port=port,
                password=password,
                db=db,
                decode_responses=True
            )
            logger.info("✅ Redis storage initialized")
        except ImportError:
            logger.error("❌ redis package not installed. Run: pip install redis")
            raise

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        try:
            data = await self._redis.get(f"deepseek:reasoning:{session_id}")
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Redis get error: {e}")
            return None

    async def set(self, session_id: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        try:
            key = f"deepseek:reasoning:{session_id}"
            await self._redis.setex(key, ttl, json.dumps(data))
        except Exception as e:
            logger.error(f"Redis set error: {e}")

    async def delete(self, session_id: str) -> None:
        try:
            await self._redis.delete(f"deepseek:reasoning:{session_id}")
        except Exception as e:
            logger.error(f"Redis delete error: {e}")


class Storage:
    """Storage factory with fallback support."""

    def __init__(self):
        self._backend: BaseStorage = self._init_backend()

    def _init_backend(self) -> BaseStorage:
        """Initialize storage backend based on environment."""
        redis_host = os.getenv("REDIS_HOST")
        redis_port = os.getenv("REDIS_PORT")

        if redis_host and redis_port:
            try:
                return RedisStorage(
                    host=redis_host,
                    port=int(redis_port),
                    password=os.getenv("REDIS_PASSWORD"),
                    db=int(os.getenv("REDIS_DB", "0"))
                )
            except Exception as e:
                logger.warning(f"Failed to init Redis, falling back to memory: {e}")

        logger.info("Using in-memory storage")
        return InMemoryStorage()

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return await self._backend.get(session_id)

    async def set(self, session_id: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        await self._backend.set(session_id, data, ttl)

    async def delete(self, session_id: str) -> None:
        await self._backend.delete(session_id)

    async def append_to_chain(
        self,
        session_id: str,
        entry: Dict[str, Any],
        max_length: int = MAX_CHAIN_LENGTH
    ) -> None:
        """
        Append entry to reasoning chain, maintaining FIFO with max_length.
        Creates new chain if none exists.
        """
        try:
            stored = await self.get(session_id) or {}
            chain = stored.get("reasoning_chain", [])

            # Append new entry
            chain.append(entry)

            # Trim to max length (FIFO)
            if len(chain) > max_length:
                chain = chain[-max_length:]

            # Update stored data
            stored["reasoning_chain"] = chain
            stored["last_updated"] = datetime.now(timezone.utc).isoformat()

            await self.set(session_id, stored, REASONING_SESSION_TTL)
            logger.debug(f"Appended entry to chain for {session_id[:8]}... (len={len(chain)})")

        except Exception as e:
            logger.error(f"Failed to append to chain: {e}")


# ==================== DeepSeek Context Handler ====================

class DeepSeekContextHandler(CustomLogger):
    """
    Handler for DeepSeek V3.2 reasoning chain management.

    Maintains up to 10 turns of reasoning history, injecting full chain
    on tool-only turns so the model sees its complete thought process.
    """

    def __init__(self):
        super().__init__()
        self.storage = Storage()
        logger.info(f"🚀 DeepSeekContextHandler initialized (v5.0 - Reasoning Chain)")

    # ==================== Utility Methods ====================

    def _get_session_id(self, user_api_key_dict: UserAPIKeyAuth) -> Optional[str]:
        """Get stable session ID from user API key token."""
        token = getattr(user_api_key_dict, "token", None)
        if token:
            session_id = hashlib.sha256(token.encode()).hexdigest()[:16]
            logger.debug(f"Session ID: {session_id[:8]}...")
            return session_id
        return None

    def _is_tool_result(self, message: Dict[str, Any]) -> bool:
        """Check if message is a tool result (handles OpenAI and Anthropic formats)."""
        if message.get("role") in ["tool", "function"]:
            return True

        content = message.get("content")
        if isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and item.get("type") == "tool_result":
                    return True
        return False

    def _is_user_message(self, message: Dict[str, Any]) -> bool:
        """Check if message is from user (not a tool result)."""
        if message.get("role") != "user":
            return False
        return not self._is_tool_result(message)

    def _is_assistant_message(self, message: Dict[str, Any]) -> bool:
        """Check if message is from assistant."""
        return message.get("role") == "assistant"

    def _detect_interaction_type(self, messages: List[Dict[str, Any]]) -> str:
        """
        Detect the type of current interaction.

        Returns:
            - "new_user": Last message is from user (new turn)
            - "tool_only": Last message is tool result (continuation)
            - "other": Unknown or assistant message
        """
        if not messages:
            return "other"

        last_message = messages[-1]

        if self._is_user_message(last_message):
            return "new_user"
        elif self._is_tool_result(last_message):
            return "tool_only"
        else:
            return "other"

    def _count_assistant_turns(self, messages: List[Dict[str, Any]]) -> int:
        """Count number of assistant messages (used for turn indexing)."""
        return sum(1 for msg in messages if self._is_assistant_message(msg))

    def _strip_reasoning_from_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove reasoning_content and thinking blocks from all messages."""
        cleaned = []
        for msg in messages:
            msg_copy = msg.copy()

            # Strip top-level reasoning_content
            if "reasoning_content" in msg_copy:
                del msg_copy["reasoning_content"]
                logger.debug(f"Stripped reasoning from {msg.get('role')} message")

            # Strip thinking blocks from content list (Anthropic format)
            content = msg_copy.get("content")
            if isinstance(content, list):
                new_content = [
                    item for item in content
                    if not (isinstance(item, dict) and item.get("type") in ["thinking", "thought"])
                ]
                msg_copy["content"] = new_content

            cleaned.append(msg_copy)
        return cleaned

    def _format_reasoning_chain(self, chain: List[Dict[str, Any]]) -> str:
        """Format reasoning chain entries into concatenated string."""
        if not chain:
            return ""

        parts = []
        for entry in chain:
            turn_num = entry.get("turn_index", 0) + 1
            reasoning = entry.get("reasoning", "")
            has_tools = entry.get("has_tool_calls", False)
            tool_indicator = " [tools]" if has_tools else ""

            parts.append(f"[Turn {turn_num}{tool_indicator}]\n{reasoning}")

        return "\n\n---\n\n".join(parts)

    def _inject_reasoning_chain(
        self,
        messages: List[Dict[str, Any]],
        reasoning_chain: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Inject formatted reasoning chain into the last assistant message.
        If no assistant message exists, prepend one with reasoning.
        """
        if not messages or not reasoning_chain:
            return messages

        formatted_reasoning = self._format_reasoning_chain(reasoning_chain)
        modified = [m.copy() for m in messages]

        # Find last assistant message
        for i in range(len(modified) - 1, -1, -1):
            if self._is_assistant_message(modified[i]):
                modified[i]["reasoning_content"] = formatted_reasoning
                logger.debug(f"Injected reasoning chain ({len(reasoning_chain)} entries) at index {i}")
                return modified

        # No assistant message found - prepend one
        reasoning_msg = {
            "role": "assistant",
            "content": "",
            "reasoning_content": formatted_reasoning
        }
        logger.debug("Prepended reasoning message with chain")
        return [reasoning_msg] + modified

    def _extract_reasoning_from_response(self, response: ModelResponse) -> Optional[str]:
        """Extract reasoning_content from model response."""
        try:
            if not isinstance(response, ModelResponse) or not response.choices:
                return None

            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)

            if reasoning:
                logger.debug(f"Extracted reasoning: {len(reasoning)} chars")
            return reasoning

        except Exception as e:
            logger.warning(f"Failed to extract reasoning: {e}")
            return None

    def _detect_tool_calls_in_response(self, response: ModelResponse) -> bool:
        """Check if response contained tool calls."""
        try:
            if not isinstance(response, ModelResponse) or not response.choices:
                return False

            message = response.choices[0].message
            tool_calls = getattr(message, "tool_calls", None)
            return bool(tool_calls and len(tool_calls) > 0)

        except Exception:
            return False

    def _extract_reasoning_from_chunk(self, chunk: ModelResponseStream) -> Optional[str]:
        """Extract reasoning_content from a streaming chunk."""
        try:
            if not chunk.choices:
                return None

            delta = chunk.choices[0].delta
            reasoning = getattr(delta, "reasoning_content", None)
            return str(reasoning) if reasoning else None

        except Exception as e:
            logger.debug(f"Failed to extract reasoning from chunk: {e}")
            return None

    def _chunk_has_reasoning(self, chunk: ModelResponseStream) -> bool:
        """Check if a streaming chunk contains reasoning content."""
        try:
            if not chunk.choices:
                return False

            delta = chunk.choices[0].delta
            return hasattr(delta, "reasoning_content") and delta.reasoning_content is not None

        except Exception:
            return False

    # ==================== Hook 1: Pre-Call ====================

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        """
        Handle pre-call logic for reasoning chain management.

        - New user input: Strip reasoning, clear stored chain
        - Tool-only turns: Inject full reasoning chain
        """
        try:
            messages = data.get("messages", [])
            if not messages:
                return data

            interaction_type = self._detect_interaction_type(messages)
            session_id = self._get_session_id(user_api_key_dict)

            if not session_id:
                logger.warning("No session ID available")
                return data

            if interaction_type == "new_user":
                logger.info("🧹 New user message - stripping reasoning, clearing chain")
                cleaned_messages = self._strip_reasoning_from_messages(messages)
                data["messages"] = cleaned_messages

                await self.storage.delete(session_id)
                logger.debug(f"Cleared reasoning chain for session {session_id[:8]}...")

            elif interaction_type == "tool_only":
                logger.info("🔧 Tool-only turn - injecting reasoning chain")

                stored = await self.storage.get(session_id)
                chain = stored.get("reasoning_chain", []) if stored else []

                if chain:
                    modified_messages = self._inject_reasoning_chain(messages, chain)
                    data["messages"] = modified_messages
                    logger.info(f"✅ Injected {len(chain)} reasoning entries into tool-only turn")
                else:
                    logger.warning("⚠️ No reasoning chain found for tool-only turn")

            return data

        except Exception as e:
            logger.error(f"Error in async_pre_call_hook: {e}", exc_info=True)
            return data

    # ==================== Hook 2: Post-Call Success ====================

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Capture reasoning and append to chain (non-streaming responses).
        """
        try:
            if not isinstance(response, ModelResponse):
                return response

            session_id = self._get_session_id(user_api_key_dict)
            if not session_id:
                return response

            reasoning = self._extract_reasoning_from_response(response)
            if not reasoning:
                logger.debug("No reasoning in response")
                return response

            messages = data.get("messages", [])
            turn_index = self._count_assistant_turns(messages)
            has_tool_calls = self._detect_tool_calls_in_response(response)

            entry = {
                "turn_index": turn_index,
                "reasoning": reasoning,
                "has_tool_calls": has_tool_calls,
                "stored_at": datetime.now(timezone.utc).isoformat()
            }

            await self.storage.append_to_chain(session_id, entry)
            logger.info(f"💾 Appended reasoning entry (turn {turn_index}, tools={has_tool_calls}) to chain")

            return response

        except Exception as e:
            logger.error(f"Error in async_post_call_success_hook: {e}", exc_info=True)
            return response

    # ==================== Hook 3: Post-Call Streaming ====================

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: AsyncGenerator[ModelResponseStream, None],
        request_data: Dict[str, Any],
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Capture reasoning from streaming and append to chain.
        """
        reasoning_buffer: List[str] = []
        session_id = self._get_session_id(user_api_key_dict)
        messages = request_data.get("messages", [])
        turn_index = self._count_assistant_turns(messages)
        has_tool_calls = False

        try:
            async for chunk in response:
                # Check for tool calls in streaming chunks
                if chunk.choices and chunk.choices[0].delta:
                    delta = chunk.choices[0].delta
                    if getattr(delta, "tool_calls", None):
                        has_tool_calls = True

                # Accumulate reasoning
                if self._chunk_has_reasoning(chunk):
                    reasoning_piece = self._extract_reasoning_from_chunk(chunk)
                    if reasoning_piece:
                        reasoning_buffer.append(reasoning_piece)
                        logger.debug(f"Accumulated reasoning chunk: {len(reasoning_piece)} chars")

                yield chunk

        except Exception as e:
            logger.error(f"Error in streaming hook: {e}", exc_info=True)
            raise

        finally:
            # Store accumulated reasoning if we have any
            if reasoning_buffer and session_id:
                try:
                    full_reasoning = "".join(reasoning_buffer)
                    entry = {
                        "turn_index": turn_index,
                        "reasoning": full_reasoning,
                        "has_tool_calls": has_tool_calls,
                        "stored_at": datetime.now(timezone.utc).isoformat(),
                        "source": "streaming"
                    }
                    await self.storage.append_to_chain(session_id, entry)
                    logger.info(f"💾 Appended streaming reasoning (turn {turn_index}, tools={has_tool_calls}) to chain")
                except Exception as e:
                    logger.error(f"Failed to store streaming reasoning: {e}")


# Create handler instance for use in proxy_config.yaml
interleaved_thinking = DeepSeekContextHandler()
