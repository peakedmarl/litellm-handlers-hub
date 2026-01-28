 ```litellm-handlers-hub/DeepSeek V3.2 Context Management System/deepseek_context_handler.py
"""
DeepSeek V3.2 Context Handler - v3.0 (Clean Slate)

Proper three-hook architecture for DeepSeek V3.2 reasoning content management.
Uses session-based storage with Redis primary and in-memory fallback.

Core Rules:
1. Strip reasoning on new user input (async_pre_call_hook)
2. Inject stored reasoning for tool-only turns (async_get_chat_completion_prompt)
3. Capture and store reasoning from responses (async_post_call_success_hook)
"""

import json
import logging
import os
import uuid
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import UserAPIKeyAuth
from litellm.types.utils import ModelResponse

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== Storage Abstraction ====================

class BaseStorage(ABC):
    """Abstract base for reasoning storage backends."""

    @abstractmethod
    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve stored reasoning for a session."""
        pass

    @abstractmethod
    async def set(self, session_id: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        """Store reasoning for a session with TTL."""
        pass

    @abstractmethod
    async def delete(self, session_id: str) -> None:
        """Delete stored reasoning for a session."""
        pass


class InMemoryStorage(BaseStorage):
    """In-memory storage for development/testing."""

    def __init__(self):
        self._store: Dict[str, Dict[str, Any]] = {}

    async def get(self, session_id: str) -> Optional[Dict[str, Any]]:
        return self._store.get(session_id)

    async def set(self, session_id: str, data: Dict[str, Any], ttl: int = 3600) -> None:
        # TTL is ignored in memory, but we store it for compatibility
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


# ==================== DeepSeek Context Handler ====================

class DeepSeekContextHandler(CustomLogger):
    """
    Handler for DeepSeek V3.2 context management.
    
    Implements three-hook architecture:
    1. async_pre_call_hook: Strip reasoning from messages on new user input
    2. async_get_chat_completion_prompt: Inject stored reasoning for tool-only turns
    3. async_post_call_success_hook: Capture and store reasoning from responses
    """

    def __init__(self):
        super().__init__()
        self.storage = Storage()
        logger.info("🚀 DeepSeekContextHandler initialized")

    # ==================== Utility Methods ====================

    def _get_session_id(self, data: Dict[str, Any]) -> str:
        """Extract or generate session ID from request data."""
        session_id = data.get("litellm_session_id") or data.get("litellm_trace_id")
        if not session_id:
            session_id = str(uuid.uuid4())
            data["litellm_session_id"] = session_id
            logger.debug(f"Generated new session ID: {session_id}")
        return session_id

    def _is_tool_result(self, message: Dict[str, Any]) -> bool:
        """Check if message is a tool result."""
        return message.get("role") in ["tool", "function"]

    def _is_user_message(self, message: Dict[str, Any]) -> bool:
        """Check if message is from user."""
        return message.get("role") == "user"

    def _is_assistant_message(self, message: Dict[str, Any]) -> bool:
        """Check if message is from assistant."""
        return message.get("role") == "assistant"

    def _has_reasoning_content(self, message: Dict[str, Any]) -> bool:
        """Check if message contains reasoning content."""
        return bool(message.get("reasoning_content"))

    def _strip_reasoning_from_messages(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove reasoning_content from all messages."""
        cleaned = []
        for msg in messages:
            msg_copy = msg.copy()
            if "reasoning_content" in msg_copy:
                del msg_copy["reasoning_content"]
                logger.debug(f"Stripped reasoning from {msg.get('role')} message")
            cleaned.append(msg_copy)
        return cleaned

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

    def _extract_reasoning_from_response(self, response: ModelResponse) -> Optional[str]:
        """Extract reasoning_content from model response."""
        try:
            if not isinstance(response, ModelResponse):
                return None
            if not response.choices:
                return None
            
            message = response.choices[0].message
            reasoning = getattr(message, "reasoning_content", None)
            
            if reasoning:
                logger.debug(f"Extracted reasoning: {len(reasoning)} chars")
            return reasoning
            
        except Exception as e:
            logger.warning(f"Failed to extract reasoning: {e}")
            return None

    def _inject_reasoning_into_messages(
        self, 
        messages: List[Dict[str, Any]], 
        reasoning: str
    ) -> List[Dict[str, Any]]:
        """
        Inject reasoning content into the last assistant message.
        If no assistant message exists, prepend one with reasoning.
        """
        if not messages:
            return messages

        modified = [m.copy() for m in messages]
        
        # Find last assistant message
        for i in range(len(modified) - 1, -1, -1):
            if self._is_assistant_message(modified[i]):
                modified[i]["reasoning_content"] = reasoning
                logger.debug(f"Injected reasoning into assistant message at index {i}")
                return modified
        
        # No assistant message found - prepend one
        reasoning_msg = {
            "role": "assistant",
            "content": "",
            "reasoning_content": reasoning
        }
        logger.debug("Prepended reasoning message (no assistant found)")
        return [reasoning_msg] + modified

    # ==================== Hook 1: Pre-Call ====================

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        """
        Strip reasoning content from messages when new user input is detected.
        
        This runs BEFORE the LLM call, so we clean the incoming messages.
        """
        try:
            messages = data.get("messages", [])
            if not messages:
                return data

            interaction_type = self._detect_interaction_type(messages)

            if interaction_type == "new_user":
                logger.info("🧹 New user message detected - stripping reasoning content")
                cleaned_messages = self._strip_reasoning_from_messages(messages)
                data["messages"] = cleaned_messages
                
                # Also clear any stored reasoning for this session
                session_id = self._get_session_id(data)
                await self.storage.delete(session_id)
                logger.debug(f"Cleared stored reasoning for session {session_id}")

            return data

        except Exception as e:
            logger.error(f"Error in async_pre_call_hook: {e}", exc_info=True)
            # Fail open - return original data
            return data

    # ==================== Hook 2: Get Chat Completion Prompt ====================

    async def async_get_chat_completion_prompt(
        self,
        model: str,
        messages: List[Dict[str, Any]],
        non_default_params: Dict[str, Any],
        **kwargs
    ) -> Tuple[str, List[Dict[str, Any]], Dict[str, Any]]:
        """
        Inject stored reasoning for tool-only interactions.
        
        This runs BEFORE the LLM call and can modify the messages.
        """
        try:
            # Get session ID from litellm_logging_obj if available
            litellm_logging_obj = kwargs.get("litellm_logging_obj")
            session_id = None
            
            if litellm_logging_obj and hasattr(litellm_logging_obj, "litellm_params"):
                session_id = litellm_logging_obj.litellm_params.get("litellm_session_id")
            
            if not session_id:
                # Try to extract from messages metadata or generate
                session_id = str(uuid.uuid4())

            interaction_type = self._detect_interaction_type(messages)

            if interaction_type == "tool_only":
                logger.info("🔧 Tool-only interaction - checking for stored reasoning")
                
                stored = await self.storage.get(session_id)
                if stored and stored.get("reasoning"):
                    reasoning = stored["reasoning"]
                    modified_messages = self._inject_reasoning_into_messages(messages, reasoning)
                    logger.info(f"✅ Injected stored reasoning ({len(reasoning)} chars)")
                    return model, modified_messages, non_default_params
                else:
                    logger.warning("No stored reasoning found for tool-only interaction")

            return model, messages, non_default_params

        except Exception as e:
            logger.error(f"Error in async_get_chat_completion_prompt: {e}", exc_info=True)
            # Fail open - return original
            return model, messages, non_default_params

    # ==================== Hook 3: Post-Call Success ====================

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Capture and store reasoning content from successful responses.
        
        This runs AFTER the LLM call, so we extract and store reasoning
        for potential use in the next tool-only turn.
        """
        try:
            if not isinstance(response, ModelResponse):
                return response

            messages = data.get("messages", [])
            session_id = self._get_session_id(data)

            # Extract reasoning from response
            reasoning = self._extract_reasoning_from_response(response)
            
            if reasoning:
                # Store for next turn
                await self.storage.set(session_id, {
                    "reasoning": reasoning,
                    "timestamp": str(uuid.uuid4())  # Simple timestamp placeholder
                })
                logger.info(f"💾 Stored reasoning for session {session_id[:8]}... ({len(reasoning)} chars)")
            else:
                logger.debug("No reasoning content in response")

            return response

        except Exception as e:
            logger.error(f"Error in async_post_call_success_hook: {e}", exc_info=True)
            # Fail open - return original response
            return response


# Create handler instance for use in proxy_config.yaml
deepseek_context_handler = DeepSeekContextHandler()
