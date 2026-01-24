"""
DeepSeek V3.2 Context Handler (OpenAI Format) - v2.0

This handler implements context management rules for DeepSeek V3.2 model
using OpenAI-compatible format via LiteLLM endpoints.

Core Rules:
1. Preserve reasoning during tool interactions (keep all reasoning content)
2. Discard reasoning on new user input (strip all reasoning content)
3. Always preserve tool history (tool calls and results must stay intact)

Key Implementation Details:
- Uses async_post_call_success_hook to inspect responses AFTER model generation
- Accesses reasoning_content directly from response.choices[0].message.reasoning_content
- Does NOT modify raw messages - reasoning exists naturally in the response structure
"""

from litellm.integrations.custom_logger import CustomLogger
from litellm.proxy.proxy_server import UserAPIKeyAuth
from litellm.types.utils import ModelResponse, Choices
from typing import List, Dict, Any, Optional
import logging
import json
import os
import re
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeepSeekContextHandler(CustomLogger):
    """Handler for DeepSeek V3.2 context management using post-call hooks."""

    def __init__(self):
        """Initialize the handler."""
        super().__init__()
        # Store reasoning content from previous turns to rebuild conversation history
        self.conversation_history: List[Dict[str, Any]] = []

    # ==================== Message Type Detection ====================

    def _detect_message_type(self, message: Dict[str, Any], messages: Optional[List[Dict[str, Any]]] = None, message_index: Optional[int] = None) -> str:
        """
        Detect the type of a message based on its role and content.

        Returns: "user", "assistant", "tool", "tool_instruction", or "unknown"
        """
        role = message.get("role", "").lower()

        # OpenAI format: explicit tool/function roles
        if role in ["tool", "function"]:
            return "tool"

        # Check for tool instruction messages masquerading as user
        if role == "user" and messages is not None and message_index is not None:
            if self._is_tool_instruction_message(message, messages, message_index):
                return "tool_instruction"

        # Standard roles
        if role == "user":
            return "user"
        elif role == "assistant":
            return "assistant"

        return "unknown"

    # ==================== Tool Call Detection ====================

    def _has_tool_calls(self, message: Dict[str, Any]) -> bool:
        """Check if an assistant message contains tool calls (OpenAI format)."""
        return bool(message.get("tool_calls"))

    # ==================== Reasoning Extraction ====================

    def _extract_reasoning_from_response(self, response: ModelResponse) -> Optional[str]:
        """
        Extract reasoning content from a successful ModelResponse.
        
        Returns: reasoning_content string or None
        """
        try:
            if not isinstance(response, ModelResponse):
                return None
                
            if not response.choices:
                return None
                
            message = response.choices[0].message
            return getattr(message, 'reasoning_content', None)
            
        except Exception as e:
            logger.warning(f"Failed to extract reasoning from response: {e}")
            return None

    # ==================== Interaction Detection ====================

    def _is_new_user_message(self, data: Dict[str, Any]) -> bool:
        """
        Check if the current request contains a new user message.
        This is determined by examining the request data before model generation.
        """
        messages = data.get("messages", [])
        if not messages:
            return False

        last_message = messages[-1]
        return self._detect_message_type(last_message) == "user"

    def _is_tool_only_interaction(self, data: Dict[str, Any]) -> bool:
        """
        Check if this is a tool-only interaction (no new user message).
        This occurs when the last message is a tool result.
        """
        messages = data.get("messages", [])
        if not messages:
            return False

        last_message = messages[-1]
        return self._detect_message_type(last_message) == "tool"

    # ==================== Tool Instruction Detection ====================

    def _is_tool_instruction_message(self, message: Dict[str, Any], messages: List[Dict[str, Any]], message_index: int) -> bool:
        """
        Detect if a user message is actually a tool instruction masquerading as user content.

        This catches edge cases like Zed Editor's edit_file tool that sends instructions
        in user role format before actual tool calls.

        Returns: True if this is a tool instruction, False otherwise
        """
        # Only check user messages
        if message.get("role", "").lower() != "user":
            return False

        content = message.get("content", "")
        if not content or not isinstance(content, str):
            return False

        content_lower = content.lower()

        # Pattern matching with signature indicators
        tool_instruction_patterns = [
            # XML-style tool tags
            r"<edits>",
            r"<old_text",
            r"<new_text>",
            r"<file_to_edit>",
            r"<edit_description>",
        ]

        pattern_score = 0
        for pattern in tool_instruction_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                pattern_score += 1

        # If multiple patterns match, it's likely a tool instruction
        if pattern_score >= 2:
            logger.info(f"🎯 Detected tool instruction message (pattern score: {pattern_score})")
            return True

        return False

    # ==================== Debug Dump ====================

    def _dump_raw_conversation_turn(self, data: Dict[str, Any], response: ModelResponse) -> None:
        """
        Dump raw conversation turns to JSON for debugging.
        
        This helps identify edge cases and verify that reasoning_content is properly
        structured in the response object.
        """
        try:
            # Create debug directory if it doesn't exist
            debug_dir = "debug_conversation_dumps"
            os.makedirs(debug_dir, exist_ok=True)

            # Generate timestamped filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
            filename = f"{debug_dir}/conversation_{timestamp}.json"

            # Prepare dump data with metadata
            dump_data = {
                "timestamp": datetime.now().isoformat(),
                "request_messages_count": len(data.get("messages", [])),
                "response_type": type(response).__name__,
                "messages": []
            }

            # Add request messages with type detection
            messages = data.get("messages", [])
            for idx, message in enumerate(messages):
                message_copy = message.copy()
                message_type = self._detect_message_type(message, messages, idx)
                
                message_copy["_debug"] = {
                    "index": idx,
                    "detected_type": message_type,
                    "role": message.get("role", "unknown"),
                    "has_tool_calls": self._has_tool_calls(message),
                    "is_tool_instruction": (message_type == "tool_instruction"),
                }
                
                dump_data["messages"].append(message_copy)

            # Add response information
            if isinstance(response, ModelResponse) and response.choices:
                response_message = response.choices[0].message
                dump_data["response"] = {
                    "has_reasoning_content": hasattr(response_message, 'reasoning_content'),
                    "reasoning_content": getattr(response_message, 'reasoning_content', None),
                    "content": getattr(response_message, 'content', None),
                    "tool_calls": getattr(response_message, 'tool_calls', None),
                }

            # Write to JSON file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dump_data, f, indent=2, ensure_ascii=False)

            logger.info(f"🔍 Dumped conversation to: {filename}")

        except Exception as e:
            logger.error(f"Failed to dump conversation to JSON: {e}", exc_info=True)

    # ==================== Main Hook: async_post_call_success_hook ====================

    async def async_post_call_success_hook(
        self,
        data: Dict[str, Any],
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
    ) -> Any:
        """
        Post-call hook that manages reasoning content based on interaction type.
        
        This hook runs AFTER a successful LLM API call and implements the three core rules:
        1. Preserve reasoning during tool interactions
        2. Discard reasoning on new user input
        3. Always preserve tool history
        
        Args:
            data: Original request payload (model, messages, params, etc.)
            user_api_key_dict: Auth context for the calling user/API key
            response: Raw response object from the underlying LLM provider
            
        Returns:
            Modified response object with reasoning content managed according to the rules
        """
        try:
            # Only process ModelResponse objects
            if not isinstance(response, ModelResponse):
                logger.debug("Response is not ModelResponse, skipping processing")
                return response

            messages = data.get("messages", [])
            if not messages:
                logger.debug("No messages in request, skipping processing")
                return response

            logger.info("Processing successful response for reasoning management")

            # Dump conversation for debugging
            self._dump_raw_conversation_turn(data, response)

            # Extract reasoning from the response
            reasoning_content = self._extract_reasoning_from_response(response)
            
            # Determine interaction type
            is_new_user = self._is_new_user_message(data)
            is_tool_only = self._is_tool_only_interaction(data)

            # Apply rules based on interaction type
            if is_new_user:
                # Rule 2: Discard reasoning on new user input
                logger.info("New user message detected - discarding reasoning content")
                
                # Remove reasoning_content from response if it exists
                if hasattr(response.choices[0].message, 'reasoning_content'):
                    response.choices[0].message.reasoning_content = None
                    
            elif is_tool_only:
                # Rule 1: Preserve reasoning during tool interactions
                logger.info("Tool-only interaction detected - preserving reasoning content")
                
                # Reasoning is already in the response, no action needed
                if reasoning_content:
                    logger.debug(f"Preserved reasoning content: {len(reasoning_content)} chars")
                    
            else:
                # Default: preserve everything
                logger.debug("Default interaction - preserving all content")
                
            # Rule 3: Always preserve tool history is handled automatically by LiteLLM
            # Tool calls and results remain in the conversation history
            
            logger.info(f"Response processed. Reasoning preserved: {is_tool_only and reasoning_content is not None}")

        except Exception as e:
            # Log error but don't break the response
            logger.error(f"Error in async_post_call_success_hook: {e}", exc_info=True)
            logger.warning("Continuing with original response due to error")

        return response


# Create handler instance for use in proxy_config.yaml
interleaved_thinking = DeepSeekContextHandler()