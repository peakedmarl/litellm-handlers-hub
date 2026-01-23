"""
DeepSeek V3.2 Context Handler (OpenAI Format)

This handler implements context management rules for DeepSeek V3.2 model
using OpenAI-compatible format via LiteLLM endpoints.

Core Rules:
1. Preserve reasoning during tool interactions (keep all reasoning content)
2. Discard reasoning on new user input (strip all reasoning content)
3. Always preserve tool history (tool calls and results must stay intact)

Note: LiteLLM automatically converts all endpoint formats to OpenAI-compatible
format before reaching this handler, so we only need to handle OpenAI format.
"""

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import _parse_content_for_reasoning
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
    """Handler for DeepSeek V3.2 context management (OpenAI format only)."""

    def __init__(self):
        """Initialize the handler."""
        super().__init__()

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

    def _extract_reasoning(self, content: str) -> tuple[Optional[str], str]:
        """
        Extract reasoning content from OpenAI format (<think> tags).

        Returns: (reasoning_content, cleaned_content)
        """
        if not content or not isinstance(content, str):
            return None, content

        try:
            reasoning, cleaned = _parse_content_for_reasoning(content)
            return reasoning, cleaned
        except Exception as e:
            logger.warning(f"Failed to extract reasoning: {e}")
            return None, content

    # ==================== Reasoning Removal ====================

    def _remove_reasoning_from_message(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Remove reasoning content from a message."""
        new_message = message.copy()
        content = message.get("content")

        if content and isinstance(content, str):
            _, cleaned = self._extract_reasoning(content)
            new_message["content"] = cleaned

        return new_message

    # ==================== Interaction Detection ====================

    def _is_new_user_message(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if the last message is a new user message."""
        if not messages:
            return False

        last_message = messages[-1]
        return self._detect_message_type(last_message) == "user"

    def _is_tool_only_interaction(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if this is a tool-only interaction (no new user message)."""
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
            # Imperative commands
            r"you must respond with",
            r"you are required to",
            r"you need to",
            r"please respond using",

            # XML-style tool tags
            r"<edits>",
            r"<old_text",
            r"<new_text>",
            r"<file_to_edit>",
            r"<edit_description>",

            # Tool behavior mentions
            r"tool calls have been (disabled|enabled)",
            r"using the following format",
            r"respond with a series of edits",

            # Structured formatting instructions
            r"# file editing instructions",
            r"# instructions",
            r"format:\n*```",
        ]

        pattern_score = 0
        for pattern in tool_instruction_patterns:
            if re.search(pattern, content_lower, re.IGNORECASE):
                pattern_score += 1

        # If multiple patterns match, it's likely a tool instruction
        if pattern_score >= 2:
            logger.info(f"🎯 Detected tool instruction message (pattern score: {pattern_score})")
            return True

        # Contextual validation - check if next message has tool calls
        if message_index + 1 < len(messages):
            next_message = messages[message_index + 1]
            if next_message.get("role", "").lower() == "assistant":
                if next_message.get("tool_calls"):
                    logger.info("🎯 Contextual validation: next message has tool calls")
                    return True

        return False

    # ==================== Message Filtering ====================

    def _filter_messages_with_rules(self, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Apply the three core rules to filter messages.

        Rule 1: Preserve reasoning during tool interactions
        Rule 2: Discard reasoning on new user input
        Rule 3: Always preserve tool history

        Returns: Filtered messages list
        """
        if not messages:
            return messages

        # Determine interaction type
        is_new_user = self._is_new_user_message(messages)
        is_tool_only = self._is_tool_only_interaction(messages)

        if is_new_user:
            # Rule 2: Discard all reasoning content on new user input
            logger.info("New user message detected - stripping all reasoning content")
            filtered_messages = []

            for idx, message in enumerate(messages):
                message_type = self._detect_message_type(message, messages, idx)

                # Rule 3: Always preserve tool history and tool instructions
                if message_type in ["tool", "tool_instruction"]:
                    # Tool and tool instruction messages - keep as is
                    filtered_messages.append(message.copy())
                else:
                    # User and assistant messages - remove reasoning
                    filtered_message = self._remove_reasoning_from_message(message)
                    filtered_messages.append(filtered_message)

            return filtered_messages

        elif is_tool_only:
            # Rule 1: Preserve all reasoning content during tool interactions
            logger.info("Tool-only interaction detected - preserving all reasoning content")
            return messages.copy()

        else:
            # Default: preserve everything
            return messages.copy()

    # ==================== Debug Dump ====================

    def _dump_raw_conversation_turn(self, messages: List[Dict[str, Any]]) -> None:
        """
        Dump raw conversation turns to JSON for debugging tool message structures.

        This helps identify edge cases like Zed Editor's edit_file tool that
        may appear as user messages instead of tool/function roles.
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
                "message_count": len(messages),
                "messages": []
            }

            # Add message type detection for each message
            for idx, message in enumerate(messages):
                message_copy = message.copy()
                message_type = self._detect_message_type(message, messages, idx)

                # Add debug metadata including tool instruction flag
                is_tool_instruction = (message_type == "tool_instruction")
                message_copy["_debug"] = {
                    "index": idx,
                    "detected_type": message_type,
                    "role": message.get("role", "unknown"),
                    "has_tool_calls": self._has_tool_calls(message),
                    "is_tool_instruction": is_tool_instruction
                }

                dump_data["messages"].append(message_copy)

            # Write to JSON file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dump_data, f, indent=2, ensure_ascii=False)

            logger.info(f"🔍 Dumped raw conversation to: {filename}")

        except Exception as e:
            logger.error(f"Failed to dump conversation to JSON: {e}", exc_info=True)

    # ==================== Main Hook ====================

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        """
        Main hook that filters messages before they're sent to the model.

        Applies the three core rules using OpenAI format (LiteLLM converts all
        formats to OpenAI-compatible before reaching this handler).
        """
        try:
            messages = data.get("messages", [])

            if not messages:
                logger.debug("No messages to process")
                return data

            logger.info("Processing OpenAI format request")

            # Dump raw conversation for debugging tool structures
            self._dump_raw_conversation_turn(messages)

            # Apply filtering rules
            filtered_messages = self._filter_messages_with_rules(messages)

            # Update data with filtered messages
            data["messages"] = filtered_messages

            logger.info(f"Filtered {len(messages)} messages -> {len(filtered_messages)} messages")

        except Exception as e:
            # Log error and continue with original messages
            logger.error(f"Error during message filtering: {e}", exc_info=True)
            logger.warning("Continuing with original messages due to error")

        return data


# Create handler instance for use in proxy_config.yaml
deepseek_context_handler = DeepSeekContextHandler()