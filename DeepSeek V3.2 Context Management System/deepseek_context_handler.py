"""
DeepSeek V3.2 Dual-Format Context Handler

This handler implements context management rules for DeepSeek V3.2 model
across both OpenAI-compatible and Anthropic-native LiteLLM endpoints:

Core Rules:
1. Preserve reasoning during tool interactions (keep all reasoning content)
2. Discard reasoning on new user input (strip all reasoning content)
3. Always preserve tool history (tool calls and results must stay intact)

Supported Formats:
- OpenAI-compatible: /v1/chat/completions (DeepSeek, OpenAI, etc.)
- Anthropic-native: /v1/messages (Claude models)
"""

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import _parse_content_for_reasoning
from typing import List, Dict, Any, Optional, Union
import logging
import json
import os
import re
from datetime import datetime

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DualFormatContextHandler(CustomLogger):
    """Handler for DeepSeek V3.2 context management with dual endpoint support."""

    def __init__(self):
        """Initialize the handler."""
        super().__init__()

    # ==================== Endpoint Detection ====================

    def _is_anthropic_request(self, data: Dict[str, Any]) -> bool:
        """
        Detect if the request targets the Anthropic endpoint.

        Checks:
        - Model name patterns (claude-3, claude-4, etc.)
        - Anthropic-specific headers
        - Provider metadata
        """
        # Check model name
        model = data.get("model", "").lower()
        if "claude" in model:
            return True

        # Check headers for Anthropic-specific indicators
        headers = data.get("headers", {})
        if "anthropic-version" in headers or "anthropic-beta" in headers:
            return True

        # Check provider metadata
        metadata = data.get("metadata", {})
        if metadata.get("provider") == "anthropic":
            return True

        return False

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

        # Anthropic format: tool results embedded in user messages
        if role == "user":
            content = message.get("content")
            if isinstance(content, list):
                for block in content:
                    if block.get("type") == "tool_result":
                        return "tool"

        # STEP 2.2: Check for tool instruction messages masquerading as user
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

    def _has_tool_calls_openai(self, message: Dict[str, Any]) -> bool:
        """Check if an OpenAI-format assistant message contains tool calls."""
        return bool(message.get("tool_calls"))

    def _has_tool_calls_anthropic(self, message: Dict[str, Any]) -> bool:
        """Check if an Anthropic-format assistant message contains tool calls."""
        content = message.get("content")
        if not isinstance(content, list):
            return False

        return any(block.get("type") == "tool_use" for block in content)

    def _has_tool_calls(self, message: Dict[str, Any], is_anthropic: bool) -> bool:
        """Unified tool call detection across formats."""
        if is_anthropic:
            return self._has_tool_calls_anthropic(message)
        return self._has_tool_calls_openai(message)

    # ==================== Reasoning Extraction ====================

    def _extract_reasoning_openai(self, content: str) -> tuple[Optional[str], str]:
        """
        Extract reasoning content from OpenAI format ( tags).

        Returns: (reasoning_content, cleaned_content)
        """
        if not content or not isinstance(content, str):
            return None, content

        try:
            reasoning, cleaned = _parse_content_for_reasoning(content)
            return reasoning, cleaned
        except Exception as e:
            logger.warning(f"Failed to extract OpenAI reasoning: {e}")
            return None, content

    def _extract_reasoning_anthropic(self, content: Union[str, List[Dict[str, Any]]]) -> tuple[Optional[str], Union[str, List[Dict[str, Any]]]]:
        """
        Extract reasoning content from Anthropic format (thinking blocks).

        Returns: (reasoning_content, cleaned_content)
        """
        if not content:
            return None, content

        # Handle string content (simple case)
        if isinstance(content, str):
            return None, content

        # Handle content blocks array
        if not isinstance(content, list):
            return None, content

        reasoning_parts = []
        cleaned_blocks = []

        for block in content:
            block_type = block.get("type")

            # Extract reasoning from thinking blocks
            if block_type in ["thinking", "redacted_thinking"]:
                thinking_content = block.get("thinking", "")
                if thinking_content:
                    reasoning_parts.append(thinking_content)
            else:
                # Preserve all other blocks
                cleaned_blocks.append(block)

        reasoning = "\n".join(reasoning_parts) if reasoning_parts else None
        cleaned_content = cleaned_blocks if cleaned_blocks else content

        return reasoning, cleaned_content

    def _extract_reasoning_unified(self, message: Dict[str, Any], is_anthropic: bool) -> tuple[Optional[str], Any]:
        """
        Unified reasoning extraction across formats.

        Returns: (reasoning_content, cleaned_content)
        """
        content = message.get("content")

        if is_anthropic:
            return self._extract_reasoning_anthropic(content)
        else:
            return self._extract_reasoning_openai(content)

    # ==================== Reasoning Removal ====================

    def _remove_reasoning_openai(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Remove reasoning content from OpenAI-format message."""
        new_message = message.copy()
        content = message.get("content")

        if content and isinstance(content, str):
            _, cleaned = self._extract_reasoning_openai(content)
            new_message["content"] = cleaned

        return new_message

    def _remove_reasoning_anthropic(self, message: Dict[str, Any]) -> Dict[str, Any]:
        """Remove reasoning content from Anthropic-format message."""
        new_message = message.copy()
        content = message.get("content")

        if content:
            _, cleaned = self._extract_reasoning_anthropic(content)
            new_message["content"] = cleaned

        return new_message

    def _remove_reasoning_from_message(self, message: Dict[str, Any], is_anthropic: bool) -> Dict[str, Any]:
        """Unified reasoning removal across formats."""
        if is_anthropic:
            return self._remove_reasoning_anthropic(message)
        return self._remove_reasoning_openai(message)

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

        # STEP 1.2: Pattern matching with signature indicators
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

        # STEP 1.3: Contextual validation - check if next message has tool calls
        if message_index + 1 < len(messages):
            next_message = messages[message_index + 1]
            if next_message.get("role", "").lower() == "assistant":
                # Check if assistant has tool calls
                if next_message.get("tool_calls"):
                    logger.info("🎯 Contextual validation: next message has tool calls")
                    return True

        return False

    # ==================== Message Filtering ====================

    def _filter_messages_with_rules(self, messages: List[Dict[str, Any]], is_anthropic: bool) -> List[Dict[str, Any]]:
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

                # STEP 2.3: Always preserve tool history and tool instructions (Rule 3)
                if message_type in ["tool", "tool_instruction"]:
                    # Tool and tool instruction messages - keep as is
                    filtered_messages.append(message.copy())
                else:
                    # User and assistant messages - remove reasoning
                    filtered_message = self._remove_reasoning_from_message(message, is_anthropic)
                    filtered_messages.append(filtered_message)

            return filtered_messages

        elif is_tool_only:
            # Rule 1: Preserve all reasoning content during tool interactions
            logger.info("Tool-only interaction detected - preserving all reasoning content")
            return messages.copy()

        else:
            # Default: preserve everything
            return messages.copy()

    # ==================== Main Hook ====================

    def _dump_raw_conversation_turn(self, messages: List[Dict[str, Any]], is_anthropic: bool) -> None:
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
            format_name = "anthropic" if is_anthropic else "openai"
            filename = f"{debug_dir}/conversation_{format_name}_{timestamp}.json"

            # Prepare dump data with metadata
            dump_data = {
                "timestamp": datetime.now().isoformat(),
                "format": format_name,
                "message_count": len(messages),
                "messages": []
            }

            # Add message type detection for each message
            for idx, message in enumerate(messages):
                message_copy = message.copy()
                message_type = self._detect_message_type(message, messages, idx)

                # STEP 2.4: Add debug metadata including tool instruction flag
                is_tool_instruction = (message_type == "tool_instruction")
                message_copy["_debug"] = {
                    "index": idx,
                    "detected_type": message_type,
                    "role": message.get("role", "unknown"),
                    "has_tool_calls": self._has_tool_calls(message, is_anthropic),
                    "is_tool_instruction": is_tool_instruction
                }

                dump_data["messages"].append(message_copy)

            # Write to JSON file
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(dump_data, f, indent=2, ensure_ascii=False)

            logger.info(f"🔍 Dumped raw conversation to: {filename}")

        except Exception as e:
            logger.error(f"Failed to dump conversation to JSON: {e}", exc_info=True)

    async def async_pre_call_hook(
        self,
        user_api_key_dict: Any,
        cache: Any,
        data: Dict[str, Any],
        call_type: str,
    ) -> Dict[str, Any]:
        """
        Main hook that filters messages before they're sent to the model.

        Applies the three core rules across both OpenAI and Anthropic formats.
        """
        try:
            messages = data.get("messages", [])

            if not messages:
                logger.debug("No messages to process")
                return data

            # Detect endpoint format
            is_anthropic = self._is_anthropic_request(data)
            format_name = "Anthropic" if is_anthropic else "OpenAI"
            logger.info(f"Processing {format_name} format request")

            # Dump raw conversation for debugging tool structures
            self._dump_raw_conversation_turn(messages, is_anthropic)

            # Apply filtering rules
            filtered_messages = self._filter_messages_with_rules(messages, is_anthropic)

            # Update data with filtered messages
            data["messages"] = filtered_messages

            logger.info(f"Filtered {len(messages)} messages -> {len(filtered_messages)} messages")

        except Exception as e:
            # Log error and continue with original messages
            logger.error(f"Error during message filtering: {e}", exc_info=True)
            logger.warning("Continuing with original messages due to error")

        return data


# Create handler instance for use in proxy_config.yaml
dual_format_handler = DualFormatContextHandler()
