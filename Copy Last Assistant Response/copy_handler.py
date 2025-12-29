"""
LiteLLM Clipboard Handler

A custom LiteLLM handler that intercepts assistant responses and copies the last assistant's
response to clipboard when the current request contains the string "copyy".

Usage:
1. Add this handler to your LiteLLM config.yaml:
   litellm_settings:
     callbacks: copy.copy_to_clipboard

2. Any request containing "copyy" in the current user message will trigger
   the clipboard copy functionality and return a confirmation message.

Requirements:
- pyperclip library: pip install pyperclip
"""

from litellm.integrations.custom_logger import CustomLogger
from litellm.litellm_core_utils.prompt_templates.common_utils import (
    get_last_user_message
)
import re

try:
    import pyperclip
except ImportError:
    raise ImportError("pyperclip library required. Install with: pip install pyperclip")


class ClipboardHandler(CustomLogger):
    """Custom handler that copies previous assistant response to clipboard when 'copyy' is detected in current request."""

    def extract_text_content(self, response):
        """Extract text content from either Claude Code format or plain string."""

        # Case 1: Already a plain string
        if isinstance(response, str):
            # Check if it's the Claude Code format (likely contains the structure)
            if "'type': 'text'" in response and "'text':" in response:
                try:
                    # Extract content from Claude Code format using regex
                    match = re.search(r"'text':\s*'([^']*)'", response)
                    if match:
                        return match.group(1)
                except:
                    pass
            # Return as-is if it's plain text
            return response

        # Case 2: Already a dictionary (if somehow we get dict directly)
        elif isinstance(response, dict):
            if response.get('type') == 'text':
                return response.get('text', '')

        # Case 3: List format (array of dicts)
        elif isinstance(response, list):
            # Find the first text entry
            for item in response:
                if isinstance(item, dict) and item.get('type') == 'text':
                    return item.get('text', '')

        # Fallback: convert to string
        return str(response)

    async def async_pre_call_hook(self, user_api_key_dict, cache, data: dict, call_type):
        """Intercepts requests and handles clipboard copying before LLM call."""

        # Get ONLY the current/last user message from the request
        messages = data.get("messages", [])
        current_user_content = get_last_user_message(messages)

        if not current_user_content:
            return data

        # Check if "copyy" appears ONLY in the current request (not conversation history)
        if "copyy" in current_user_content.lower():
            # Extract the last assistant response from message history
            assistant_responses = []
            for message in messages:
                if message.get("role") == "assistant":
                    content = message.get("content", "")
                    if content:  # Only add non-empty responses
                        assistant_responses.append(content)

            if len(assistant_responses) >= 1:
                # Extract clean text content (handles both Claude Code format and plain text)
                last_assistant_response = self.extract_text_content(assistant_responses[-1])

                if last_assistant_response:
                    # Copy to clipboard
                    try:
                        pyperclip.copy(last_assistant_response)
                        print(f"✅ Copied last assistant response ({len(last_assistant_response)} characters) to clipboard")
                        # Return success message that will be shown to user
                        return "✅ Copied the previous response to clipboard!"
                    except pyperclip.PyperclipException as e:
                        print(f"❌ Failed to copy to clipboard: {e}")
                        return "❌ Failed to copy to clipboard. Please try again."
                else:
                    return "⚠️ No previous assistant response found to copy."
            else:
                return "⚠️ No previous assistant responses found in conversation history."

        # Return original data to proceed with normal LLM call
        return data


# Create handler instance for LiteLLM configuration
copy_to_clipboard = ClipboardHandler()
