"""
ReasoningEdgeCaseHandler - LiteLLM Proxy Custom Handler

Detects when reasoning models (like DeepSeek) dump their answer into
`reasoning_content` but leave `content` empty, then triggers a follow-up
"Go ahead!" nudge to get the proper answer format.

Usage:
    Add to proxy_config.yaml:
    litellm_settings:
        callbacks: reasoning_edge_case_handler.handler_instance
"""

import litellm
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse
from litellm.proxy.proxy_server import UserAPIKeyAuth
from typing import Optional, Union, Dict, Any


class ReasoningEdgeCaseHandler(CustomLogger):
    """
    LiteLLM Proxy handler that detects and fixes the reasoning edge case.

    When a model puts its response in reasoning_content but leaves content
    empty, this handler triggers a follow-up call with a nudge message.
    """

    def __init__(self):
        super().__init__()

    def _is_reasoning_edge_case(self, response: ModelResponse) -> bool:
        """
        Detects when model put answer in reasoning_content but left content empty.

        Args:
            response: The ModelResponse from the LLM call

        Returns:
            True if reasoning exists but content is empty/blank
        """
        if not response.choices:
            return False

        message = response.choices[0].message

        # Edge case: reasoning exists but final content is empty/blank
        has_reasoning = bool(
            getattr(message, 'reasoning_content', None) and
            message.reasoning_content.strip()
        )
        content_empty = not bool(
            getattr(message, 'content', None) and
            message.content.strip()
        )

        return has_reasoning and content_empty

    def _should_process_response(self, data: dict) -> bool:
        """
        Check if this response needs processing or is already a follow-up.

        Uses metadata to track retry count and prevent infinite loops.

        Args:
            data: The request data dict containing metadata

        Returns:
            True if we should process this response, False if it's a follow-up
        """
        metadata = data.get("metadata", {})
        retry_count = metadata.get("_reasoning_retry_count", 0)

        # Hard cap at 1 retry - never process our own follow-up responses
        if retry_count >= 1:
            return False

        return True

    async def async_pre_call_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        cache: Any,
        data: dict,
        call_type: Any,
    ) -> Optional[Union[dict, str, Exception]]:
        """
        Initialize state in metadata before the LLM call.

        This state round-trips to async_post_call_success_hook via the
        same data dict reference.
        """
        # Ensure metadata exists
        data["metadata"] = data.get("metadata", {})

        # Initialize retry tracking - critical for recursion prevention
        data["metadata"]["_reasoning_retry_count"] = 0
        data["metadata"]["_reasoning_original_request_id"] = data.get("litellm_call_id")

        return data

    async def async_post_call_success_hook(
        self,
        data: dict,
        user_api_key_dict: UserAPIKeyAuth,
        response: ModelResponse,
    ) -> Optional[ModelResponse]:
        """
        Check for reasoning edge case and trigger follow-up if needed.

        If the model dumped its answer into reasoning_content but left
        content empty, we make a follow-up call with a "Go ahead!" nudge.
        """
        # GUARD CLAUSE: Skip if this is already a follow-up response
        if not self._should_process_response(data):
            return response

        # Check for the edge case
        if not self._is_reasoning_edge_case(response):
            return response

        # EDGE CASE DETECTED - Mark as processed to prevent recursion
        metadata = data.get("metadata", {})
        metadata["_reasoning_retry_count"] = 1

        # Extract the reasoning content
        message = response.choices[0].message
        reasoning = getattr(message, 'reasoning_content', '')

        # Build follow-up messages with the nudge
        original_messages = data.get("messages", [])

        follow_up_messages = original_messages + [
            {
                "role": "assistant",
                "content": f"<thinking>{reasoning}</thinking>"
            },
            {
                "role": "user",
                "content": "Go ahead!"
            }
        ]

        # Prepare follow-up request parameters
        # Preserve original settings but force non-streaming for processing
        follow_up_params = {
            "model": data.get("model"),
            "messages": follow_up_messages,
            "stream": False,  # Force non-streaming for consistent processing
            "temperature": data.get("temperature", 0.7),
            "max_tokens": data.get("max_tokens"),
        }

        # Copy other relevant params if they exist
        for param in ["top_p", "presence_penalty", "frequency_penalty", "stop"]:
            if param in data:
                follow_up_params[param] = data[param]

        # Make the follow-up call using litellm.acompletion directly
        # This avoids re-triggering our own hooks (unlike proxy's internal completion)
        try:
            follow_up_response = await litellm.acompletion(**follow_up_params)
            return follow_up_response
        except Exception as e:
            # If follow-up fails, return original response rather than error
            # Log the failure for debugging
            print(f"[ReasoningEdgeCaseHandler] Follow-up call failed: {e}")
            return response


# Singleton instance - LiteLLM expects this exact pattern
# Import this in proxy_config.yaml: reasoning_edge_case_handler.handler_instance
handler_instance = ReasoningEdgeCaseHandler()
