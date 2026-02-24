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
from litellm.types.utils import ModelResponseStream
from litellm.proxy.proxy_server import UserAPIKeyAuth
from typing import Optional, Any, AsyncGenerator


class ReasoningEdgeCaseHandler(CustomLogger):
    """
    LiteLLM Proxy handler that detects and fixes the reasoning edge case.

    When a model puts its response in reasoning_content but leaves content
    empty, this handler triggers a follow-up call with a nudge message.
    """

    def __init__(self):
        super().__init__()

    def _should_process_response(self, request_data: dict) -> bool:
        """
        Check if this response needs processing or is already a follow-up.

        Uses metadata to track retry count and prevent infinite loops.

        Args:
            request_data: The request data dict containing metadata

        Returns:
            True if we should process this response, False if it's a follow-up
        """
        metadata = request_data.get("metadata", {})
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
    ) -> Optional[Any]:
        """
        Initialize state in metadata before the LLM call.

        This state round-trips to async_post_call_streaming_iterator_hook via the
        same data dict reference.
        """
        # Ensure metadata exists
        data["metadata"] = data.get("metadata", {})

        # Initialize retry tracking - critical for recursion prevention
        data["metadata"]["_reasoning_retry_count"] = 0
        data["metadata"]["_reasoning_original_request_id"] = data.get("litellm_call_id")

        return data

    async def async_post_call_streaming_iterator_hook(
        self,
        user_api_key_dict: UserAPIKeyAuth,
        response: Any,
        request_data: dict,
    ) -> AsyncGenerator[ModelResponseStream, None]:
        """
        Check for reasoning edge case in streaming response and trigger follow-up if needed.

        If the model dumped its answer into reasoning_content but left
        content empty, we make a follow-up call with a "Go ahead!" nudge.
        """
        # GUARD CLAUSE: Skip if this is already a follow-up response
        if not self._should_process_response(request_data):
            async for chunk in response:
                yield chunk
            return

        # Collect chunks and track what we see
        chunks = []
        has_reasoning = False
        has_content = False
        reasoning_content_parts = []

        # Phase 1: Consume stream, yield chunks, analyze content
        async for chunk in response:
            chunks.append(chunk)

            # Real-time analysis of each chunk
            if chunk.choices:
                delta = chunk.choices[0].delta

                # Check for reasoning_content
                reasoning_piece = getattr(delta, "reasoning_content", None)
                if reasoning_piece:
                    has_reasoning = True
                    reasoning_content_parts.append(reasoning_piece)

                # Check for content
                content_piece = getattr(delta, "content", None)
                if content_piece:
                    has_content = True

            yield chunk

        # Phase 2: Post-stream decision - check for edge case
        content_empty = not has_content
        edge_case_detected = has_reasoning and content_empty

        if not edge_case_detected:
            return

        # EDGE CASE DETECTED - Mark as processed to prevent recursion
        metadata = request_data.get("metadata", {})
        metadata["_reasoning_retry_count"] = 1

        # Build the full reasoning content from collected parts
        reasoning = "".join(reasoning_content_parts)

        # Build follow-up messages with the nudge
        original_messages = request_data.get("messages", [])

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
            "model": request_data.get("model"),
            "messages": follow_up_messages,
            "stream": False,  # Force non-streaming for consistent processing
            "temperature": request_data.get("temperature", 0.7),
            "max_tokens": request_data.get("max_tokens"),
        }

        # Copy other relevant params if they exist
        for param in ["top_p", "presence_penalty", "frequency_penalty", "stop"]:
            if param in request_data:
                follow_up_params[param] = request_data[param]

        # Make the follow-up call using litellm.acompletion directly
        # This avoids re-triggering our own hooks (unlike proxy's internal completion)
        try:
            follow_up_response = await litellm.acompletion(**follow_up_params)

            # Convert the non-streaming follow-up response to a stream chunk
            # This maintains the streaming interface for the client
            if follow_up_response.choices:
                message = follow_up_response.choices[0].message
                content = getattr(message, "content", "")

                # Create a synthetic stream chunk with the follow-up content
                # We use the last chunk as a template and inject the new content
                if chunks:
                    template_chunk = chunks[-1]
                    # Create a new chunk with the follow-up content
                    # This preserves the streaming interface
                    from litellm.types.utils import Delta

                    new_delta = Delta(content=content)
                    template_chunk.choices[0].delta = new_delta
                    yield template_chunk

        except Exception as e:
            # If follow-up fails, we've already yielded the original chunks
            # Log the failure for debugging but don't break the stream
            print(f"[ReasoningEdgeCaseHandler] Follow-up call failed: {e}")


# Singleton instance - LiteLLM expects this exact pattern
# Import this in proxy_config.yaml: reasoning_edge_case_handler.handler_instance
handler_instance = ReasoningEdgeCaseHandler()