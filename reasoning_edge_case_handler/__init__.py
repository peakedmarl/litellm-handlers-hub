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
import time
import copy
import traceback
from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponseStream
from litellm.proxy.proxy_server import UserAPIKeyAuth
from typing import Optional, Any, AsyncGenerator


# Debug logging helper
def _debug_log(prefix: str, message: str, call_id: str = None):
    """Centralized debug logging with timestamp and optional call_id."""
    ts = time.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
    call_str = f"[call_id={call_id}]" if call_id else ""
    print(f"[DEBUG {ts}] {prefix} {call_str} {message}")


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

        call_id = request_data.get("litellm_call_id", "unknown")

        # STEP 1.3: Log the _should_process_response decision
        _debug_log(
            "_should_process_response",
            f"retry_count={retry_count} >= 1? {retry_count >= 1}",
            call_id
        )

        # Hard cap at 1 retry - never process our own follow-up responses
        if retry_count >= 1:
            _debug_log(
                "_should_process_response",
                f"SKIPPING - already processed (retry_count={retry_count})",
                call_id
            )
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
        call_id = data.get("litellm_call_id", "unknown")

        # STEP 1.1: Log hook entry with call ID
        _debug_log(
            "async_pre_call_hook",
            f"Entry - initializing metadata for call_id={call_id}",
            call_id
        )

        # Ensure metadata exists
        data["metadata"] = data.get("metadata", {})

        # Initialize retry tracking - critical for recursion prevention
        data["metadata"]["_reasoning_retry_count"] = 0
        data["metadata"]["_reasoning_original_request_id"] = data.get("litellm_call_id")

        # STEP 6.2: Log model name
        model = data.get("model", "unknown")
        _debug_log(
            "async_pre_call_hook",
            f"Initialized _reasoning_retry_count=0, model={model}",
            call_id
        )

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
        call_id = request_data.get("litellm_call_id", "unknown")
        model = request_data.get("model", "unknown")

        # STEP 1.1: Log hook entry with call ID
        _debug_log(
            "async_post_call_streaming_iterator_hook",
            f"Entry - model={model}, call_id={call_id}",
            call_id
        )

        # STEP 1.2: Dump the full request_data metadata state
        metadata = request_data.get("metadata", {})
        _debug_log(
            "request_data.metadata",
            f"Full metadata: {metadata}",
            call_id
        )

        # STEP 1.3: Log the _should_process_response decision
        should_process = self._should_process_response(request_data)
        _debug_log(
            "async_post_call_streaming_iterator_hook",
            f"_should_process_response returned: {should_process}",
            call_id
        )

        # GUARD CLAUSE: Skip if this is already a follow-up response
        if not should_process:
            _debug_log(
                "async_post_call_streaming_iterator_hook",
                "GUARD CLAUSE: Not processing, yielding original chunks",
                call_id
            )
            async for chunk in response:
                yield chunk
            return

        # STEP 6.1: Log the original messages
        original_messages = request_data.get("messages", [])
        _debug_log(
            "request_data.messages",
            f"Message count: {len(original_messages)}, messages={original_messages}",
            call_id
        )

        # Collect chunks and track what we see
        chunks = []
        has_reasoning = False
        has_content = False

        # Phase 1: Consume stream, yield chunks, analyze content
        chunk_count = 0
        async for chunk in response:
            chunks.append(chunk)
            chunk_count += 1

            # STEP 2.1: Log the first chunk's complete structure
            if chunk_count == 1:
                _debug_log(
                    "chunk_structure",
                    f"First chunk type: {type(chunk)}, chunk.__dict__={chunk.__dict__}",
                    call_id
                )

            # STEP 3.1: Chunk-by-chunk field extraction logging
            if chunk.choices:
                delta = chunk.choices[0].delta

                # STEP 2.2: Inspect delta fields dynamically
                if chunk_count <= 3:  # Log first 3 chunks for structure discovery
                    delta_attrs = [a for a in dir(delta) if not a.startswith('_')]
                    _debug_log(
                        f"chunk_{chunk_count}_delta_fields",
                        f"delta type: {type(delta)}, attributes: {delta_attrs}",
                        call_id
                    )

                # Check for reasoning_content
                reasoning_piece = getattr(delta, "reasoning_content", None)
                if reasoning_piece:
                    if not has_reasoning:  # Log only on first detection
                        _debug_log(
                            "content_analysis",
                            f"Found reasoning_content: '{reasoning_piece[:100]}...' (truncated)",
                            call_id
                        )
                    has_reasoning = True

                # Check for content
                content_piece = getattr(delta, "content", None)
                if content_piece:
                    if not has_content:  # Log only on first detection
                        _debug_log(
                            "content_analysis",
                            f"Found content: '{content_piece[:100]}...' (truncated)",
                            call_id
                        )
                    has_content = True

                # STEP 3.2: Log accumulation of has_reasoning and has_content
                _debug_log(
                    f"chunk_{chunk_count}_analysis",
                    f"has_reasoning={has_reasoning}, has_content={has_content}",
                    call_id
                )

            yield chunk

        # STEP 3.3: Log total chunks collected
        _debug_log(
            "stream_complete",
            f"Total chunks collected: {chunk_count}",
            call_id
        )

        # Phase 2: Post-stream decision - check for edge case
        content_empty = not has_content
        edge_case_detected = has_reasoning and content_empty

        # STEP 4.1: Log the final content analysis variables
        _debug_log(
            "edge_case_check",
            f"has_reasoning={has_reasoning}, has_content={has_content}, content_empty={content_empty}",
            call_id
        )

        # STEP 4.2: Log the edge case detection result
        _debug_log(
            "edge_case_detection",
            f"edge_case_detected = has_reasoning({has_reasoning}) AND content_empty({content_empty}) = {edge_case_detected}",
            call_id
        )

        if not edge_case_detected:
            _debug_log(
                "async_post_call_streaming_iterator_hook",
                "No edge case detected, returning without follow-up",
                call_id
            )
            return

        # EDGE CASE DETECTED!
        # STEP 4.3: Log when edge case IS detected
        _debug_log(
            "EDGE_CASE_DETECTED!",
            f"Model {model} dumped response in reasoning_content but content is empty",
            call_id
        )

        # EDGE CASE DETECTED - Mark as processed to prevent recursion
        metadata = request_data.get("metadata", {})
        metadata["_reasoning_retry_count"] = 1

        # Build follow-up messages with the nudge
        original_messages = request_data.get("messages", [])

        follow_up_messages = original_messages + [
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

        # STEP 5.1: Log follow-up call preparation
        _debug_log(
            "follow_up_preparation",
            f"follow_up_params={follow_up_params}",
            call_id
        )

        # Make the follow-up call using litellm.acompletion directly
        # This avoids re-triggering our own hooks (unlike proxy's internal completion)
        try:
            # STEP 5.2: Log follow-up call attempt
            _debug_log(
                "follow_up_call",
                f"Starting follow-up call with 'Go ahead!' nudge",
                call_id
            )

            follow_up_response = await litellm.acompletion(**follow_up_params)

            # STEP 5.3: Log follow-up response structure
            _debug_log(
                "follow_up_response",
                f"Response type: {type(follow_up_response)}",
                call_id
            )

            if follow_up_response.choices:
                message = follow_up_response.choices[0].message
                content = getattr(message, "content", "")

                _debug_log(
                    "follow_up_response",
                    f"message.content = '{content[:100]}...' (truncated) or empty: '{content}'",
                    call_id
                )

                # STEP 5.4: Log synthetic chunk creation
                _debug_log(
                    "synthetic_chunk",
                    f"Creating synthetic chunk from follow-up response, chunks collected: {len(chunks)}",
                    call_id
                )

                # Create a synthetic stream chunk with the follow-up content
                # We use the last chunk as a template and inject the new content
                if chunks:
                    # Find the last chunk that has choices, or fallback to the last chunk
                    template_chunk = None
                    for c in reversed(chunks):
                        if getattr(c, "choices", None):
                            template_chunk = copy.deepcopy(c)
                            break

                    _debug_log(
                        "synthetic_chunk",
                        f"template_chunk found: {template_chunk is not None}",
                        call_id
                    )

                    if template_chunk and template_chunk.choices:
                        # Create a new chunk with the follow-up content
                        # This preserves the streaming interface
                        from litellm.types.utils import Delta

                        new_delta = Delta(content=content)
                        template_chunk.choices[0].delta = new_delta

                        _debug_log(
                            "synthetic_chunk",
                            f"Yielding synthetic chunk with content: '{content[:50]}...' (truncated)",
                            call_id
                        )
                        yield template_chunk
                    else:
                        _debug_log(
                            "synthetic_chunk_ERROR",
                            "Could not find template chunk with choices, cannot yield follow-up",
                            call_id
                        )
                else:
                    _debug_log(
                        "synthetic_chunk_ERROR",
                        "No original chunks to use as template",
                        call_id
                    )

        except Exception as e:
            # STEP 5.5: Log exception details with full traceback
            tb = traceback.format_exc()
            _debug_log(
                "follow_up_ERROR",
                f"Follow-up call failed: {str(e)}\nTraceback: {tb}",
                call_id
            )
            # If follow-up fails, we've already yielded the original chunks
            # Log the failure for debugging but don't break the stream


# Singleton instance - LiteLLM expects this exact pattern
# Import this in proxy_config.yaml: reasoning_edge_case_handler.handler_instance
handler_instance = ReasoningEdgeCaseHandler()