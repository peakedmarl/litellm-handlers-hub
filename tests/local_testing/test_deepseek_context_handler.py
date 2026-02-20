import pytest
import json
import os
import uuid
import asyncio
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, AsyncMock, patch, Mock

from litellm.integrations.custom_logger import CustomLogger
from litellm.types.utils import ModelResponse, ModelResponseStream, Choices, StreamingChoices, Message, Delta
from litellm.proxy.proxy_server import UserAPIKeyAuth

# The handler under test
from deepseek_context_handler import (
    DeepSeekContextHandler,
    InMemoryStorage,
    RedisStorage,
    Storage,
    BaseStorage,
)

# ==================== Fixtures ====================

@pytest.fixture
def handler():
    return DeepSeekContextHandler()

@pytest.fixture
def mock_redis():
    """Mock Redis using AsyncMock pattern from LiteLLM's test_caching.py"""
    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_instance = AsyncMock()
        mock_instance.get = AsyncMock(return_value=None)
        mock_instance.setex = AsyncMock(return_value=True)
        mock_instance.delete = AsyncMock(return_value=1)
        mock_redis_class.return_value = mock_instance
        yield mock_instance

@pytest.fixture
def mock_redis_with_data():
    """Simulate Redis with pre-existing data using local dict pattern"""
    storage = {}

    async def mock_get(key):
        return storage.get(key)

    async def mock_setex(key, ttl, value):
        storage[key] = value
        return True

    async def mock_delete(key):
        storage.pop(key, None)
        return 1

    with patch("redis.asyncio.Redis") as mock_redis_class:
        mock_instance = AsyncMock()
        mock_instance.get = AsyncMock(side_effect=mock_get)
        mock_instance.setex = AsyncMock(side_effect=mock_setex)
        mock_instance.delete = AsyncMock(side_effect=mock_delete)
        mock_redis_class.return_value = mock_instance
        yield mock_instance, storage

def create_mock_response_with_reasoning(reasoning_content: str, content: str = "Hello!"):
    """Follow LiteLLM's pattern from tests/local_testing/test_router.py"""
    return ModelResponse(
        id="chatcmpl-test-123",
        object="chat.completion",
        created=1699896916,
        model="deepseek-chat",
        choices=[
            Choices(
                index=0,
                message=Message(
                    role="assistant",
                    content=content,
                    reasoning_content=reasoning_content,  # DeepSeek reasoning
                ),
                finish_reason="stop",
            )
        ],
        usage={"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30},
    )

def create_mock_stream_chunks_with_reasoning(reasoning_pieces: List[str], content_pieces: List[str]):
    """
    Follow LiteLLM's pattern from tests/test_litellm/test_main.py::test_stream_chunk_builder_thinking_blocks
    """
    chunks = []

    # Reasoning chunks
    for piece in reasoning_pieces:
        chunks.append(ModelResponseStream(
            id="chatcmpl-stream-123",
            created=1699896916,
            model="deepseek-chat",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        reasoning_content=piece,  # DeepSeek streaming reasoning
                        content="",  # Empty content during reasoning phase
                        role="assistant",
                    ),
                    finish_reason=None,
                )
            ]
        ))

    # Content chunks
    for piece in content_pieces:
        chunks.append(ModelResponseStream(
            id="chatcmpl-stream-123",
            created=1699896916,
            model="deepseek-chat",
            object="chat.completion.chunk",
            choices=[
                StreamingChoices(
                    index=0,
                    delta=Delta(
                        content=piece,
                        reasoning_content=None,
                    ),
                    finish_reason=None,
                )
            ]
        ))

    # Final chunk
    chunks.append(ModelResponseStream(
        id="chatcmpl-stream-123",
        created=1699896916,
        model="deepseek-chat",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                index=0,
                delta=Delta(),
                finish_reason="stop",
            )
        ]
    ))

    return chunks

async def mock_async_generator(chunks: List[ModelResponseStream]):
    """LiteLLM pattern for mocking streaming responses"""
    for chunk in chunks:
        yield chunk

# ==================== Group 1: Reasoning Extraction Tests ====================

@pytest.mark.unit
def test_extract_reasoning_from_response_success(handler):
    """1. Extract from ModelResponse with reasoning_content"""
    response = create_mock_response_with_reasoning("Thinking process...")
    reasoning = handler._extract_reasoning_from_response(response)
    assert reasoning == "Thinking process..."

@pytest.mark.unit
def test_extract_reasoning_from_response_no_reasoning(handler):
    """2. Handle response without reasoning"""
    response = create_mock_response_with_reasoning(None, content="Just content")
    reasoning = handler._extract_reasoning_from_response(response)
    assert reasoning is None

@pytest.mark.unit
def test_extract_reasoning_from_response_invalid_response(handler):
    """3. Handle non-ModelResponse input"""
    reasoning = handler._extract_reasoning_from_response({"not": "a response"})
    assert reasoning is None

@pytest.mark.unit
def test_extract_reasoning_from_chunk_success(handler):
    """4. Extract from streaming chunk"""
    chunks = create_mock_stream_chunks_with_reasoning(["Thinking"], [])
    reasoning = handler._extract_reasoning_from_chunk(chunks[0])
    assert reasoning == "Thinking"

@pytest.mark.unit
def test_extract_reasoning_from_chunk_no_reasoning(handler):
    """5. Handle chunk without reasoning"""
    chunks = create_mock_stream_chunks_with_reasoning([], ["Hello"])
    reasoning = handler._extract_reasoning_from_chunk(chunks[0])
    assert reasoning is None

# ==================== Group 2: Storage Backend Tests ====================

@pytest.mark.storage
@pytest.mark.asyncio
async def test_in_memory_storage_roundtrip():
    """6. Store and retrieve from InMemoryStorage"""
    storage = InMemoryStorage()
    session_id = "test-session"
    data = {"reasoning": "some thinking"}
    await storage.set(session_id, data)
    retrieved = await storage.get(session_id)
    assert retrieved["reasoning"] == "some thinking"

@pytest.mark.storage
@pytest.mark.asyncio
async def test_redis_storage_roundtrip(mock_redis_with_data):
    """7. Store and retrieve from RedisStorage (mocked)"""
    mock_inst, store_dict = mock_redis_with_data
    storage = RedisStorage(host="localhost", port=6379)
    session_id = "test-session"
    data = {"reasoning": "some thinking"}
    await storage.set(session_id, data)

    # Redis key is deepseek:reasoning:test-session
    retrieved = await storage.get(session_id)
    assert retrieved["reasoning"] == "some thinking"
    mock_inst.setex.assert_called()

@pytest.mark.storage
@pytest.mark.asyncio
async def test_storage_fallback_when_redis_unavailable():
    """8. Fallback to memory on Redis init failure"""
    with patch.dict(os.environ, {"REDIS_HOST": "localhost", "REDIS_PORT": "6379"}):
        with patch("redis.asyncio.Redis", side_effect=Exception("Redis down")):
            storage_factory = Storage()
            assert isinstance(storage_factory._backend, InMemoryStorage)

@pytest.mark.storage
@pytest.mark.asyncio
async def test_storage_ttl_handling(mock_redis):
    """9. Verify TTL is passed correctly"""
    storage = RedisStorage(host="localhost", port=6379)
    session_id = "test-session"
    data = {"reasoning": "some thinking"}
    await storage.set(session_id, data, ttl=100)

    # Check if setex was called with ttl=100
    mock_redis.setex.assert_called_with(f"deepseek:reasoning:{session_id}", 100, json.dumps(data))

@pytest.mark.storage
@pytest.mark.asyncio
async def test_storage_delete():
    """10. Verify delete operation clears data"""
    storage = InMemoryStorage()
    session_id = "test-session"
    await storage.set(session_id, {"reasoning": "..."})
    await storage.delete(session_id)
    assert await storage.get(session_id) is None

# ==================== Group 3: Pre-Call Hook Tests ====================

@pytest.mark.hooks
@pytest.mark.asyncio
async def test_pre_call_hook_new_user_strips_reasoning(handler):
    """11. New user message strips existing reasoning"""
    data = {
        "messages": [
            {"role": "user", "content": "hello", "reasoning_content": "should be stripped"},
            {"role": "assistant", "content": "hi", "reasoning_content": "also stripped"},
            {"role": "user", "content": "new question"}
        ],
        "litellm_session_id": "test-session"
    }

    modified_data = await handler.async_pre_call_hook(
        user_api_key_dict=MagicMock(),
        cache=None,
        data=data,
        call_type="completion"
    )

    for msg in modified_data["messages"]:
        assert "reasoning_content" not in msg

@pytest.mark.hooks
@pytest.mark.asyncio
async def test_pre_call_hook_new_user_clears_storage(handler):
    """12. New user turn clears stored reasoning"""
    session_id = "test-session"
    await handler.storage.set(session_id, {"reasoning": "old reasoning"})

    data = {
        "messages": [{"role": "user", "content": "new question"}],
        "litellm_session_id": session_id
    }

    await handler.async_pre_call_hook(
        user_api_key_dict=MagicMock(),
        cache=None,
        data=data,
        call_type="completion"
    )

    assert await handler.storage.get(session_id) is None

@pytest.mark.hooks
@pytest.mark.asyncio
async def test_pre_call_hook_tool_only_injects_reasoning(handler):
    """13. Tool-only turn injects stored reasoning"""
    session_id = "test-session"
    reasoning = "The value is 42."
    await handler.storage.set(session_id, {"reasoning": reasoning})

    data = {
        "messages": [
            {"role": "user", "content": "What is the value?"},
            {"role": "assistant", "content": "Let me check..."},
            {"role": "tool", "content": "42", "tool_call_id": "1"}
        ],
        "litellm_session_id": session_id
    }

    modified_data = await handler.async_pre_call_hook(
        user_api_key_dict=MagicMock(),
        cache=None,
        data=data,
        call_type="completion"
    )

    # Should be injected into the last assistant message
    assert modified_data["messages"][1]["role"] == "assistant"
    assert modified_data["messages"][1]["reasoning_content"] == reasoning

@pytest.mark.hooks
@pytest.mark.asyncio
async def test_pre_call_hook_tool_only_no_stored_reasoning(handler):
    """14. Tool-only with no storage logs warning"""
    # No data in storage
    data = {
        "messages": [
            {"role": "user", "content": "What is the value?"},
            {"role": "assistant", "content": "Let me check..."},
            {"role": "tool", "content": "42", "tool_call_id": "1"}
        ],
        "litellm_session_id": "test-session"
    }

    with patch("deepseek_context_handler.logger.warning") as mock_warning:
        await handler.async_pre_call_hook(
            user_api_key_dict=MagicMock(),
            cache=None,
            data=data,
            call_type="completion"
        )
        mock_warning.assert_called_with("⚠️ No stored reasoning found for tool-only interaction")

@pytest.mark.hooks
@pytest.mark.asyncio
async def test_pre_call_hook_empty_messages(handler):
    """15. Handle empty messages gracefully"""
    data = {"messages": []}
    result = await handler.async_pre_call_hook(
        user_api_key_dict=MagicMock(),
        cache=None,
        data=data,
        call_type="completion"
    )
    assert result == data

@pytest.mark.hooks
@pytest.mark.asyncio
async def test_pre_call_hook_anthropic_tool_format(handler):
    """16. Handle Anthropic-style tool results"""
    session_id = "test-session"
    reasoning = "Thinking..."
    await handler.storage.set(session_id, {"reasoning": reasoning})

    data = {
        "messages": [
            {"role": "user", "content": "test"},
            {"role": "assistant", "content": "thinking"},
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "1", "content": "result"}]
            }
        ],
        "litellm_session_id": session_id
    }

    modified_data = await handler.async_pre_call_hook(
        user_api_key_dict=MagicMock(),
        cache=None,
        data=data,
        call_type="completion"
    )

    assert modified_data["messages"][1]["reasoning_content"] == reasoning

# ==================== Group 4: Post-Call Success Hook Tests ====================

@pytest.mark.hooks
@pytest.mark.asyncio
async def test_post_call_success_stores_reasoning(handler):
    """17. Non-streaming response stores reasoning"""
    session_id = "test-session"
    response = create_mock_response_with_reasoning("Stored thinking")
    data = {"litellm_session_id": session_id}

    await handler.async_post_call_success_hook(
        data=data,
        user_api_key_dict=MagicMock(),
        response=response
    )

    stored = await handler.storage.get(session_id)
    assert stored["reasoning"] == "Stored thinking"

@pytest.mark.hooks
@pytest.mark.asyncio
async def test_post_call_success_no_reasoning(handler):
    """18. Response without reasoning doesn't store"""
    session_id = "test-session"
    response = create_mock_response_with_reasoning(None)
    data = {"litellm_session_id": session_id}

    await handler.async_post_call_success_hook(
        data=data,
        user_api_key_dict=MagicMock(),
        response=response
    )

    stored = await handler.storage.get(session_id)
    assert stored is None

@pytest.mark.hooks
@pytest.mark.asyncio
async def test_post_call_success_invalid_response_type(handler):
    """19. Handle non-ModelResponse gracefully"""
    result = await handler.async_post_call_success_hook(
        data={},
        user_api_key_dict=MagicMock(),
        response={"not": "a response"}
    )
    assert result == {"not": "a response"}

# ==================== Group 5: Streaming Iterator Hook Tests ====================

@pytest.mark.streaming
@pytest.mark.asyncio
async def test_streaming_iterator_accumulates_reasoning(handler):
    """20. Accumulates reasoning across chunks"""
    session_id = "test-session"
    chunks = create_mock_stream_chunks_with_reasoning(["Part 1, ", "Part 2"], ["Hello"])
    request_data = {"litellm_session_id": session_id}

    stream = mock_async_generator(chunks)

    async_gen = handler.async_post_call_streaming_iterator_hook(
        user_api_key_dict=MagicMock(),
        response=stream,
        request_data=request_data
    )

    # Consume the generator
    async for _ in async_gen:
        pass

    stored = await handler.storage.get(session_id)
    assert stored["reasoning"] == "Part 1, Part 2"

@pytest.mark.streaming
@pytest.mark.asyncio
async def test_streaming_iterator_stores_on_completion(handler):
    """21. Stores accumulated reasoning after stream"""
    # Already verified in test 20, but let's be explicit
    session_id = "test-session"
    chunks = create_mock_stream_chunks_with_reasoning(["Thinking"], ["Content"])

    # Use a fresh storage to be sure
    handler.storage = Storage() # Defaults to InMemoryStorage

    stream = mock_async_generator(chunks)
    async_gen = handler.async_post_call_streaming_iterator_hook(
        user_api_key_dict=MagicMock(),
        response=stream,
        request_data={"litellm_session_id": session_id}
    )

    # Check storage BEFORE consumption
    assert await handler.storage.get(session_id) is None

    async for _ in async_gen:
        pass

    # Check storage AFTER consumption
    stored = await handler.storage.get(session_id)
    assert stored["reasoning"] == "Thinking"

@pytest.mark.streaming
@pytest.mark.asyncio
async def test_streaming_iterator_yields_all_chunks(handler):
    """22. All chunks pass through unchanged"""
    chunks = create_mock_stream_chunks_with_reasoning(["T"], ["C"])
    stream = mock_async_generator(chunks)

    async_gen = handler.async_post_call_streaming_iterator_hook(
        user_api_key_dict=MagicMock(),
        response=stream,
        request_data={"litellm_session_id": "test"}
    )

    yielded_chunks = []
    async for chunk in async_gen:
        yielded_chunks.append(chunk)

    assert len(yielded_chunks) == len(chunks)
    assert yielded_chunks == chunks

@pytest.mark.streaming
@pytest.mark.asyncio
async def test_streaming_iterator_no_reasoning_chunks(handler):
    """23. Handles stream without reasoning"""
    session_id = "test-session"
    chunks = create_mock_stream_chunks_with_reasoning([], ["Hello", " world"])
    stream = mock_async_generator(chunks)

    async_gen = handler.async_post_call_streaming_iterator_hook(
        user_api_key_dict=MagicMock(),
        response=stream,
        request_data={"litellm_session_id": session_id}
    )

    async for _ in async_gen:
        pass

    assert await handler.storage.get(session_id) is None

@pytest.mark.streaming
@pytest.mark.asyncio
async def test_streaming_iterator_empty_stream(handler):
    """24. Handles empty stream gracefully"""
    async_gen = handler.async_post_call_streaming_iterator_hook(
        user_api_key_dict=MagicMock(),
        response=mock_async_generator([]),
        request_data={"litellm_session_id": "test"}
    )

    async for _ in async_gen:
        pass
    # Should not raise any error

# ==================== Group 6: Integration/End-to-End Tests ====================

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_flow_new_user_then_tool_turn(handler):
    """25. Complete flow: user → assistant (store) → tool (inject)"""
    session_id = "e2e-session"

    # 1. New user turn
    data1 = {
        "messages": [{"role": "user", "content": "Get weather for London"}],
        "litellm_session_id": session_id
    }
    await handler.async_pre_call_hook(MagicMock(), None, data1, "completion")

    # 2. Assistant response with reasoning and tool call
    response1 = create_mock_response_with_reasoning("I need to call the weather tool.")
    await handler.async_post_call_success_hook(data1, MagicMock(), response1)

    # Verify stored
    stored = await handler.storage.get(session_id)
    assert stored["reasoning"] == "I need to call the weather tool."

    # 3. Tool turn
    data2 = {
        "messages": [
            {"role": "user", "content": "Get weather for London"},
            {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {"name": "get_weather"}}]},
            {"role": "tool", "content": "Cloudy, 15C", "tool_call_id": "1"}
        ],
        "litellm_session_id": session_id
    }

    modified_data2 = await handler.async_pre_call_hook(MagicMock(), None, data2, "completion")

    # Verify injection
    assert modified_data2["messages"][1]["reasoning_content"] == "I need to call the weather tool."

@pytest.mark.integration
@pytest.mark.asyncio
async def test_full_flow_streaming_accumulation(handler):
    """26. Streaming flow with reasoning accumulation"""
    session_id = "stream-e2e"
    data = {"litellm_session_id": session_id}

    chunks = create_mock_stream_chunks_with_reasoning(["Think", "ing"], ["Hi"])
    stream = mock_async_generator(chunks)

    async_gen = handler.async_post_call_streaming_iterator_hook(MagicMock(), stream, data)
    async for _ in async_gen:
        pass

    stored = await handler.storage.get(session_id)
    assert stored["reasoning"] == "Thinking"

@pytest.mark.integration
@pytest.mark.asyncio
async def test_session_id_persistence(handler):
    """27. Session ID consistent across hooks"""
    data = {"messages": [{"role": "user", "content": "test"}]}

    # Hook 1: Pre-call generates session_id if missing
    await handler.async_pre_call_hook(MagicMock(), None, data, "completion")
    session_id = data.get("litellm_session_id")
    assert session_id is not None

    # Hook 2: Post-call uses same session_id
    response = create_mock_response_with_reasoning("Thinking")
    await handler.async_post_call_success_hook(data, MagicMock(), response)

    stored = await handler.storage.get(session_id)
    assert stored["reasoning"] == "Thinking"

# ==================== Group 7: Edge Cases & Error Handling ====================

@pytest.mark.asyncio
async def test_pre_call_hook_exception_fails_open(handler):
    """28. Exception in pre-call returns original data"""
    handler.storage = None # Force error
    data = {"messages": [{"role": "user", "content": "test"}]}

    result = await handler.async_pre_call_hook(MagicMock(), None, data, "completion")
    assert result == data

@pytest.mark.asyncio
async def test_post_call_hook_exception_fails_open(handler):
    """29. Exception in post-call returns original response"""
    handler.storage = None # Force error
    response = create_mock_response_with_reasoning("Think")
    data = {"litellm_session_id": "test"}

    result = await handler.async_post_call_success_hook(data, MagicMock(), response)
    assert result == response

@pytest.mark.asyncio
async def test_streaming_iterator_exception_propagates(handler):
    """30. Streaming errors propagate but store reasoning"""
    session_id = "err-session"

    async def error_generator():
        yield create_mock_stream_chunks_with_reasoning(["Think"], [])[0]
        raise ValueError("Stream failed")

    async_gen = handler.async_post_call_streaming_iterator_hook(
        MagicMock(),
        error_generator(),
        {"litellm_session_id": session_id}
    )

    with pytest.raises(ValueError, match="Stream failed"):
        async for _ in async_gen:
            pass

    # Even if it failed, it might have stored what it got before the error
    stored = await handler.storage.get(session_id)
    assert stored["reasoning"] == "Think"

@pytest.mark.asyncio
async def test_message_injection_no_assistant_message(handler):
    """31. Prepend reasoning msg if no assistant found"""
    messages = [{"role": "user", "content": "test"}]
    reasoning = "Thought"

    injected = handler._inject_reasoning_into_messages(messages, reasoning)

    assert len(injected) == 2
    assert injected[0]["role"] == "assistant"
    assert injected[0]["reasoning_content"] == "Thought"
    assert injected[1] == messages[0]

@pytest.mark.asyncio
async def test_strip_reasoning_anthropic_thinking_blocks(handler):
    """32. Strip thinking blocks from content list"""
    messages = [
        {
            "role": "assistant",
            "content": [
                {"type": "thinking", "thinking": "Let me see..."},
                {"type": "text", "text": "Hello!"}
            ]
        }
    ]

    stripped = handler._strip_reasoning_from_messages(messages)

    content = stripped[0]["content"]
    assert len(content) == 1
    assert content[0]["type"] == "text"
    assert "thinking" not in [c.get("type") for c in content]
