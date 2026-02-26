import pytest
import pytest_asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta, ModelResponse, Choices, Message
from litellm.proxy._types import UserAPIKeyAuth
import litellm
import sys
import os

# Add parent directory to path to import handler
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from reasoning_edge_case_handler import ReasoningEdgeCaseHandler

@pytest.fixture
def handler():
    return ReasoningEdgeCaseHandler()

@pytest.fixture
def mock_user_api_key_dict():
    return UserAPIKeyAuth(
        api_key="test-key",
        user_id="test-user",
        team_id="test-team"
    )

@pytest.fixture
def sample_request_data():
    return {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hello!"}],
        "temperature": 0.7,
        "litellm_call_id": "test-call-id"
    }

@pytest.fixture
def mock_cache():
    return MagicMock()

async def mock_streaming_response_with_reasoning():
    """Mock stream with reasoning_content but NO content (edge case)"""
    yield ModelResponseStream(
        id="test-id-1",
        created=1234567890,
        model="deepseek-reasoner",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content="",
                    role="assistant",
                    reasoning_content="Let me analyze this step by step...",
                ),
            )
        ],
    )
    yield ModelResponseStream(
        id="test-id-2",
        created=1234567891,
        model="deepseek-reasoner",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=None,
                index=0,
                delta=Delta(
                    content="",
                    role="assistant",
                    reasoning_content=" Based on my analysis, the answer is 42.",
                ),
            )
        ],
    )
    yield ModelResponseStream(
        id="test-id-3",
        created=1234567892,
        model="deepseek-reasoner",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason="stop",
                index=0,
                delta=Delta(
                    content="",
                    role="assistant",
                    reasoning_content=None,
                ),
            )
        ],
    )

async def mock_streaming_response_with_content():
    """Mock stream with both reasoning AND content"""
    yield ModelResponseStream(
        id="test-id-1",
        choices=[StreamingChoices(index=0, delta=Delta(reasoning_content="Thinking..."))]
    )
    yield ModelResponseStream(
        id="test-id-2",
        choices=[StreamingChoices(index=0, delta=Delta(content="The answer is 42."))]
    )

async def mock_streaming_response_only_content():
    """Mock stream with content but no reasoning"""
    yield ModelResponseStream(
        id="test-id-1",
        choices=[StreamingChoices(index=0, delta=Delta(content="Hello!"))]
    )

async def mock_empty_streaming_response():
    """Mock empty async generator"""
    if False:
        yield

async def mock_streaming_response_missing_choices():
    """Mock chunk with empty choices list"""
    yield ModelResponseStream(id="test-id-1", choices=[])

async def mock_streaming_response_missing_delta():
    """Mock chunk with choice but no delta"""
    chunk = ModelResponseStream(id="test-id-1", choices=[StreamingChoices(index=0)])
    chunk.choices[0].delta = None
    yield chunk

# Test Suite 1: Hook Lifecycle & State Management

@pytest.mark.asyncio
async def test_pre_call_hook_initializes_metadata(handler, mock_user_api_key_dict, mock_cache, sample_request_data):
    data = await handler.async_pre_call_hook(mock_user_api_key_dict, mock_cache, sample_request_data, "call_type")
    assert data["metadata"]["_reasoning_retry_count"] == 0
    assert data["metadata"]["_reasoning_original_request_id"] == "test-call-id"

def test_should_process_response_returns_true_for_new_requests(handler):
    assert handler._should_process_response({}) is True
    assert handler._should_process_response({"metadata": {"_reasoning_retry_count": 0}}) is True

def test_should_process_response_returns_false_for_follow_ups(handler):
    assert handler._should_process_response({"metadata": {"_reasoning_retry_count": 1}}) is False
    assert handler._should_process_response({"metadata": {"_reasoning_retry_count": 2}}) is False

# Test Suite 2 & 3: Streaming Response Analysis & "Go Ahead!" Nudge

@pytest.mark.asyncio
async def test_detects_edge_case_when_only_reasoning_present(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_streaming_response_with_reasoning()

    # Mock follow-up response
    follow_up_response = ModelResponse(
        id="follow-up-id",
        choices=[Choices(message=Message(content="The answer is 42!"))]
    )

    with patch("litellm.acompletion", AsyncMock(return_value=follow_up_response)) as mock_acompletion:
        sample_request_data["metadata"] = {"_reasoning_retry_count": 0}

        collected_chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict=mock_user_api_key_dict,
            response=mock_stream,
            request_data=sample_request_data
        ):
            collected_chunks.append(chunk)

        assert mock_acompletion.called
        call_kwargs = mock_acompletion.await_args.kwargs
        assert call_kwargs["stream"] is False
        assert any("Go ahead!" in msg["content"] for msg in call_kwargs["messages"])
        assert any("<thinking>Let me analyze this step by step... Based on my analysis, the answer is 42.</thinking>" in msg["content"] for msg in call_kwargs["messages"])

        # Verify injected chunk
        assert any(chunk.choices[0].delta.content == "The answer is 42!" for chunk in collected_chunks)

@pytest.mark.asyncio
async def test_no_edge_case_when_content_present(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_streaming_response_with_content()
    with patch("litellm.acompletion", AsyncMock()) as mock_acompletion:
        sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
        async for _ in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
            pass
        assert not mock_acompletion.called

@pytest.mark.asyncio
async def test_no_edge_case_when_no_reasoning(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_streaming_response_only_content()
    with patch("litellm.acompletion", AsyncMock()) as mock_acompletion:
        sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
        async for _ in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
            pass
        assert not mock_acompletion.called

@pytest.mark.asyncio
async def test_no_edge_case_when_both_empty(handler, mock_user_api_key_dict, sample_request_data):
    async def mock_empty_stream():
        yield ModelResponseStream(choices=[StreamingChoices(index=0, delta=Delta())])

    mock_stream = mock_empty_stream()
    with patch("litellm.acompletion", AsyncMock()) as mock_acompletion:
        sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
        async for _ in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
            pass
        assert not mock_acompletion.called

# Test Suite 3: "Go Ahead!" Nudge Verification

@pytest.mark.asyncio
async def test_follow_up_includes_correct_nudge_message(handler, mock_user_api_key_dict, sample_request_data):
    # This is partially covered by test_detects_edge_case_when_only_reasoning_present
    # But let's be explicit about preserving original messages
    sample_request_data["messages"] = [{"role": "user", "content": "Original message"}]
    mock_stream = mock_streaming_response_with_reasoning()

    follow_up_response = ModelResponse(
        id="follow-up-id",
        choices=[Choices(message=Message(content="Response"))]
    )

    with patch("litellm.acompletion", AsyncMock(return_value=follow_up_response)) as mock_acompletion:
        sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
        async for _ in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
            pass

        call_messages = mock_acompletion.await_args.kwargs["messages"]
        assert call_messages[0] == {"role": "user", "content": "Original message"}
        assert call_messages[1]["role"] == "assistant"
        assert "<thinking>" in call_messages[1]["content"]
        assert call_messages[2] == {"role": "user", "content": "Go ahead!"}

@pytest.mark.asyncio
async def test_follow_up_preserves_request_parameters(handler, mock_user_api_key_dict, sample_request_data):
    sample_request_data.update({
        "temperature": 0.5,
        "max_tokens": 100,
        "top_p": 0.9,
        "presence_penalty": 0.1,
        "frequency_penalty": 0.2,
        "stop": ["END"]
    })
    mock_stream = mock_streaming_response_with_reasoning()
    follow_up_response = ModelResponse(id="id", choices=[Choices(message=Message(content=""))])

    with patch("litellm.acompletion", AsyncMock(return_value=follow_up_response)) as mock_acompletion:
        sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
        async for _ in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
            pass

        call_kwargs = mock_acompletion.await_args.kwargs
        assert call_kwargs["model"] == "deepseek-reasoner"
        assert call_kwargs["temperature"] == 0.5
        assert call_kwargs["max_tokens"] == 100
        assert call_kwargs["top_p"] == 0.9
        assert call_kwargs["presence_penalty"] == 0.1
        assert call_kwargs["frequency_penalty"] == 0.2
        assert call_kwargs["stop"] == ["END"]
        assert call_kwargs["stream"] is False

# Test Suite 4: Response Injection

@pytest.mark.asyncio
async def test_follow_up_content_injected_into_stream(handler, mock_user_api_key_dict, sample_request_data):
    # Covered by test_detects_edge_case_when_only_reasoning_present
    pass

@pytest.mark.asyncio
async def test_original_chunks_preserved_when_follow_up_fails(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_streaming_response_with_reasoning()

    with patch("litellm.acompletion", AsyncMock(side_effect=Exception("API Error"))):
        sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
        collected_chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
            collected_chunks.append(chunk)

        # Should have 3 original chunks
        assert len(collected_chunks) == 3
        assert "Let me analyze" in collected_chunks[0].choices[0].delta.reasoning_content

# Test Suite 5: Recursion Prevention

@pytest.mark.asyncio
async def test_no_infinite_loop_on_follow_up(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_streaming_response_with_reasoning()
    # If we are already in a retry, we should just yield chunks and NOT call acompletion
    sample_request_data["metadata"] = {"_reasoning_retry_count": 1}

    with patch("litellm.acompletion", AsyncMock()) as mock_acompletion:
        collected_chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
            collected_chunks.append(chunk)

        assert not mock_acompletion.called
        assert len(collected_chunks) == 3

# Test Suite 6: Edge Cases & Error Handling

@pytest.mark.asyncio
async def test_empty_stream_handling(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_empty_streaming_response()
    sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
    async for _ in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
        pass

@pytest.mark.asyncio
async def test_missing_choices_in_chunk(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_streaming_response_missing_choices()
    sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
    async for _ in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
        pass

@pytest.mark.asyncio
async def test_missing_delta_in_choice(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_streaming_response_missing_delta()
    sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
    async for _ in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
        pass

@pytest.mark.asyncio
async def test_follow_up_with_no_choices(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_streaming_response_with_reasoning()
    follow_up_response = ModelResponse(id="id", choices=[])

    with patch("litellm.acompletion", AsyncMock(return_value=follow_up_response)):
        sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
        collected_chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
            collected_chunks.append(chunk)
        # Should only have the 3 original chunks
        assert len(collected_chunks) == 3

@pytest.mark.asyncio
async def test_follow_up_with_empty_content(handler, mock_user_api_key_dict, sample_request_data):
    mock_stream = mock_streaming_response_with_reasoning()
    follow_up_response = ModelResponse(
        id="follow-up-id",
        choices=[Choices(message=Message(content=""))]
    )

    with patch("litellm.acompletion", AsyncMock(return_value=follow_up_response)):
        sample_request_data["metadata"] = {"_reasoning_retry_count": 0}
        collected_chunks = []
        async for chunk in handler.async_post_call_streaming_iterator_hook(mock_user_api_key_dict, mock_stream, sample_request_data):
            collected_chunks.append(chunk)
        # Should have 4 chunks (3 original + 1 injected)
        assert len(collected_chunks) == 4
        assert collected_chunks[-1].choices[0].delta.content == ""
