import pytest
import asyncio
import hashlib
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime, timezone, timedelta

import litellm
from litellm.types.utils import ModelResponseStream, StreamingChoices, Delta, Message
from litellm.proxy._types import UserAPIKeyAuth
from reasoning_edge_case_handler import ReasoningEdgeCaseHandler, SESSION_TTL_SECONDS

@pytest.fixture
def handler():
    return ReasoningEdgeCaseHandler()

@pytest.fixture
def user_auth():
    return UserAPIKeyAuth(
        api_key="sk-test-123",
        user_id="test-user",
        team_id="test-team",
        metadata={},
    )

def make_chunk(content=None, reasoning_content=None, finish_reason=None):
    return ModelResponseStream(
        id="test-id",
        created=1234567890,
        model="deepseek-reasoner",
        object="chat.completion.chunk",
        choices=[
            StreamingChoices(
                finish_reason=finish_reason,
                index=0,
                delta=Delta(
                    content=content,
                    reasoning_content=reasoning_content,
                ),
            )
        ],
    )

def make_mock_nudge_response(content="Final answer"):
    mock_resp = MagicMock()
    mock_resp.choices = [
        MagicMock(message=MagicMock(content=content))
    ]
    # Set choice index 0
    mock_resp.choices[0].index = 0
    # message is actually a Message or similar
    mock_resp.choices[0].message = Message(content=content, role="assistant")
    return mock_resp

async def async_gen_wrapper(chunks):
    for chunk in chunks:
        yield chunk

@pytest.mark.asyncio
async def test_detects_edge_case_with_reasoning_only(handler, user_auth):
    """Test 1: Detects reasoning-only response and triggers nudge."""
    chunks = [
        make_chunk(reasoning_content="Thinking..."),
        make_chunk(reasoning_content="Almost there..."),
        make_chunk(finish_reason="stop")
    ]

    request_data = {
        "model": "deepseek-reasoner",
        "messages": [{"role": "user", "content": "Hello"}],
        "litellm_call_id": "call-123"
    }

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = make_mock_nudge_response()

        response_gen = handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_auth,
            response=async_gen_wrapper(chunks),
            request_data=request_data
        )

        collected_chunks = []
        async for chunk in response_gen:
            collected_chunks.append(chunk)

        assert mock_acompletion.called
        assert mock_acompletion.call_count == 1

        # Check if nudge content was yielded (it's yielded as a synthetic chunk)
        assert any(getattr(c.choices[0].delta, "content", None) == "Final answer" for c in collected_chunks)

@pytest.mark.asyncio
async def test_skips_processing_when_already_nudged(handler, user_auth):
    """Test 2: Skip processing if nudge was already triggered for this session."""
    session_id = handler._get_session_id(user_auth)
    await handler.storage.mark_edge_case_detected(session_id, "call-1")
    await handler.storage.mark_recovery_nudge_triggered(session_id)

    chunks = [make_chunk(reasoning_content="Reasoning again")]
    request_data = {"model": "m", "messages": [], "litellm_call_id": "call-2"}

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        response_gen = handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_auth,
            response=async_gen_wrapper(chunks),
            request_data=request_data
        )
        async for _ in response_gen: pass

        assert not mock_acompletion.called

@pytest.mark.asyncio
async def test_clears_session_on_normal_response(handler, user_auth):
    """Test 3: Normal response clears session storage."""
    session_id = handler._get_session_id(user_auth)
    # Pre-populate session
    await handler.storage.mark_edge_case_detected(session_id, "old-call")

    chunks = [
        make_chunk(reasoning_content="Thinking"),
        make_chunk(content="Hello world")
    ]
    request_data = {"model": "m", "messages": [], "litellm_call_id": "call-3"}

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        response_gen = handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_auth,
            response=async_gen_wrapper(chunks),
            request_data=request_data
        )
        async for _ in response_gen: pass

        assert not mock_acompletion.called
        # Storage should be cleared
        session = await handler.storage.get_session(session_id)
        assert session is None

@pytest.mark.asyncio
async def test_ttl_expiration_allows_renudge(handler, user_auth):
    """Test 4: Expired session allows a new nudge."""
    session_id = handler._get_session_id(user_auth)

    # Manually set an expired state
    expired_time = (datetime.now(timezone.utc) - timedelta(seconds=SESSION_TTL_SECONDS + 1)).isoformat()
    await handler.storage._backend.set(session_id, {
        "edge_case_detected": True,
        "recovery_nudge_triggered": True,
        "last_detection_at": expired_time
    })
    # Overwrite the timestamp set by `set` (InMemoryStorage.set updates it to now)
    handler.storage._backend._store[session_id]["last_detection_at"] = expired_time

    chunks = [make_chunk(reasoning_content="Reasoning")]
    request_data = {"model": "m", "messages": [], "litellm_call_id": "call-4"}

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = make_mock_nudge_response()
        response_gen = handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_auth,
            response=async_gen_wrapper(chunks),
            request_data=request_data
        )
        async for _ in response_gen: pass

        assert mock_acompletion.called

@pytest.mark.asyncio
async def test_yields_all_original_chunks(handler, user_auth):
    """Test 5: All original chunks are yielded even when nudge is triggered."""
    chunks = [
        make_chunk(reasoning_content="Thought 1"),
        make_chunk(reasoning_content="Thought 2"),
        make_chunk(finish_reason="stop")
    ]
    request_data = {"model": "m", "messages": [], "litellm_call_id": "call-5"}

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = make_mock_nudge_response()
        response_gen = handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_auth,
            response=async_gen_wrapper(chunks),
            request_data=request_data
        )
        collected_chunks = []
        async for chunk in response_gen:
            collected_chunks.append(chunk)

        # Should have 3 original chunks + 1 synthetic nudge chunk
        assert len(collected_chunks) == 4
        # Verify original chunks are present
        assert collected_chunks[0].choices[0].delta.reasoning_content == "Thought 1"
        assert collected_chunks[1].choices[0].delta.reasoning_content == "Thought 2"

@pytest.mark.asyncio
async def test_handles_missing_session_id(handler):
    """Test 6: Handles user_api_key_dict without token attribute."""
    # Auth without token
    user_auth_no_token = MagicMock(spec=UserAPIKeyAuth)
    del user_auth_no_token.token

    chunks = [make_chunk(reasoning_content="Reasoning only")]
    request_data = {"model": "m", "messages": [], "litellm_call_id": "call-6"}

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        response_gen = handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_auth_no_token,
            response=async_gen_wrapper(chunks),
            request_data=request_data
        )
        collected = []
        async for chunk in response_gen:
            collected.append(chunk)

        assert not mock_acompletion.called
        assert len(collected) == 1

@pytest.mark.asyncio
async def test_preserves_request_params_in_nudge(handler, user_auth):
    """Test 7: Nudge call preserves request parameters."""
    chunks = [make_chunk(reasoning_content="Thinking")]
    request_data = {
        "model": "gpt-4",
        "messages": [{"role": "user", "content": "Tell me a joke"}],
        "temperature": 0.8,
        "max_tokens": 100,
        "top_p": 0.9,
        "presence_penalty": 0.5,
        "litellm_call_id": "call-7"
    }

    with patch("litellm.acompletion", new_callable=AsyncMock) as mock_acompletion:
        mock_acompletion.return_value = make_mock_nudge_response()
        response_gen = handler.async_post_call_streaming_iterator_hook(
            user_api_key_dict=user_auth,
            response=async_gen_wrapper(chunks),
            request_data=request_data
        )
        async for _ in response_gen: pass

        assert mock_acompletion.called
        _, kwargs = mock_acompletion.call_args

        assert kwargs["model"] == "gpt-4"
        assert kwargs["temperature"] == 0.8
        assert kwargs["max_tokens"] == 100
        assert kwargs["top_p"] == 0.9
        assert kwargs["presence_penalty"] == 0.5
        assert kwargs["messages"][-1] == {"role": "user", "content": "Go ahead!"}
        assert kwargs["stream"] is False
