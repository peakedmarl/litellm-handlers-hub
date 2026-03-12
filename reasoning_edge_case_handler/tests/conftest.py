import pytest
import litellm

@pytest.fixture(autouse=True)
def isolate_litellm_state():
    """Prevents callback/state leakage between tests"""
    original_callbacks = litellm.callbacks.copy()
    original_success = litellm.success_callback.copy()
    original_failure = litellm.failure_callback.copy()
    original_async_success = litellm._async_success_callback.copy()
    original_async_failure = litellm._async_failure_callback.copy()

    litellm.callbacks = []
    litellm.success_callback = []
    litellm.failure_callback = []
    litellm._async_success_callback = []
    litellm._async_failure_callback = []

    if hasattr(litellm, 'in_memory_llm_clients_cache'):
        if hasattr(litellm.in_memory_llm_clients_cache, 'flush_all'):
            litellm.in_memory_llm_clients_cache.flush_all()
        elif hasattr(litellm.in_memory_llm_clients_cache, 'flush_cache'):
            litellm.in_memory_llm_clients_cache.flush_cache()

    yield

    litellm.callbacks = original_callbacks
    litellm.success_callback = original_success
    litellm.failure_callback = original_failure
    litellm._async_success_callback = original_async_success
    litellm._async_failure_callback = original_async_failure
