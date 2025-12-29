# Dual-Format Context Handler

A LiteLLM custom callback handler that implements DeepSeek V3.2's Thinking Context Management across both OpenAI-compatible and Anthropic-native endpoints.

## What It Does

This handler automatically manages reasoning context in multi-turn conversations with tool calling, applying three core rules:

1. **Preserve reasoning during tool interactions** - Keeps all thinking content when only tool messages are exchanged
2. **Discard reasoning on new user input** - Strips reasoning when a new user message arrives
3. **Always preserve tool history** - Tool calls and results remain intact regardless of filtering

## Supported Formats

- **OpenAI-compatible**: `/v1/chat/completions` (DeepSeek, OpenAI, etc.)
- **Anthropic-native**: `/v1/messages` (Claude models)

Automatic format detection based on model name, headers, and metadata.

## Research Foundation

This implementation replicates the **Thinking Context Management** strategy from the DeepSeek-V3.2 paper:

> *DeepSeek-R1 has demonstrated that incorporating a thinking process can significantly enhance a model's ability to solve complex problems. Building on this insight, we aim to integrate thinking capabilities into tool-calling scenarios.*
>
> *We observed that replicating DeepSeek-R1's strategy—discarding reasoning content upon the arrival of the second round of messages—results in significant token inefficiency. This approach forces the model to redundantly re-reason through the entire problem for each subsequent tool call.*
>
> **Context Management Rules:**
> - Historical reasoning content is discarded only when a new user message is introduced to the conversation. If only tool-related messages (e.g., tool outputs) are appended, the reasoning content is retained throughout the interaction.
> - When reasoning traces are removed, the history of tool calls and their results remains preserved in the context.

## Usage

Add to your `proxy_config.yaml`:

```yaml
litellm_settings:
  custom_callbacks:
    - dual_format_context_handler.dual_format_handler
```

## Key Features

- **Automatic format detection** - No manual configuration needed
- **Unified reasoning extraction** - Handles both `<think>` tags (OpenAI) and thinking blocks (Anthropic)
- **Tool-aware filtering** - Preserves tool calls and results across all formats
- **Error-resilient** - Falls back to original messages on any errors
- **Comprehensive logging** - Clear visibility into filtering decisions

## Note

Certain agent frameworks (e.g., Roo Code, Terminus) simulate tool interactions via user messages. These frameworks may not fully benefit from enhanced reasoning persistence due to the context management rules. Non-thinking models are recommended for optimal performance with such architectures.