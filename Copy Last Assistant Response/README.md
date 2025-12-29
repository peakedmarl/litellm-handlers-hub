# Clipboard Copy Handler

A LiteLLM custom callback handler that automatically copies the last assistant's response to clipboard when triggered.

## What It Does

This handler monitors incoming requests and when it detects the keyword `"copyy"` in the current user message, it:
- Extracts the last assistant's response from conversation history
- Copies the response text to your system clipboard
- Returns a confirmation message

## Usage

Add to your `proxy_config.yaml`:

```yaml
litellm_settings:
  custom_callbacks:
    - copy_handler.copy_to_clipboard
```

Then simply include `"copyy"` in any user message to trigger the copy action.

## Requirements

- **pyperclip library**: `pip install pyperclip`

## Key Features

- **Smart content extraction** - Handles multiple response formats:
  - Plain text strings (OpenAI-format)
  - Claude Code format (Anthropic-structured content blocks)
  - Dictionary and list formats
- **Conversation-aware** - Only copies from message history, not the current request
- **User-friendly feedback** - Returns clear confirmation/error messages
- **Error handling** - Gracefully handles clipboard failures

## Example

**User message:**
```
Can you explain quantum computing? copyy
```

**Handler action:**
- Copies the previous assistant's explanation to clipboard
- Returns: `✅ Copied the previous response to clipboard!`

## Notes

- The keyword `"copyy"` is case-insensitive
- Only the last non-empty assistant response is copied
- Requires clipboard permissions on your operating system
