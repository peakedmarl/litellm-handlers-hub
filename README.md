# LiteLLM Handlers Hub 🚀

A curated collection of custom LiteLLM handlers for enhanced functionality and advanced use cases.

## What This Is

This hub contains production-ready LiteLLM custom callbacks that extend the capabilities of LiteLLM's proxy and completion systems. Each handler is self-contained, well-documented, and ready to drop into your LiteLLM configuration.

## Available Handlers

### 🧠 DeepSeek V3.2 Context Management System
**Dual-format handler for OpenAI and Anthropic endpoints**

Implements DeepSeek V3.2's Thinking Context Management strategy across both OpenAI-compatible and Anthropic-native LiteLLM endpoints.

**Key Features:**
- Automatic format detection (OpenAI vs. Anthropic)
- Preserves reasoning during tool interactions
- Discards reasoning on new user input
- Always preserves tool history
- Unified reasoning extraction across formats

**Use Case:** Multi-turn conversations with tool calling where you want to optimize token usage while maintaining reasoning context.

**Location:** `DeepSeek V3.2 Context Management System/`

---

### 📋 Copy Last Assistant Response
**Clipboard copy handler with smart content extraction**

Automatically copies the last assistant's response to clipboard when triggered by a keyword.

**Key Features:**
- Smart content extraction (handles multiple response formats)
- Conversation-aware (only copies from history)
- User-friendly feedback messages
- Error handling for clipboard failures

**Use Case:** Quick copying of assistant responses for documentation, sharing, or further processing.

**Location:** `Copy Last Assistant Response/`

---

## Installation

1. Clone this repository:
```bash
git clone https://github.com/peakedmarl/litellm-handlers-hub.git
cd litellm-handlers-hub
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Add handlers to your LiteLLM `proxy_config.yaml`:
```yaml
litellm_settings:
  custom_callbacks:
    - DeepSeek V3.2 Context Management System.deepseek_context_handler.dual_format_handler
    - Copy Last Assistant Response.copy_handler.copy_to_clipboard
```

## Structure

```
litellm-handlers-hub/
├── handlers/
│   ├── DeepSeek V3.2 Context Management System/
│   │   ├── deepseek_context_handler.py
│   │   └── README.md
│   └── Copy Last Assistant Response/
│       ├── copy_handler.py
│       └── README.md
├── .gitignore
└── README.md
```

Each handler has its own directory containing:
- The handler code (`.py`)
- A README with usage instructions and examples

## Contributing

Got a cool LiteLLM handler? We'd love to add it to the hub!

1. Create a new directory for your handler
2. Add your handler code and a README
3. Submit a pull request

**Guidelines:**
- Handlers should be well-documented
- Include usage examples in the README
- Handle errors gracefully
- Log important actions for debugging

## Requirements

- Python 3.8+
- LiteLLM 1.0+
- Handler-specific dependencies (see individual handler READMEs)

## License

MIT License - feel free to use these handlers in your projects!

## Credits

Built with ❤️ for the LiteLLM community

---

**Got questions or ideas?** Open an issue or start a discussion! 🚀