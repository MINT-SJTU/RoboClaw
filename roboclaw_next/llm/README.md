# LLM Providers

This package contains the minimal LLM boundary for RoboClaw Next.

The current implementation intentionally supports only OpenAI-compatible chat APIs:

- `openai`
- `deepseek`

Both providers share the same normalized interface:

```python
from roboclaw_next.llm import create_llm_provider

provider = create_llm_provider("deepseek")

response = await provider.chat_with_retry([
    {"role": "system", "content": "You are RoboClaw."},
    {"role": "user", "content": "Summarize the current robot task."},
])

print(response.content)
```

Environment variables:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` optional
- `OPENAI_MODEL` optional
- `DEEPSEEK_API_KEY`
- `DEEPSEEK_BASE_URL` optional, defaults to `https://api.deepseek.com`
- `DEEPSEEK_MODEL` optional, defaults to `deepseek-chat`

Design notes:

- The provider interface returns `LLMResponse`, independent of vendor SDK objects.
- Tool calls are normalized into `ToolCall`.
- OpenAI and DeepSeek are handled by one `OpenAICompatibleProvider`.
- Larger concerns such as multi-provider registry, OAuth, Azure, LiteLLM routing, memory, and agent scheduling should live outside this package.
