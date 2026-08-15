# Tools and MCP Adapter

This package defines the first minimal tool boundary for RoboClaw Next.

The core idea is:

```text
MCP server tool
    -> MCPClientRuntime
    -> MCPToolAdapter
    -> ToolRegistry
    -> LLM tool schema / tool execution
```

## Files

- `base.py`: defines `AgentTool`, `ToolExecutionContext`, and `ToolResult`.
- `registry.py`: registers tools and invokes them by name.
- `mcp_runtime.py`: connects to a stdio MCP server with the official Python MCP SDK.
- `mcp_adapter.py`: converts MCP tools into RoboClaw Next tools.

## Run the local MCP adapter demo

From the repository root:

```bash
PYTHONPATH=. uv run --no-project --with "mcp[cli]<2" \
    python roboclaw_next/examples/mcp_adapter_demo.py
```

This starts `mcp_demo_server.py` as a stdio MCP server, discovers its tools, converts
them into RoboClaw Next tools, prints their OpenAI-compatible schemas, and calls:

- `demo__add`
- `demo__echo`

## Optional LLM loop demo

With OpenAI:

```bash
OPENAI_API_KEY=... PYTHONPATH=. uv run --no-project \
    --with "mcp[cli]<2" --with openai \
    python roboclaw_next/examples/mcp_llm_loop_demo.py
```

With DeepSeek:

```bash
DEEPSEEK_API_KEY=... ROBOCLAW_LLM_PROVIDER=deepseek PYTHONPATH=. \
    uv run --no-project --with "mcp[cli]<2" --with openai \
    python roboclaw_next/examples/mcp_llm_loop_demo.py
```

## Dependency note

The current root RoboClaw project already pins `mcp>=1.26.0,<2.0.0`, so this
prototype targets the official MCP Python SDK v1 line. The v2 SDK has a cleaner
high-level `Client` API, but adopting it in the root project would conflict with
the existing RoboClaw implementation until that code is migrated.
