# Tools

This package defines the first minimal tool boundary for RoboClaw Next.

The core idea is:

```text
AgentTool
    -> ToolRegistry
    -> LLM tool schema / tool execution
```

## Files

- `base.py`: defines `AgentTool`, `ToolExecutionContext`, and `ToolResult`.
- `registry.py`: registers tools and invokes them by name.
- `mcp_runtime.py`: connects to a stdio MCP server with the official Python MCP SDK.
- `mcp_adapter.py`: converts MCP tools into the same `AgentTool` shape.

## MCP Tool-Call Chain

The main end-to-end example is:

```bash
DEEPSEEK_API_KEY=... ROBOCLAW_LLM_PROVIDER=deepseek PYTHONPATH=. \
    uv run --no-project --with openai --with "mcp[cli]<2" \
    python roboclaw_next/examples/mcp_tool_call_demo.py
```
