# RoboClaw Next

This directory is a workspace for rethinking and rebuilding RoboClaw's future architecture.

It is intentionally separated from the current `roboclaw/` implementation so larger design experiments, architectural notes, and prototype modules can evolve without disturbing the existing codebase.

Current prototype areas:

- `llm/`: minimal OpenAI-compatible provider boundary for OpenAI and DeepSeek.
- `tools/`: minimal AgentTool, ToolRegistry, and MCP adapter boundary.
- `agent/`: minimal LLM tool-call loop prototype.
- `examples/`: small runnable examples for learning each boundary.
