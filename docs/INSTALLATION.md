# RoboClaw Installation Guide

This guide is the native host installation path. If you want Docker-based workflows, use:

- [Docker Installation](./DOCKERINSTALLATION.md)

## 1. Prerequisites

Start from a clean clone:

```bash
git clone https://github.com/MINT-SJTU/RoboClaw.git
cd RoboClaw
```

## 2. Install RoboClaw

Install the package in editable mode:

```bash
pip install -e ".[dev]"
```

After installation, the `roboclaw` command should be available:

```bash
roboclaw --help
```

Expected result:

- commands such as `onboard`, `status`, `agent`, and `provider` are listed

## 3. Initialize RoboClaw

Run:

```bash
roboclaw onboard
```

This should create `~/.roboclaw/config.json`, `~/.roboclaw/workspace/`, and the initial workspace scaffold. You can verify it with:

```bash
find ~/.roboclaw -maxdepth 4 -type f | sort
```

You should see at least:

```text
~/.roboclaw/config.json
~/.roboclaw/workspace/AGENTS.md
~/.roboclaw/workspace/HEARTBEAT.md
~/.roboclaw/workspace/SOUL.md
~/.roboclaw/workspace/TOOLS.md
~/.roboclaw/workspace/USER.md
~/.roboclaw/workspace/memory/MEMORY.md
```

## 4. Verify Status Output

Run:

```bash
roboclaw status
```

Check that:

- `Config` is shown as `✓`
- `Workspace` is shown as `✓`
- the current `Model` looks correct
- provider status matches the actual state of your machine

## 5. Configure the Model Provider

Before testing `roboclaw agent`, make sure the model provider is configured.

First run:

```bash
roboclaw status
```

This tells you which providers are already available on the current machine.

Two common cases:

### 5.1 OAuth provider

If you are using an OAuth-based provider, log in directly.

The current codebase supports:

```bash
roboclaw provider login openai-codex
roboclaw provider login github-copilot
```

### 5.2 API key provider

If you are using an API-key-based provider, edit:

```bash
~/.roboclaw/config.json
```

Fill in the provider key and default model there.

Common API key providers include:

- `openai`
- `anthropic`
- `openrouter`
- `deepseek`
- `gemini`
- `zhipu`
- `dashscope`
- `moonshot`
- `minimax`
- `siliconflow`
- `volcengine`
- `azureOpenai`
- `custom`
- `vllm`

Then run:

```bash
roboclaw status
```

Check that:

- the current `Model` is correct
- the provider you want to use is no longer `not set`

## 6. Verify the Basic Model Path

Run one minimal message to confirm that RoboClaw can respond:

```bash
roboclaw agent -m "hello"
```

Check that:

- the agent starts successfully
- the agent returns a normal reply
- failures point clearly to model configuration, provider setup, network, or permissions
