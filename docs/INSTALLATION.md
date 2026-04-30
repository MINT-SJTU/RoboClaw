# RoboClaw Installation Guide

This guide is the native host installation path. RoboClaw uses `uv` as the only supported Python environment and dependency workflow. If you want Docker-based workflows, use:

- [Docker Installation](./DOCKERINSTALLATION.md)

## 1. Install uv

Install `uv` with the official installer:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Verify that `uv` is available:

```bash
uv --version
```

## 2. Clone RoboClaw

Start from a clean clone and fetch submodules:

```bash
git clone --recurse-submodules https://github.com/MINT-SJTU/RoboClaw.git
cd RoboClaw
```

The embodied engine lives in `roboclaw/embodied/engine` as a git submodule and is required during installation.

If you already cloned the repository without submodules, run:

```bash
git submodule update --init --recursive
```

## 3. Sync the Project with uv

RoboClaw pins Python with `.python-version` and uses `uv sync` to create `.venv` and install all default development dependencies:

```bash
uv sync
```

After sync, verify that the `roboclaw` command is available:

```bash
uv run roboclaw --help
```

Expected result:

- commands such as `onboard`, `status`, `agent`, and `provider` are listed

## 4. Initialize RoboClaw

Run:

```bash
uv run roboclaw onboard
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

## 5. Verify Status Output

Run:

```bash
uv run roboclaw status
```

Check that:

- `Config` is shown as `✓`
- `Workspace` is shown as `✓`
- the current `Model` looks correct
- provider status matches the actual state of your machine

## 6. Configure the Model Provider

Before testing `roboclaw agent`, make sure the model provider is configured.

First run:

```bash
uv run roboclaw status
```

This tells you which providers are already available on the current machine.

Two common cases:

### 5.1 OAuth provider

If you are using an OAuth-based provider, log in directly.

The current codebase supports:

```bash
uv run roboclaw provider login openai-codex
uv run roboclaw provider login github-copilot
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
uv run roboclaw status
```

Check that:

- the current `Model` is correct
- the provider you want to use is no longer `not set`

## 7. Verify the Basic Model Path

Run one minimal message to confirm that RoboClaw can respond:

```bash
uv run roboclaw agent -m "hello"
```

Check that:

- the agent starts successfully
- the agent returns a normal reply
- failures point clearly to model configuration, provider setup, network, or permissions

## 8. Launch the Web Dashboard

The web dashboard provides a browser-based UI for chatting with RoboClaw.

Install the frontend dependencies:

```bash
cd ui
npm install
```

### Production Mode

Build the frontend and start the server:

```bash
cd ui && npm run build && cd ..
uv run roboclaw web start
```

Open **http://127.0.0.1:8765** in your browser.

### Development Mode (with hot reload)

```bash
# Terminal 1: start backend
uv run roboclaw web start

# Terminal 2: start frontend dev server
cd ui
npm run dev
```

Open **http://localhost:5173** in your browser. The Vite dev server proxies `/api` and `/ws` to the backend automatically.

### Options

```bash
uv run roboclaw web start --host 0.0.0.0 --port 9000
```

| Flag          | Default       | Description                |
|---------------|---------------|----------------------------|
| `--host`      | `127.0.0.1`  | Bind address               |
| `--port`      | `8765`        | Port number                |
| `--workspace` | `~/.roboclaw/workspace` | Workspace directory |
| `--verbose`   | off           | Enable debug logging       |
