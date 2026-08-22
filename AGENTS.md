# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

> `CLAUDE.md` is a symlink to `AGENTS.md`. Edit `AGENTS.md`; both views stay in sync.

---

## 工作规范

运用第一性原理思考，拒绝经验主义和路径盲从，不要假设我完全清楚目标，保持审慎，从原始需求和问题出发，若目标模糊请停下和我讨论，若目标清晰但路径非最优，请直接建议更短、更低成本的办法。

### 架构原则

- 对象优先：所有功能从对象建模出发——先识别实体与职责边界，再设计对象间的协作关系。
- 物理文件组织必须直接反映逻辑架构。目录名、文件名、包结构要让人一眼看出模块职责和层级关系。如果架构调整了，文件结构必须同步调整。

### 代码规范

- 框架代码放 `roboclaw/embodied/`，用户资产放 `~/.roboclaw/workspace/embodied/`。
- 不写向后兼容代码。不保留旧接口、不做 fallback 适配、不写 deprecated wrapper。旧的不要了就直接删掉。
- 不用 try/except 吞错误。有报错就让它直接抛出来，不要静默捕获。只在确实需要处理特定异常时才 catch。
- 复用优先。已有实现的功能不要重复造轮子，先找现有代码再决定是否新写。
- 单个 .py 文件不超过 1000 行。超过时必须将独立逻辑拆分到单独的模块。
- 嵌套不超过 3 层缩进。超过时应将内层逻辑提取为独立函数。

---

## Commands

This project uses `uv` as the **only** supported Python workflow. The embodied engine (`roboclaw/embodied/engine`) is a git submodule — always clone with `--recurse-submodules`, or run `git submodule update --init --recursive` after the fact.

### Environment

```bash
uv sync                          # create .venv and install dev deps
uv run roboclaw --help           # verify CLI entry is wired
uv run roboclaw onboard          # scaffold ~/.roboclaw/{config.json, workspace/}
uv run roboclaw status           # show config / workspace / provider state
```

### Running the agent

```bash
uv run roboclaw agent -m "hello"                          # one-shot
uv run roboclaw agent                                      # interactive REPL
uv run roboclaw agent --session cli:<id>                   # resume a session
uv run roboclaw gateway                                    # multi-channel bus gateway
uv run roboclaw web start [--host 0.0.0.0 --port 9000]     # FastAPI + dashboard
uv run roboclaw provider login openai-codex|github-copilot # OAuth providers
```

For dashboard development, run `uv run roboclaw web start` in one terminal and `cd ui && npm install && npm run dev` in another. Vite proxies `/api` and `/ws` to the backend.

### Tests & lint

```bash
uv run pytest                                  # full suite (asyncio_mode=auto)
uv run pytest tests/test_commands.py           # single file
uv run pytest tests/test_commands.py::test_x   # single test
uv run pytest -m "not hardware"                # skip tests that need physical robots
uv run pytest -m pty                           # PTY integration tests (pexpect)
uv run ruff check roboclaw tests               # lint (line-length 100, E/F/I/N/W)
```

Custom pytest markers: `hardware` (real motors/cameras required) and `pty` (spawns a real agent subprocess via pexpect). CI runs the full suite on Python 3.11/3.12/3.13.

### WhatsApp bridge (Node)

`bridge/` is a TypeScript Baileys-based bridge built separately: `cd bridge && npm install && npm run build`. Docker image builds it automatically.

---

## Architecture

RoboClaw is an embodied-AI personal assistant: a channel-agnostic agent loop that speaks to LLMs, orchestrates tools, and drives physical robot arms via a LeRobot fork.

### Top-level layout

| Path | Role |
|------|------|
| `roboclaw/cli/commands.py` | Typer CLI — `onboard`, `agent`, `gateway`, `web`, `channels`, `provider`, `dev`, `plugins` subapps. |
| `roboclaw/agent/` | The core `AgentLoop` plus `SessionManager`, `ContextBuilder`, `MemoryConsolidator`, `SubagentManager`, and all agent tools. |
| `roboclaw/bus/` | In-process `MessageBus` (`InboundMessage` / `OutboundMessage`) shared by CLI, gateway, channels, and web. |
| `roboclaw/channels/` | Pluggable messaging adapters: CLI, web, Feishu, Slack, Telegram, DingTalk, Matrix, WeCom, Discord, QQ, WhatsApp, Email. Discovery via `roboclaw/channels/registry.py`. |
| `roboclaw/providers/` | LLM provider abstraction (`factory.build_provider`): `litellm`, `openai`, `azure_openai`, `custom`, `openai-codex`, `github-copilot`. |
| `roboclaw/http/` | FastAPI server (`server.py`), runtime wiring (`runtime.py`), and REST routes (`routes/*`) for the web dashboard. |
| `roboclaw/embodied/` | Embodied-AI stack — see below. |
| `roboclaw/skills/` | Built-in skills (Markdown + YAML frontmatter, OpenClaw-compatible): `github`, `weather`, `summarize`, `tmux`, `clawhub`, `skill-creator`. |
| `roboclaw/cron/`, `roboclaw/heartbeat/` | Scheduled jobs + periodic agent heartbeat. |
| `roboclaw/config/` | Pydantic `Config` schema (`schema.py`), loader (`loader.py`), and path resolution (`paths.py`). |
| `roboclaw/data/` | Dataset curation/exploration APIs used by the web dashboard. |
| `roboclaw/templates/` | `AGENTS.md`, `SOUL.md`, `TOOLS.md`, `USER.md`, `HEARTBEAT.md`, seed `memory/` — synced into the workspace by `sync_workspace_templates`. |
| `ui/` | Vite + React + Tailwind dashboard. |
| `bridge/` | Node.js WhatsApp bridge (Baileys). |
| `roboclaw/embodied/engine/` | **git submodule** — LeRobot fork `lerobot-roboclaw` (installed via `[tool.uv.sources]`). |

### Framework vs. user data split

Hard rule baked into `CLAUDE.md`: framework code lives in `roboclaw/embodied/`, user assets live in `~/.roboclaw/workspace/embodied/`. Runtime paths (media, cron, logs, CLI history, bridge install) are resolved through `roboclaw/config/paths.py` and the active config path — not from hardcoded home dirs.

### Agent loop ([loop.py](roboclaw/agent/loop.py))

`AgentLoop` is the processing engine:
1. Receive an `InboundMessage` from the bus (or direct via `process_direct`).
2. `ContextBuilder` assembles system prompt + session history + memory + workspace templates (`SOUL.md`, `TOOLS.md`, `USER.md`, `AGENTS.md`).
3. Call the LLM provider, then execute any returned tool calls via the `ToolRegistry`.
4. `MemoryConsolidator` periodically distills history into `~/.roboclaw/workspace/memory/`.
5. Emit `OutboundMessage` back on the bus; channel adapters fan out.

Tools live in `roboclaw/agent/tools/` and all subclass `Tool` ([base.py](roboclaw/agent/tools/base.py)) which enforces JSON-schema parameter validation + casting. Built-ins: `ReadFileTool`, `WriteFileTool`, `EditFileTool`, `ListDirTool`, `ExecTool`, `WebFetchTool`, `WebSearchTool`, `SpawnTool` (subagents), `CronTool`, `MessageTool`, `MCPTool`. MCP servers declared in config are mounted as dynamic tools.

### Channel & gateway model

`ChannelManager` discovers channel classes through `channels/registry.py` (entry points + built-ins), instantiates enabled ones based on `Config.channels`, and wires them to the shared `MessageBus`. `gateway` and `web start` both spin up the same runtime (AgentLoop + CronService + HeartbeatService + ChannelManager) — web adds a FastAPI layer on top.

### Embodied subsystem (`roboclaw/embodied/`)

The embodied stack is the most involved part of the codebase — read files together, not in isolation.

- **`service/__init__.py` — `EmbodiedService`**: single point of control for physical operations. Holds a two-layer mutex (thread lock + cross-process file lock via `EmbodimentFileLock`) so only one teleop/record/replay/infer/calibrate/train session runs at a time. Exposes `start_teleop`, `start_recording`, `start_replay`, `start_inference`, `start_calibration`, plus manifest mutations (`bind_arm`, `bind_camera`, `bind_hand`, …). Sub-services: `calibration`, `setup`, `teleop`, `record`, `replay`, `train`, `infer`, `hub`, `doctor`.
- **`board/` — the 看板 / Board**: per-embodiment pub/sub state hub. `OutputConsumer` parses subprocess stdout into structured state; agent + dashboard read state, post commands (save/discard/skip/stop) back. All channels route through `board/channels.py`.
- **`command/` — argv builders**: `CommandBuilder` turns a Manifest + parameters into LeRobot CLI argv; `wrapper.py` is the actual subprocess entrypoint (`python -m roboclaw.embodied.command.wrapper <action>`); `headless_patch.py` disables interactive prompts when the agent drives LeRobot.
- **`embodiment/` — hardware model**: `manifest/` serializes which arms/cameras/hands are bound and their roles (follower/leader, side). `arm/`, `hand/`, `hardware/`, `interface/` (serial/CAN/video) cover the physical layer. `doctor.py` runs health checks; `lock.py` owns the cross-process file lock.
- **`engine/`**: **git submodule** — the LeRobot fork that actually implements `teleoperate`/`record`/`replay`/`train`/`eval`. Exposed as a local package via `[tool.uv.sources] lerobot = { path = "roboclaw/embodied/engine" }`.
- **`executor.py` — `SubprocessExecutor`**: async subprocess runner that injects UTF-8 and HuggingFace env (`HF_ENDPOINT`, `HF_TOKEN`, `HTTPS_PROXY`) from config.
- **`toolkit/` — agent-facing**: `tools.py` exposes the embodied tools to the agent; `tty.py` implements a TTY handoff so the CLI can yield the terminal to an interactive LeRobot subprocess (calibration, teleop) and reclaim it afterward.

### Workspace files

After `onboard`, `~/.roboclaw/workspace/` contains `AGENTS.md`, `HEARTBEAT.md`, `SOUL.md`, `TOOLS.md`, `USER.md`, and `memory/MEMORY.md`. These are loaded into the agent's context on every turn and are the primary place to persist persona, tool hints, user profile, and long-term memory. `sync_workspace_templates` refreshes them from `roboclaw/templates/` without clobbering user edits.

### Config

Config lives at `~/.roboclaw/config.json` (override with `--config`). `load_runtime_config` returns a Pydantic `Config`; `_make_provider` / `build_provider` selects the LLM provider. `contextWindowTokens` has replaced the deprecated `memoryWindow` field — an onboard refresh warns when old configs are detected.
