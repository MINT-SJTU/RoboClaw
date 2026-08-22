# Docker Installation

This guide is the minimal stateless Docker path for RoboClaw — a pure-LLM
agent + dashboard sandbox, no hardware, no persistent state beyond a
bind-mounted config volume.

**Pick the right install path for your use case:**

| Use case | Guide |
|----------|-------|
| Native Linux / macOS, or WSL2 running Python directly | [INSTALLATION.md](./INSTALLATION.md) |
| Minimal, stateless Docker for a pure-LLM agent / dashboard (no hardware) | **this document** |
| Windows 11 + WSL2, with SO-101 arms + cameras to drive via `usbipd` passthrough | [WSL2 + Docker Deployment](./WSL2_DOCKER_DEPLOYMENT.md) |

This guide does NOT cover: USB/serial device passthrough, camera access, the
embodied SO-101 workflow, OAuth provider login through a Windows browser, or
the systemd interop guard. If you need any of those, use the WSL2+Docker
guide above.

## 1. Prerequisites

Start from a clean clone with submodules populated — the Dockerfile's stage-2
has a defensive guard that fails the build if `roboclaw/embodied/engine/` is
empty. The submodule vendors a LeRobot fork; see
[SO101_BIMANUAL_DRIVER.md](./SO101_BIMANUAL_DRIVER.md) for the rationale.

```bash
git clone --recurse-submodules https://github.com/MINT-SJTU/RoboClaw.git
cd RoboClaw
# If you cloned without --recurse-submodules:
git submodule update --init --recursive
```

## 2. Build the Docker Image

```bash
docker build -t roboclaw .
```

## 3. Initialize RoboClaw

```bash
docker run -v ~/.roboclaw:/root/.roboclaw --rm roboclaw onboard
```

## 4. Configure the Model Provider

Edit `~/.roboclaw/config.json` on your host to add API keys or provider settings. See [INSTALLATION.md](./INSTALLATION.md#6-configure-the-model-provider) for provider details.

## 5. Verify Inside Docker

```bash
docker run -v ~/.roboclaw:/root/.roboclaw --rm roboclaw status
```

Check that:

- `Config` is shown as `✓`
- `Workspace` is shown as `✓`
- the current `Model` is correct

## 6. Run the Agent

```bash
docker run -v ~/.roboclaw:/root/.roboclaw --rm roboclaw agent -m "hello"
```

## 7. Run the Gateway

```bash
docker run -v ~/.roboclaw:/root/.roboclaw -p 18790:18790 roboclaw gateway
```

## 8. Docker Compose

You can also use Docker Compose:

```bash
docker compose run --rm roboclaw-cli onboard     # first-time setup
docker compose up -d roboclaw-gateway             # start gateway
docker compose run --rm roboclaw-cli agent -m "Hello!"
docker compose logs -f roboclaw-gateway           # view logs
```
