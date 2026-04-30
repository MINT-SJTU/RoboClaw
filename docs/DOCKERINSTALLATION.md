# Docker Installation

This guide is the Docker installation path for RoboClaw.

If you do not want Docker, use [INSTALLATION.md](./INSTALLATION.md).

## 1. Prerequisites

Start from a clean clone:

```bash
git clone --recurse-submodules https://github.com/MINT-SJTU/RoboClaw.git
cd RoboClaw
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

Edit `~/.roboclaw/config.json` on your host to add API keys or provider settings. See [INSTALLATION.md](./INSTALLATION.md#5-configure-the-model-provider) for provider details.

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
