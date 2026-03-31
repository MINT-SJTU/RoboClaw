# Development Guide

## Prerequisites

- Python 3.11+
- Git

## Local Setup

```bash
# Clone & install in editable mode with dev extras
git clone https://github.com/MINT-SJTU/RoboClaw.git
cd RoboClaw
pip install -e ".[dev]"

# First-time setup (creates ~/.roboclaw/config.json & workspace)
roboclaw onboard
```

## Running Tests

```bash
# Unit tests (no hardware required)
python -m pytest tests/ -x -q

# Skip PTY integration tests (useful in minimal CI environments)
python -m pytest tests/ -x -q -m "not pty"

# Run only PTY integration tests
python -m pytest tests/integration/ -x -q -m pty
```

## Stub Mode

Set `ROBOCLAW_STUB=1` to replace real hardware calls with deterministic
stubs.  This allows the full embodied pipeline (scan, identify, calibrate,
teleoperate, record) to run on a laptop without any robot arms or cameras.

```bash
# Run the agent in stub mode
ROBOCLAW_STUB=1 roboclaw agent

# PTY integration tests use stub mode automatically
python -m pytest tests/integration/ -x -q -m pty
```

What gets stubbed:

| Component | Real behaviour | Stub behaviour |
|---|---|---|
| `scan_serial_ports()` | reads `/dev/serial/by-*` | returns 2 fake ports |
| `scan_cameras()` | probes `/dev/video*` via OpenCV | returns 1 fake camera |
| `probe_port()` | reads Feetech motor positions | returns motor IDs [1..6] |
| `read_positions()` | reads motor positions via SCS | returns all zeros |
| `run_interactive()` | spawns a subprocess | returns exit-code 0 immediately |
| `_find_moved_port()` | reads motor positions | picks scripted port |

All stub defaults are overridable via env vars for per-test flexibility:

| Variable | Default |
|---|---|
| `ROBOCLAW_STUB_PORTS` | 2 fake serial ports (JSON list) |
| `ROBOCLAW_STUB_CAMERAS` | 1 fake camera (JSON list) |
| `ROBOCLAW_STUB_MOTORS` | 6 motors per port (JSON object) |
| `ROBOCLAW_STUB_MOVED_PORT` | first port's by_id |

All stub logic lives in `roboclaw/embodied/stub.py`.

## Workspace Reset

During development you may want to return to a clean state:

```bash
# Interactive (asks for confirmation)
roboclaw dev reset

# Non-interactive
roboclaw dev reset --yes

# Reset and configure a specific model
roboclaw dev reset --yes --model openai/gpt-4o --api-key sk-...
```

This deletes `~/.roboclaw/workspace` and `~/.roboclaw/config.json`, then
re-runs `roboclaw onboard` non-interactively.

## Environment Variables

| Variable | Purpose |
|---|---|
| `ROBOCLAW_HOME` | Override the base directory (default `~/.roboclaw`). Useful for tests and parallel instances. |
| `ROBOCLAW_STUB` | Set to `1` to activate stub mode (fake hardware). |
| `ROBOCLAW_STUB_PORTS` | JSON list of fake serial ports (override defaults). |
| `ROBOCLAW_STUB_CAMERAS` | JSON list of fake cameras (override defaults). |
| `ROBOCLAW_STUB_MOTORS` | JSON object mapping port by_id → motor ids. |
| `ROBOCLAW_STUB_MOVED_PORT` | by_id of port that identify detects as moved. |

## Troubleshooting

### `ModuleNotFoundError: No module named 'lerobot'`

LeRobot is an optional dependency under the `research` extra:

```bash
pip install -e ".[research]"
```

### PTY tests fail with `ModuleNotFoundError: No module named 'pexpect'`

Install the dev extra:

```bash
pip install -e ".[dev]"
```

### Terminal messed up after Ctrl-C

Run `reset` in your shell to restore terminal settings.

### Tests fail with `roboclaw.config` import errors

Make sure you installed in editable mode:

```bash
pip install -e ".[dev]"
```
