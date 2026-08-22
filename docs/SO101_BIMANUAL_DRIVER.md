# SO-101 Bimanual Driver — Source Selection Guide

This doc captures *where the SO-101 driver actually comes from* in the RoboClaw
deployment, why we vendor a LeRobot fork instead of pinning the upstream PyPI
release, and how to verify the bimanual configuration names against current
upstream.

It exists because the question "is SO-101 in upstream LeRobot?" has a nuanced
answer: **single-arm yes, bimanual config-name aliases probably need the
fork** — and that nuance has bitten implementers more than once.

---

## 1. TL;DR — pick a source

| Use case | Recommended source | Why |
|----------|-------------------|-----|
| Single-arm SO-101 (one leader OR one follower) | **Upstream HuggingFace LeRobot** (`pip install lerobot[feetech]`) | First-party `SO101FollowerConfig` + `SO101LeaderConfig`. Zero fork dependency. |
| Bimanual SO-101 (2 leaders driving 2 followers) — this deployment | **RoboClaw's LeRobot fork** ([Elvin-yk/lerobot-roboclaw](https://github.com/Elvin-yk/lerobot-roboclaw) branch `roboclaw`, vendored at `roboclaw/embodied/engine/` via `[tool.uv.sources]` in [pyproject.toml](../pyproject.toml)) | Known-good `bi_so_follower` / `bi_so_leader` config names referenced by [roboclaw/embodied/command/builder.py:22-25](../roboclaw/embodied/command/builder.py); also carries `headless_patch.py` for non-TTY calibration. |
| Direct programmatic single-arm control (IK/FK, gravity comp, custom RL) | **phosphobot's first-party driver** (`/home/hafnium/phosphobot/phosphobot/phosphobot/hardware/so100.py` — its `SO100Hardware` covers SO-101 too; the dashboard label in `phosphobot/dashboard/src/components/common/add-robot-connection.tsx:70` literally reads `"SO-100 / SO-101"`) | Standalone driver, no LeRobot dependency. Different operating model — see §3. |
| LeRobot-format dataset recording / ACT/Pi0/SmolVLA training | **Upstream HuggingFace LeRobot** (or fork — both store at `~/.cache/huggingface/lerobot/`) | Either works for the dataset format. Stick with whichever your teleop uses to avoid path divergence. |

---

## 2. Three driver sources, side-by-side

### 2.1 Upstream HuggingFace LeRobot

- **Repo**: <https://github.com/huggingface/lerobot>
- **Install**: `uv pip install lerobot[feetech,dynamixel]` (PyPI)
- **SO-101 support**: `lerobot.robots.so101_follower` + `lerobot.robots.so101_leader` configs are first-party, declared with the same hardware spec RoboClaw mirrors in [`roboclaw/embodied/embodiment/arm/registry.py:46-55`](../roboclaw/embodied/embodiment/arm/registry.py) — Feetech protocol, motor IDs `(1, 2, 3, 4, 5, 6)`, 1 Mbps baudrate, `sts3215` motor model, joint set `shoulder_pan, shoulder_lift, elbow_flex, wrist_flex, wrist_roll, gripper`.
- **Bimanual support**: upstream has bimanual coordination code, but the **specific config names** that RoboClaw passes to `lerobot-teleoperate --robot.type=...` (see `_BIMANUAL["so101"] = ("bi_so_follower", "bi_so_leader")` at [`roboclaw/embodied/command/builder.py:22-25`](../roboclaw/embodied/command/builder.py)) may use different identifiers upstream (LeRobot historically prefers model-suffixed names like `bi_so100_*`). Run the probe in §4 against a fresh upstream install to confirm exactly which names work.
- **Servo bus**: `lerobot.motors.feetech.FeetechMotorsBus`. Pure Python, pyserial, direct serial protocol — no ROS2.
- **Dataset/training**: full HuggingFace dataset format integration; downstream `lerobot-train act_so101_*` works out of the box.

### 2.2 RoboClaw's LeRobot fork (what this repo actually ships)

- **Repo**: [Elvin-yk/lerobot-roboclaw](https://github.com/Elvin-yk/lerobot-roboclaw), branch `roboclaw`
- **Vendored at**: `roboclaw/embodied/engine/` (git submodule; see [.gitmodules](../.gitmodules))
- **Wired into uv via**: `[tool.uv.sources]` block in [pyproject.toml](../pyproject.toml#L89-L90):
  ```toml
  [tool.uv.sources]
  lerobot = { path = "roboclaw/embodied/engine" }
  ```
  This overrides the PyPI `lerobot` dependency (`pyproject.toml:51`) with the local-path install. Same package import surface, different commit graph.
- **Bimanual support**: known-good for `bi_so_follower` / `bi_so_leader` because RoboClaw's own [`command/builder.py:22-25`](../roboclaw/embodied/command/builder.py) hardcodes those names in `_BIMANUAL["so101"]` and uses them in production. If those names didn't work, RoboClaw bimanual mode would have shipped broken — empirically it doesn't.
- **Headless calibration patches**: [`roboclaw/embodied/command/headless_patch.py`](../roboclaw/embodied/command/headless_patch.py) silences LeRobot's interactive prompts (ESC / space) when invoked from a non-TTY context — the agent path. If you swap to upstream LeRobot, the agent's `start_calibration` flow will hang waiting on keypress events that never arrive.
- **Drift policy**: the fork tracks `roboclaw` branch; rebases against upstream `main` happen periodically but are not automatic.

### 2.3 phosphobot's first-party driver

- **Repo**: `/home/hafnium/phosphobot/` (sibling project; see its [CLAUDE.md](/home/hafnium/phosphobot/CLAUDE.md))
- **Driver location**: `phosphobot/phosphobot/hardware/so100.py` (680 LOC `SO100Hardware` class extending `BaseManipulator`) + `phosphobot/phosphobot/hardware/motors/feetech.py` (1066 LOC, full Feetech bus inline with `SCS_SERIES_CONTROL_TABLE` at line 64, baudrate table at line 114, torque/drive/calibration enums at 236/241/246, `JointOutOfRangeError` at line 253).
- **Same model**: SO-100 and SO-101 share servos, baudrate, motor count. phosphobot treats them as one class; the dashboard label at `phosphobot/dashboard/src/components/common/add-robot-connection.tsx:70` reads `"SO-100 / SO-101"` exactly.
- **Bimanual support**: not built in. Two `SO100Hardware` instances with externally-coordinated teleop loops would be required.
- **Concurrency model**: long-lived `FeetechMotorsBus` with worker thread + task queue (`task_queue = queue.Queue()`, `threading.Thread(target=self._worker, daemon=True)`). Single persistent serial connection per arm; multiple async clients submit tasks. Better for interactive scripted control with concurrent gravity-comp.
- **IK/FK**: built-in (`URDF_FILE_PATH`, `inverse_kinematics`, `forward_kinematics`, `get_end_effector_state`). RoboClaw delegates these to LeRobot.
- **No `lerobot` dependency**: phosphobot's `pyproject.toml` and `environment.yml` contain zero lerobot references. The Feetech bus was forked from an early LeRobot commit then drifted independently — frozen control table, no automatic upstream tracking.
- **No ROS2 either**: confirmed by `grep -iE 'rclpy|ament|colcon|ros[12]_'` returning zero hits in either repo's dependency files.

---

## 3. Why three sources exist (the real architecture)

| Concern | Upstream LeRobot | RoboClaw fork | phosphobot |
|---------|------------------|---------------|------------|
| Driver locus | First-party Python package | Submodule of fork | First-party in repo |
| Bimanual SO-101 | Possibly under different config names | `bi_so_follower` / `bi_so_leader` known-working | Not built in |
| LeRobot dataset format | Native | Native (forked from upstream) | Separate format |
| IK/FK exposed as Python API | No (CLI only) | No (inherits LeRobot's CLI-first design) | Yes (direct method calls) |
| Worker-thread concurrency | No (subprocess per action) | No (subprocess per action) | Yes |
| Servo control table | Tracks Feetech upstream | Tracks LeRobot upstream | Frozen at fork point |
| ROS2 | No | No | No |
| Lines of driver code | ~1200 (FeetechMotorsBus + so101_*) | Same (vendored) | ~1700 (so100.py + feetech.py) |

The split is not redundancy — it's three different design centers:

- **Upstream LeRobot**: the canonical *training-and-eval* platform. SDK + CLI for any roboticist to record datasets, train policies, run inference.
- **RoboClaw fork**: a *channel-driven LLM agent on top of LeRobot* — the LLM orchestrates LeRobot's CLI subprocesses for teleop / record / replay / eval. The fork adds bimanual aliases and headless tweaks needed for that orchestration mode.
- **phosphobot**: a *programmatic Python SDK for one arm* — direct method calls, IK/FK, dashboard, RL inference (`am/`). Different surface area from LeRobot (which is CLI/dataset-centric).

---

## 4. Verification probe — confirm what the runtime accepts

The fastest way to know which `bi_so_*` config names your *currently-installed*
`lerobot` recognizes is to ask it directly. The probe is non-destructive — it
attempts to construct each config object and reports `OK` or `NOT FOUND`.

### Inside the deployed RoboClaw container (uses the fork)

```bash
docker compose exec roboclaw-web python -c '
from lerobot.robots import make_robot_config
for name in ["bi_so_follower", "bi_so_leader",
             "bi_so100_follower", "bi_so100_leader",
             "bi_so101_follower", "bi_so101_leader"]:
    try:
        cfg = make_robot_config(name)
        print(f"{name}: OK ({type(cfg).__name__})")
    except Exception as e:
        print(f"{name}: NOT FOUND ({type(e).__name__})")
'
```

**Expected result inside this RoboClaw deployment**: `bi_so_follower` and `bi_so_leader` resolve OK because [`roboclaw/embodied/command/builder.py:23`](../roboclaw/embodied/command/builder.py) ships them in `_BIMANUAL["so101"]` and the deployment's bimanual-mode runs against the fork. The other four names return `NOT FOUND` is consistent with the fork using the generic name.

### Against a fresh upstream LeRobot install

```bash
mkdir -p /tmp/lerobot-upstream-probe
cd /tmp/lerobot-upstream-probe
uv venv
uv pip install 'lerobot[feetech,dynamixel]'
uv run python -c '
from lerobot.robots import make_robot_config
for name in ["bi_so_follower", "bi_so_leader",
             "bi_so100_follower", "bi_so100_leader",
             "bi_so101_follower", "bi_so101_leader"]:
    try:
        cfg = make_robot_config(name)
        print(f"{name}: OK ({type(cfg).__name__})")
    except Exception as e:
        print(f"{name}: NOT FOUND ({type(e).__name__})")
'
```

The diff between the two probe outputs is the bimanual-name divergence
between the fork and upstream — and the answer to "do I need the fork for
bimanual?" The probe takes ~2 minutes (install dominates) and is the
authoritative source rather than this doc's prose.

### When to re-run the probe

- Whenever upstream LeRobot ships a major release (check the [HuggingFace LeRobot releases](https://github.com/huggingface/lerobot/releases)).
- Whenever the fork is rebased — git submodule update inside RoboClaw and re-run.
- Before swapping `[tool.uv.sources] lerobot = { path = ... }` for a PyPI pin.

---

## 5. What the fork carries beyond bimanual aliases

The fork's reasons for existence are not solely the `bi_so_*` config names. Even
if upstream eventually adds them, two more pieces of the fork still matter:

**5.1. `headless_patch.py`** — `roboclaw/embodied/command/headless_patch.py` runs
inside the wrapper subprocess (`python -m roboclaw.embodied.command.wrapper
<action>`) and rewrites LeRobot's interactive-prompt code paths so calibration
finishes without keyboard input. Without this patch, the agent's
`EmbodiedService.start_calibration` flow stalls on the first ESC-confirmation
prompt that LeRobot tries to issue.

**5.2. Calibration JSON path conventions** — both fork and upstream write
calibrations under `~/.cache/huggingface/lerobot/.cache/calibration/<arm-id>.json`.
RoboClaw's `command/builder.py:30` references `_DEFAULT_REPLAY_ROOT = Path("~/.cache/huggingface/lerobot")` for dataset replay, so the path is shared. **No divergence here** — moving from fork to upstream wouldn't break calibration storage.

If you ever migrate the deployment off the fork onto upstream LeRobot:

1. Drop the `[tool.uv.sources]` override in `pyproject.toml`.
2. Move `headless_patch.py`'s logic upstream (PR or maintain locally).
3. Update RoboClaw's `_BIMANUAL["so101"]` mapping in
   `command/builder.py:23` to whatever upstream calls its bimanual SO-101
   configs (verify with the probe in §4).

That's the migration. Until then, the fork is what makes bimanual reliable in
this deployment.

---

## 6. What NOT to do

- **Don't** pin `lerobot` to a specific PyPI version while keeping
  `_BIMANUAL["so101"] = ("bi_so_follower", "bi_so_leader")` if upstream uses
  different names. The mismatch produces a silent fallback to single-arm —
  teleop works but only one arm moves, with no error.
- **Don't** mix phosphobot's `FeetechMotorsBus` with LeRobot's CLI invocations
  inside the same process. Different lifecycle expectations: phosphobot's bus is
  long-lived with a worker thread, LeRobot's CLI subprocess opens its own
  serial connection per invocation. They will fight over the `/dev/ttyACM*`
  port.
- **Don't** keep both `roboclaw/embodied/engine/` (the fork submodule) AND a
  separate `pip install lerobot` — the `[tool.uv.sources]` override is
  intentional and the path-install must win. Bypassing it (e.g. by manually
  installing PyPI lerobot in the container) leaves the import system in an
  ambiguous state.
- **Don't** modify the fork's source in-place inside `roboclaw/embodied/engine/`
  expecting the changes to ship — the submodule is read-only from this repo's
  perspective. Patches go upstream to `Elvin-yk/lerobot-roboclaw` then come back
  via `git submodule update`.

---

## 7. Reference files

| File | What it anchors |
|------|-----------------|
| [pyproject.toml line 51](../pyproject.toml) | The `lerobot[feetech,dynamixel]` dependency name (matches upstream PyPI package name). |
| [pyproject.toml lines 89-90](../pyproject.toml) | The `[tool.uv.sources] lerobot = { path = "roboclaw/embodied/engine" }` override that vendors the fork in place of PyPI. |
| [.gitmodules](../.gitmodules) | Records fork URL `Elvin-yk/lerobot-roboclaw.git` and branch `roboclaw`. |
| [roboclaw/embodied/embodiment/arm/registry.py](../roboclaw/embodied/embodiment/arm/registry.py) lines 46–55 | SO-101 model spec (probe protocol, motor IDs, baudrate, default servo, joint names). Identical between fork and upstream — defines what RoboClaw expects at the manifest layer. |
| [roboclaw/embodied/command/builder.py](../roboclaw/embodied/command/builder.py) lines 22–25 | The `_BIMANUAL["so101"] = ("bi_so_follower", "bi_so_leader")` mapping — the load-bearing fork-dependent string. |
| [roboclaw/embodied/command/headless_patch.py](../roboclaw/embodied/command/headless_patch.py) | Non-TTY calibration patches that the fork ships and upstream may not. |
| `phosphobot/phosphobot/hardware/so100.py` | phosphobot's first-party single-arm SO-101 class (covers SO-100 + SO-101). |
| `phosphobot/phosphobot/hardware/motors/feetech.py` | phosphobot's standalone Feetech bus (1066 LOC, frozen at early-LeRobot fork point). |
| [docs/INSTALLATION.md](./INSTALLATION.md) | Native-uv install path; uses the same fork via `uv sync`. |
| [docs/DOCKERINSTALLATION.md](./DOCKERINSTALLATION.md) | Stateless Docker install path; build inherits the fork via `--recurse-submodules`. |
| [docs/WSL2_DOCKER_DEPLOYMENT.md](./WSL2_DOCKER_DEPLOYMENT.md) | Operator runbook for the deployed stack; §13 records the commit chain. |
