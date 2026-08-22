# WSL2 + Docker Deployment (SO-101 Bimanual)

This guide documents the local deployment of RoboClaw on Windows 11 with WSL2,
isolated from any other WSL2 project that may already own the SO-101 hardware.
It corresponds to the committed changes in `Dockerfile`, `docker-compose.yml`,
and `scripts/attach_usb_roboclaw.ps1`.

Reach for this doc when:
- You have a Windows 11 host, SO-101 leader+follower arms on CH343 USB-serial,
  and at least one UVC camera.
- You want RoboClaw to run in a Docker container without disturbing an unrelated
  WSL2 project that already uses `usbipd-win` passthrough.
- You want the FastAPI dashboard served by the same container that runs the
  agent and the embodied subsystem — one port, one container, one volume.

It is **not** a replacement for `INSTALLATION.md` (native uv path) or
`DOCKERINSTALLATION.md` (minimal stateless Docker path). It is a third,
hardware-grade deployment mode.

---

## 1. Architecture at a glance

```
+-----------------------------------------+        +-----------------------+
|  Windows 11                             |        |  WSL2 distro "Ubuntu" |
|  -----------                            |        |  (existing, untouched)|
|  usbipd-win  <-- SO-101 x4 + UVC cam    |        |  used by unrelated    |
|       |                                 |        |  so101-rl-deploy      |
|       | (usbipd attach --wsl            |        +-----------------------+
|       |   Ubuntu-roboclaw --auto-attach)
|       v
|  +-----------------------------------+  |        +-----------------------+
|  |  WSL2 distro "Ubuntu-roboclaw"    |  |        |  Windows browser      |
|  |  (dedicated, created by operator) |  |        |  http://localhost:8765|
|  |                                   |  |        +-----------^-----------+
|  |  /dev/ttyACM0..3  /dev/video0     |  |                    |
|  |  udev -> /dev/serial/by-id/*      |  |        port 8765 + 1455 forwarded
|  |                                   |  |
|  |  Docker Engine (NOT Docker Desktop)
|  |    |                              |  |
|  |    v                              |  |
|  |  Container: roboclaw-web          |  |
|  |    - roboclaw web start :8765     |  |
|  |    - FastAPI + React SPA          |  |
|  |    - embodied subsystem           |  |
|  |    - openai-codex provider        |  |
|  |    volumes:                       |  |
|  |      /root/.roboclaw  <- ext4     |  |
|  |      /root/.cache/huggingface     |  |
|  +-----------------------------------+  |
+-----------------------------------------+
```

Key design choices and why they exist:

| Choice | Reason |
|--------|--------|
| Dedicated WSL2 distro (`Ubuntu-roboclaw`), not `docker-desktop` | Docker Desktop's internal LinuxKit VM lacks `vhci-hcd`/`usbip-core` kernel modules and udev, so `usbipd attach --wsl docker-desktop` fails and `/dev/serial/by-id/` stays empty. A full Ubuntu distro has both. |
| Docker **Engine** inside that distro, not Docker Desktop | Removes the Docker Desktop VM hop entirely. Containers run under the same kernel that `usbipd` attaches devices to, so `--device /dev/ttyACMx` works. |
| Multi-stage Dockerfile with **editable** Python install | `roboclaw/http/server.py:297` resolves `ui_dist = Path(__file__).parent.parent.parent / "ui" / "dist"`. Under a system install into `site-packages` this resolves to a non-existent path and the UI silently 404s. `uv pip install -e .` keeps the import root at `/app/roboclaw/…` so `/app/ui/dist/` is found. |
| `-p 127.0.0.1:1455:1455` | `roboclaw/providers/openai_codex_provider.py` runs an OAuth callback server on `localhost:1455` inside the container; the redirect must reach that port from the Windows browser. |
| Volume at `/home/hafnium/.roboclaw` on ext4 (inside the distro) | DrvFs (`/mnt/c/…`) is 5–20× slower for fsync-heavy workloads and mangles `0600` perms on the OAuth token. |
| Calibration via `docker compose exec -it`, not the browser | Calibration uses `termios` raw mode (`roboclaw/embodied/toolkit/tty.py`); the detached `web start` container has no PTY for the browser code path. CLI `exec -it` has a PTY. |
| Single container per volume | The cross-process flock in `roboclaw/embodied/embodiment/lock.py` is safe with a single writer; running two containers against the same volume is not supported. |

Full review rationale lives in `/home/hafnium/.claude/plans/let-s-deploy-roboclaw-locally-serene-finch.md`.

---

## 2. Prerequisites

- Windows 11 with WSL2 enabled (`wsl --install` if starting fresh).
- `usbipd-win` installed: `winget install usbipd`. Verify `usbipd --version`.
- An elevated PowerShell window available when binding USB devices.
- The SO-101 arms + camera physically connected and visible via
  `usbipd list` on Windows. You should see five shared devices: four
  CH343 serials (`idVendor=0x1a86`) and at least one UVC camera.

---

## 3. Create the dedicated WSL2 distro + install everything (automated)

Run in an **admin PowerShell** on Windows from a clone of this repo:

```powershell
cd <path-to-RoboClaw>\scripts
.\bootstrap_distro.ps1
```

What the script does (all idempotent):
1. Downloads the Ubuntu 24.04 WSL rootfs to `$env:USERPROFILE\wsl\ubuntu-roboclaw\rootfs.tar.gz` (cached).
2. `wsl --import`s `Ubuntu-roboclaw` if the distro does not already exist.
3. Tar-packs `provision_distro.sh`, `setup-udev.sh`, `install-interop-guard.sh`, and `deploy.sh` into the distro at `/root/bootstrap/` — the canonical in-distro location.
4. Runs `provision_distro.sh` as root inside the distro to create the `hafnium` user with passwordless sudo, write `/etc/wsl.conf`, install Docker Engine via `get.docker.com`, register udev rules for the CH343 USB-serial chips, and enable the WSLInterop guard systemd timer (see §13 Troubleshooting for what the guard does).
5. Terminates the distro so `[user]` default + `[boot] systemd=true` take effect on next launch.

`Ubuntu-roboclaw` does **not** become the default WSL distro. Always invoke it
explicitly with `wsl -d Ubuntu-roboclaw`.

---

## 4. End-to-end bringup (cloning, build, onboard)

After the bootstrap completes, one more admin-PS line does the clone, Docker
build, and onboard:

```powershell
wsl -d Ubuntu-roboclaw -u root -- bash /root/bootstrap/deploy.sh
```

(Equivalently: `bash $env:USERPROFILE\wsl\ubuntu-roboclaw\bootstrap\deploy.sh` — `bootstrap_distro.ps1` stages a copy into `$WslRoot\bootstrap\` so you can invoke either path.)

What `deploy.sh` does (all idempotent):
1. **Provisioning**: skipped if `/etc/roboclaw/provisioned.v<N>` marker matches the current schema version; otherwise re-runs `provision_distro.sh`. Bump the version (`PROVISION_SCHEMA` in `deploy.sh`) when you add new provisioner steps to force re-provisioning of existing distros.
2. **Repo sync**: clones `https://github.com/hafnium49/RoboClaw.git` into `/home/hafnium/RoboClaw/` with `--recurse-submodules`, or pulls if already present.
3. **Image build**: `docker compose build roboclaw-web` (multi-stage; ~10 min first run, ~30s on re-runs with cache hits).
4. **Onboard**: `docker run --rm -v ~/.roboclaw:/root/.roboclaw roboclaw:local onboard` scaffolds `~/.roboclaw/`. Bypasses `docker compose run` deliberately — the compose service pins `devices: [/dev/ttyACM*..]`, which on first bringup (before USB attach) would fail with "error gathering device information … no such file or directory". Plain `docker run` doesn't inherit that constraint. See commit `53a5eb8` for context.

Expected final image size: ~2.5 GB (ffmpeg + libav* runtime libs contribute ~150 MB; CPU-only torch wheel is ~180 MB vs 900 MB for CUDA).

Interactive steps that deploy.sh does NOT do (you run manually after). Two
paths depending on whether USB passthrough is already live:

**Path A (recommended) — attach USB first, then use compose.** `docker compose
run` inherits the `roboclaw-web` service's `devices:` list, so `/dev/ttyACM*`
must exist in the distro first:

```powershell
# 1. Admin PowerShell on Windows:
.\scripts\attach_usb_roboclaw.ps1
```
```bash
# 2. Inside Ubuntu-roboclaw:
wsl -d Ubuntu-roboclaw
cd ~/RoboClaw
docker compose run --rm roboclaw-web provider login openai-codex   # OAuth (browser opens)
nano ~/.roboclaw/config.json                                       # set agents.defaults.model
docker compose run --rm roboclaw-web agent -m "hello"              # smoke test
```

**Path B — OAuth / smoke test before USB is available.** Bypass compose with
plain `docker run` and pass the port/volume flags the compose service would
have provided. Same image, no `devices:` constraint:

```bash
# Provider login — needs port 1455 for the OAuth redirect from the Windows browser:
docker run --rm -it -p 127.0.0.1:1455:1455 \
    -v ~/.roboclaw:/root/.roboclaw \
    roboclaw:local provider login openai-codex

# Set agents.defaults.model (same as Path A):
nano ~/.roboclaw/config.json

# Smoke test — no port needed, no hardware needed:
docker run --rm -v ~/.roboclaw:/root/.roboclaw roboclaw:local agent -m "hello"
```

Either path populates `/home/hafnium/.roboclaw/` (bind-mounted into the
container at `/root/.roboclaw`) with `config.json` and
`workspace/{AGENTS,SOUL,TOOLS,USER,HEARTBEAT}.md` plus `memory/MEMORY.md`.

---

## 5. Provider login (openai-codex OAuth) — mechanics

The login line in §4 works because port `1455` is exposed by `docker-compose.yml`:
the provider opens an `HTTPServer` on `localhost:1455` **inside the container**,
Docker forwards that to Windows `127.0.0.1:1455`, your Windows browser hits the
redirect, and the flow completes. The token lands in `~/.roboclaw/` with proper
`0600` perms on the ext4 volume.

Accepted model identifiers live in `roboclaw/providers/openai_codex_provider.py`
— set the one you want in `agents.defaults.model` of `~/.roboclaw/config.json`.

---

## 6. Route USB into `Ubuntu-roboclaw`

Do **not** modify any existing `attach_usb_wsl.ps1` — that one routes the same
BUSIDs to your other project's distro. Instead, use the committed script:

```powershell
# From an elevated PowerShell on Windows:
cd <repo>\scripts
.\attach_usb_roboclaw.ps1
```

What the script does:
1. Verifies it is running elevated.
2. Confirms `Ubuntu-roboclaw` exists.
3. Detaches each BUSID first (so we never collide with a prior attachment to
   another distro).
4. `usbipd bind --busid <b> --force` — makes the bind survive Windows reboots.
5. Spawns one hidden `usbipd attach --wsl Ubuntu-roboclaw --busid <b>
   --auto-attach` per BUSID. These processes stay alive in the background so
   devices reattach automatically on replug.
6. Verifies inside `Ubuntu-roboclaw` by listing `/dev/ttyACM*`,
   `/dev/serial/by-id/`, and `/dev/video*`, then asserts the **expected device
   counts**: `arms = 4` AND `cameras = 3` (1 scene + 2 wrist). Camera count is
   derived from `/dev/v4l/by-path/*-video-index0` (one entry per physical UVC
   capture endpoint), with a `udevadm info → ID_PATH` grouping fallback if
   `by-path/` is empty. Exits non-zero on mismatch so re-runs in CI / scripts
   can detect a partial attach.

Flags:

| Flag | Purpose |
|------|---------|
| `-Distro <name>` | Override target distro (default `Ubuntu-roboclaw`). |
| `-BusIds <array>` | Override BUSIDs (default `4-1, 4-2, 4-3, 4-4, 5-1, 5-3, 5-4` = 4 arms + 1 scene cam + 2 wrist cams). Re-verify yours with `usbipd list`. |
| `-Detach` | Reverse direction — detach all BUSIDs from every distro. Use before handing the hardware back to another project. |

To make auto-attach persist across Windows reboots, register the script as a
Task Scheduler entry triggered **At log on**, running hidden.

---

## 7. Start the runtime

```bash
wsl -d Ubuntu-roboclaw
cd ~/RoboClaw
docker compose up -d roboclaw-web
docker compose ps
docker compose logs -f roboclaw-web
```

Open `http://localhost:8765/` in a Windows browser. The single port serves
both the React dashboard (static files from `/app/ui/dist/`) and the
FastAPI + WebSocket API — same-origin, no CORS.

Health:

```bash
curl -fsS http://localhost:8765/api/health
docker compose exec roboclaw-web roboclaw status
```

---

## 8. Embodied onboarding (bimanual SO-101)

### 8.1 Calibrate arms (CLI only)

```bash
docker compose exec -it roboclaw-web roboclaw agent
# Ask the agent: "List detected arms and cameras, then calibrate each SO-101 arm."
```

Calibration must be driven from `docker compose exec -it` because the
termios raw-mode path in `roboclaw/embodied/toolkit/tty.py` needs a PTY. The
browser-based calibration panel will hang in this deployment — that is a known
constraint of the detached `web start` container.

### 8.2 Bind arms and cameras

After calibration, use the dashboard's Setup panel (or the agent with embodied
tools) to build the manifest:

- 2× `so101_leader` — `role=leader`, `side ∈ {left, right}`
- 2× `so101_follower` — `role=follower`, `side ∈ {left, right}`
- 1–3× cameras, each labelled with `side`

Bind arms by their stable `/dev/serial/by-id/usb-...` paths (visible in the
script's verification output). `ttyACM` indices reshuffle on replug; `by-id`
does not.

### 8.3 Teleop / record / replay / infer (dashboard)

Once bound, those four workflows run fine from the browser dashboard — they do
not need a PTY. Bimanual mode activates automatically:
`roboclaw/embodied/command/builder.py:22-25` maps `"so101"` →
`("bi_so_follower", "bi_so_leader")` when `arms=""`.

---

## 9. Operating procedures

### 9.1 Switching the robot between projects

Only one WSL distro can own a given USB device at a time, so switching projects
is explicit:

```powershell
# Give the robot to RoboClaw:
.\scripts\attach_usb_roboclaw.ps1

# Give it back to the other project (example):
.\scripts\attach_usb_roboclaw.ps1 -Detach
.\<path-to-other-project>\attach_usb_wsl.ps1
```

Stop the RoboClaw container before detaching, to avoid mid-op failures:

```bash
docker compose stop roboclaw-web
```

### 9.2 Persistent auto-attach across reboots

Register `attach_usb_roboclaw.ps1` as a Windows Task Scheduler task:
- Trigger: **At log on of any user**.
- Action: `powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File "<full path>\attach_usb_roboclaw.ps1"`.
- Run with highest privileges.

### 9.3 Data persistence

- `/home/hafnium/.roboclaw/` — config, workspace, memory, session state.
- Named volume `hf_cache` — LeRobot dataset cache at
  `~/.cache/huggingface/lerobot` (see `roboclaw/embodied/command/builder.py:30`).
  Recorded episodes live here; do not delete the volume unless you intend to
  lose them.

Back up with:

```bash
tar czf /tmp/roboclaw-backup-$(date +%F).tgz /home/hafnium/.roboclaw
docker run --rm -v hf_cache:/data -v /tmp:/out alpine \
  tar czf /out/hf_cache-$(date +%F).tgz -C /data .
```

### 9.4 Logs

JSON logs rotate at 10 MB × 3 files per service:

```bash
docker compose logs -f roboclaw-web
docker compose logs --tail 500 roboclaw-web
```

### 9.5 Updating to a newer RoboClaw

```bash
cd ~/RoboClaw
git pull --rebase upstream main
docker compose build roboclaw-web
docker compose up -d roboclaw-web
```

A full image rebuild is required for UI changes because `ui/dist/` is baked in.
There is no in-container hot reload — the UI is intentionally static in this
deployment mode.

---

## 10. Troubleshooting

| Symptom | Likely cause | Fix |
|---------|--------------|-----|
| Dashboard at `http://localhost:8765/` returns HTTP 404 for assets | Non-editable install; `ui/dist/` path mismatch | Rebuild image; confirm Dockerfile uses `uv pip install --system --no-cache -e .`. |
| `roboclaw provider login openai-codex` hangs; browser can't open callback | Port 1455 not published | Confirm `docker-compose.yml` maps `127.0.0.1:1455:1455` on `roboclaw-web`. |
| `/dev/ttyACM*` missing inside the distro | usbipd still attached elsewhere; or CH343 replugged after Windows reboot | Re-run `attach_usb_roboclaw.ps1`. Confirm BUSIDs match `usbipd list`. |
| `/dev/serial/by-id/` empty | udev not running in distro | Happens in minimal distros; `Ubuntu-roboclaw` from the 24.04 cloud rootfs has udev. Verify with `systemctl status systemd-udevd`. |
| Teleop at 30 Hz drops frames intermittently | usbipd-over-TCP jitter through the Windows scheduler | Drop to 20 Hz. Close heavy Windows workloads. If persistent, consider moving the robot to a native Linux host. |
| Camera fails to enumerate at >720p | UVC isochronous endpoints over usbipd | Force MJPEG at 720p30; the scanner already sets `CAP_PROP_FOURCC=MJPG` in `roboclaw/embodied/embodiment/hardware/scan.py:503`. Using more than one UVC camera through usbipd is unreliable. |
| Browser-driven calibration hangs | No PTY in the detached container | Calibrate via `docker compose exec -it roboclaw-web roboclaw agent`. |
| `docker compose up` complains about cgroup rule 188/189 | Older kernel without those majors | Harmless in practice; remove the two lines if your runtime refuses them. The plan keeps them as defensive entries for CH343 fallback modes. |
| `wsl.exe`/`pwsh.exe` from the Ubuntu shell fails with `Exec format error` | Known WSL2 bug: cross-distro `wsl -d Ubuntu-roboclaw ...` exit cleanup unregisters the kernel-shared `WSLInterop` binfmt entry | `install-interop-guard.sh` is enabled by §3 inside `Ubuntu-roboclaw` and restores the entry within ~30s. For the operator's `Ubuntu` distro, run the guard installer once: `wsl -d Ubuntu -u root -- bash /mnt/c/Users/<you>/wsl/ubuntu-roboclaw/bootstrap/install-interop-guard.sh`. Check with `systemctl is-active wsl-interop-guard.timer`. |
| `uv pip install` dies on `evdev==1.9.3` with missing `linux/input.h` | Runtime base image lacks kernel headers | Committed fix in `f4a7591`: stage-2 installs `linux-libc-dev`. If you see this, you're on a stale build — rebuild. |
| `roboclaw agent` ImportErrors on `torchcodec` / `av` | Runtime missing ffmpeg shared libs | Committed fix in `f4a7591`: stage-2 installs `ffmpeg libavcodec59 libavformat59 libavutil57 libswresample4 libswscale6`. If you see this, rebuild. |
| `provider login openai-codex` prints `✓ Authenticated with OpenAI Codex` but the next `agent -m "hello"` replies `OAuth credentials not found` | `docker compose run --rm …` wrote the token to the ephemeral writable overlay that Docker deletes on `--rm`. `oauth_cli_kit` stores tokens via `platformdirs.user_data_dir("openai-codex")` which resolves to `/root/.local/share/openai-codex/auth/<token>.json`. That path was NOT originally mounted. | Committed fix in `0adc679`: `docker-compose.yml` now binds `/home/hafnium/.roboclaw-local-share:/root/.local/share`, persisting the token to host ext4. Prefer `docker compose exec roboclaw-web roboclaw provider login openai-codex` over `docker compose run --rm …` so the token is written into the already-mounted persistent container. If you must use `docker compose run`, drop the `--rm` and extract via `docker cp <cid>:/root/.local/share /home/hafnium/.roboclaw-local-share/` before `docker rm <cid>`. |
| Arm manifest binds to raw `/dev/ttyACMx` instead of `/dev/serial/by-id/usb-1a86_*` | Docker's `devices:` passes raw device inodes but not udev's `/dev/serial/` symlink tree; scan.py falls back to `ttyACMx` indices that shuffle on replug | Committed fix in `0adc679`: `docker-compose.yml` now binds `/dev/serial:/dev/serial:ro`, exposing udev's stable `usb-1a86_USB_Single_Serial_<SN>` symlinks inside the container. Re-record the manifest via the dashboard or `roboclaw agent` once — it will pick up by-id paths automatically. |
| `docker compose run --rm roboclaw-web provider login openai-codex` succeeds but container and its filesystem are gone on next `docker ps -a` | `--rm` is atomic with process exit: Docker removes the container and its upper overlay simultaneously. Any file written to a non-mounted path during the run is unrecoverable. | Never use `--rm` for a command that writes state you care about. Use `docker compose exec` against the running service (all mounts active), or `docker compose run` (no `--rm`) and clean up with explicit `docker rm` after extracting anything needed via `docker cp` / `docker diff`. |

---

## 11. Explicit non-goals of this deployment

- Running two containers against the same `~/.roboclaw` volume (flock
  semantics over a bind mount are not verified for multi-writer).
- Enabling Docker Desktop's WSL integration with the primary `Ubuntu` distro
  (would re-couple the two projects).
- Training runs on the same host — teleop + record are the first-pass goal;
  training can move to a dedicated GPU host consuming datasets from
  `hf_cache`.
- Exposing `8765` beyond `127.0.0.1`. If you need LAN access, put a reverse
  proxy with authentication in front of the container.

---

## 12. Referenced repo files

| File | What it anchors |
|------|-----------------|
| `Dockerfile` | Multi-stage build, editable install, exposed ports, CPU-torch index, evdev + ffmpeg runtime deps. |
| `docker-compose.yml` | `roboclaw-web` service, devices, cgroup rules, volumes, healthcheck. |
| `scripts/bootstrap_distro.ps1` | Windows orchestrator: rootfs download, `wsl --import`, tar-packs scripts into `/root/bootstrap/`. |
| `scripts/provision_distro.sh` | In-distro provisioner: user, wsl.conf, Docker Engine, udev rules, interop guard. |
| `scripts/deploy.sh` | Idempotent end-to-end bringup (provision + clone + build + onboard) with marker-file-versioned skip. |
| `scripts/install-interop-guard.sh` | Systemd unit + timer that re-registers `WSLInterop` binfmt_misc when wiped. |
| `scripts/attach_usb_roboclaw.ps1` | Windows-side USB routing to `Ubuntu-roboclaw`. |
| `scripts/setup-udev.sh` | CH343 (`idVendor=0x1a86`) + video4linux udev rules. |
| `roboclaw/http/server.py:297` | UI path resolution; relies on editable install. |
| `roboclaw/providers/openai_codex_provider.py` | OAuth callback on port 1455. |
| `roboclaw/embodied/toolkit/tty.py` | Termios raw-mode handoff (PTY-only). |
| `roboclaw/embodied/embodiment/lock.py` | Cross-process embodiment lock. |
| `roboclaw/embodied/embodiment/hardware/scan.py` | `/dev/serial/by-id/` discovery, UVC fourcc settings. |
| `roboclaw/embodied/command/builder.py:22-25` | Bimanual SO-101 preset `_BIMANUAL["so101"]`. |
| `docs/INSTALLATION.md` | Native uv path (alternative to this doc). |
| `docs/DOCKERINSTALLATION.md` | Minimal stateless Docker path (alternative to this doc). |
| `docs/SO101_BIMANUAL_DRIVER.md` | Driver-source decision (RoboClaw's LeRobot fork vs upstream PyPI vs phosphobot) for bimanual SO-101, with verification probe. |
| `docs/FORK_PROGRESS_REPORT.md` | Standalone fork-vs-upstream progress report (commit timeline, validation status, churn signals). |

---

## 13. Session commit chain (reference)

This deployment was brought up incrementally. Each commit below corresponds to a specific failure surfaced by the build log; preserved here so a future operator hitting a regression can `git bisect` around the exact fix rather than guessing.

| Commit | Area | Purpose |
|--------|------|---------|
| `81664a8` | interop guard | New `scripts/install-interop-guard.sh`: systemd oneshot + 30s timer re-registers `WSLInterop` when it's wiped by cross-distro `/init` cleanup. |
| `49d8e6e` | provisioner | `provision_distro.sh` calls the guard installer; `bootstrap_distro.ps1` tar-packs the new script into `/root/bootstrap/`. |
| `53c555c` | Dockerfile | ui-builder copies `roboclaw/i18n/{common,setup}.json` before `npm run build`; stage-2 guards missing LeRobot submodule; CPU-only torch index (saves ~4 GB download); `PYTHONDONTWRITEBYTECODE=1`. |
| `f4a7591` | build deps | Add `linux-libc-dev` + ffmpeg runtime libs (`libavcodec59`, `libavformat59`, `libavutil57`, `libswresample4`, `libswscale6`) to stage-2 apt; drop the `[pi]` extra from `lerobot[…]` in `pyproject.toml` (GPIO-only, useless on WSL2 x86_64). |
| `d848a1a` | deploy.sh | Promoted from staged-only `/mnt/c/.../deploy.sh` into `scripts/deploy.sh`; reads exclusively from `/root/bootstrap/` (the canonical in-distro location written by `bootstrap_distro.ps1`); marker-file-versioned provisioning skip at `/etc/roboclaw/provisioned.v<N>`. |
| `2fe2a28` | build toolchain | Add `build-essential` to stage-2: the uv base image didn't ship `gcc`, so `evdev`'s C-extension source build couldn't compile despite `linux-libc-dev` being present. |
| `53a5eb8` | deploy.sh | Bypass `docker compose run` for the `onboard` step; use plain `docker run` against `roboclaw:local` with only the workspace volume. Fixes first-bringup chicken-and-egg: compose's `devices: [/dev/ttyACM0..]` would fail before USB passthrough is active. |
| `d7f732e` | deploy.sh | `chown -R ${ROBOCLAW_USER}` on `~/.roboclaw` after the root-uid container run; lets the operator edit `config.json` without sudo. |
| `a9cc9fd` | docs | Three-row decision table at the top of each install guide; `docs/INSTALLATION.md` renamed 5.1/5.2 → 6.1/6.2; `docs/DOCKERINSTALLATION.md` documents submodule requirement + explicit "doesn't cover" list; WSL2 guide §12 expanded + new §13 session commit chain. |
| `6f4320f` | docs | Fix stale §4 Step 4 description (onboard uses `docker run`, not `docker compose run`); add Path A (USB first → compose) / Path B (no-USB → `docker run` direct) for post-deploy interactive commands. |
| `0adc679` | compose | Add two volumes to `roboclaw-web`: `/home/hafnium/.roboclaw-local-share:/root/.local/share` for oauth_cli_kit token persistence across `--rm` containers, and `/dev/serial:/dev/serial:ro` so arm manifests can bind to udev's stable `/dev/serial/by-id/usb-1a86_*` symlinks instead of shuffling ttyACMx indices. |
| `49c6788` | scripts | `scripts/attach_usb_roboclaw.ps1` default BUSID list grows from 5 → 7: add `4-1` and `4-2` for the two new wrist cameras on the followers; the original 4 arms + 1 scene cam remain unchanged. |
| `a3b8d99` | scripts | `scripts/attach_usb_roboclaw.ps1` verification step rewritten to assert `arms=4` AND `cameras=3` and exit non-zero on mismatch; camera count derives from `/dev/v4l/by-path/*-video-index0` with udev `ID_PATH` fallback. Docs §6/§13 + `FORK_PROGRESS_REPORT.md` updated to match. |

End-to-end time on a warm cache: ~14 seconds (deploy.sh run 6 on day 2). Cold build: ~10 minutes dominated by apt/npm/uv downloads.

---

## 14. Operational lessons learned (runtime session notes)

Findings from the first end-to-end bringup with real hardware, preserved so a future operator doesn't re-learn them:

**USB passthrough mechanics (non-obvious even though the pieces are standard):**
- `usbipd bind --force` is the one-time admin step and persists across Windows reboots. `usbipd attach --wsl <distro>` does **not** require admin, only that the target distro is running.
- If `usbipd attach` errors with `The selected WSL distribution is not running`, wake the target by starting any process in it (e.g. `wsl -d Ubuntu-roboclaw -- sh -c 'sleep 600' &` from Windows) before the attach loop. Keep that sleep alive or attach multiple BUSIDs back-to-back before the distro idles out.
- After `usbipd attach`, the distro's udev populates both `/dev/ttyACMx` (with CH343 vendor 0x1a86 rule giving `crw-rw-rw-`) and `/dev/serial/by-id/usb-1a86_USB_Single_Serial_<SN>`. The by-id path is stable across replugs; the `ttyACMx` index is not.

**Docker + USB:**
- `--device /dev/ttyACM0` passes the raw device inode but not udev's symlink tree. Without the compose binding `/dev/serial:/dev/serial:ro`, the container sees the raw nodes but cannot resolve by-id. Commit `0adc679` added that bind.
- Compose's `devices:` list fails at `docker run`/`docker compose run` time if the node doesn't exist at the host. This is the chicken-and-egg on first bringup that commit `53a5eb8` worked around for `onboard` by using plain `docker run` (no service definition).
- The device cgroup rules in compose (majors 81/166/188/189) are what allow newly-plugged devices of those types to appear inside a running container without a restart. They're defensive — the container already has its `--device` inodes at start.

**OAuth + ephemeral containers:**
- `oauth_cli_kit` stores tokens via `platformdirs.user_data_dir("openai-codex")` → `/root/.local/share/openai-codex/auth/`. This is NOT under the existing `/root/.roboclaw` bind mount by default. Commit `0adc679` added the separate `.local/share` bind to fix this.
- `docker compose run --rm` with a login flow is a trap: the login succeeds, prints the token expiry, then the container deletes itself on exit, and the token is gone. `roboclaw status` will still show `OpenAI Codex: ✓ (OAuth)` because that status check tests the *capability* (provider config present), not the actual token state. The next real model call fails with `OAuth credentials not found`. The clean pattern is:
  - `docker compose exec <service> roboclaw provider login openai-codex` against the persistent running service (all mounts active) — preferred.
  - `docker compose run` (no `--rm`) + `docker cp <cid>:/root /dest/` + `docker rm <cid>` — fallback when the service isn't running.

**Interop guard (the session's load-bearing piece of infra):**
- Verified across a full PC restart: `systemctl enable --now wsl-interop-guard.timer` at install time writes `/etc/systemd/system/timers.target.wants/wsl-interop-guard.timer`. On every WSL2 boot, systemd replays the symlink and the timer auto-starts. No re-install needed.
- Observed this session: one cross-distro `wsl.exe` call wiped `WSLInterop` mid-run; the guard restored it in **3 seconds** (not the 30s worst case). `journalctl -u wsl-interop-guard` empty until the wipe, one line after, back to silent.

**Build caching:**
- Warm-cache deploy.sh end-to-end: ~14 seconds (provisioning SKIPs via marker, repo pulls, cached Docker layers, onboard is a single `docker run`).
- Cold build: ~10 min (apt + npm + uv downloads dominate). CPU-only torch cuts this from the original ~25 min.
- Rebuilding the image invalidates the stage-3 `uv pip install` layer whenever `roboclaw/**` or `pyproject.toml` changes, triggering a full Python-dep re-resolve (not re-download if the wheels are still in uv's cache dir, but recomputation is ~30-90s).

**`WSLInterop` drop effects I hit during driving this remotely:**
- Even with the guard, a just-fired cross-distro call can leave a ~0-3s window where `wsl.exe` returns `Exec format error`. Retrying within 10s almost always succeeds. Commands I issued during that window failed once and worked the second time.
