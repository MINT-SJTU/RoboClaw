# Fork Progress Report — `hafnium49/RoboClaw`

> **Generated**: 2026-05-08 against commit `fabdd50` (HEAD).
> **Fork base**: `MINT-SJTU/RoboClaw` (upstream remote `upstream/main`); first-fork-commit pre-divergence `0968774`.
> **Scope**: `git rev-list --count upstream/main..HEAD = 26` commits ahead.

This report stands alone — no conversation context required, no other doc required. It records what this fork has changed since branching from upstream, why each change exists, what is verified vs deferred, and where each delta lives. Future operators (and future Claude sessions) should be able to read it linearly and pick up the deployment without re-deriving anything from scratch.

---

## 1. Executive summary

This fork takes upstream `MINT-SJTU/RoboClaw` (a channel-agnostic embodied-AI agent that drives LeRobot-vendored robot arms through an LLM tool loop) and wires it onto a specific physical setup: a Windows 11 + WSL2 host with **2× SO-101 leader arms + 2× SO-101 follower arms** on CH343 USB-serial plus a UVC camera, run from inside a dedicated `Ubuntu-roboclaw` WSL2 distro using **Docker Engine** (not Docker Desktop).

The fork's center of gravity is operational: zero changes to the Python source under `roboclaw/`. Every delta lives in one of five areas — `Dockerfile`, `docker-compose.yml`, `scripts/` (Windows + Linux bootstrap), `docs/` (operator guides), and one tweak to `pyproject.toml`. The reason to fork at all rather than upstream-PR was that several of the changes are host-specific (a non-root user named `hafnium`, hardcoded `/home/hafnium/.roboclaw` bind mounts, `Ubuntu-roboclaw` distro name) and others were emergency forensics during a single bring-up where a clean PR cycle would have stalled the deployment.

**Current state** (2026-05-08): `roboclaw-web` container is `Up ... (healthy)`; OAuth token is persisted at `/home/hafnium/.roboclaw-local-share/oauth-cli-kit/auth/codex.json`; 4 CH343 arms + 1 UVC camera visible inside the container; agent smoke test (`gpt-5.2`) returns. Embodied calibration is in progress — manifest was probed (4 arms identified) but binding/calibration is the next operator-interactive step. Bimanual teleop @ 30 Hz is not yet attempted.

### 1.1 30-second context glossary

A reader arriving cold needs six terms to follow the rest of this report.

| Term | One-line meaning |
|------|------------------|
| **WSL2** | Microsoft's Windows Subsystem for Linux 2 — runs a real Linux kernel inside a lightweight Hyper-V VM, hosts named distros side-by-side. |
| **usbipd-win** | Windows-side service that re-exports a USB device over TCP/IP into one specific WSL2 distro, exposing it as `/dev/ttyACM*` etc. inside that distro. |
| **WSLInterop** | Kernel `binfmt_misc` entry that lets Linux execute Windows `.exe` files (`wsl.exe`, `usbipd.exe`) from a WSL distro shell; can be wiped by cross-distro calls. |
| **openai-codex OAuth** | RoboClaw's ChatGPT-account login flow (`provider login openai-codex`) that mints a refresh token via a browser redirect to `localhost:1455`; persisted via `oauth_cli_kit`. |
| **`/dev/serial/by-id/`** | udev-populated stable symlinks like `usb-1a86_USB_Single_Serial_5A68009448` — the only reliable way to bind a specific physical arm to a manifest entry across reboots. |
| **bimanual SO-101** | Two leaders + two followers driven through LeRobot's `bi_so_follower` / `bi_so_leader` config aliases; preset triggered automatically via `roboclaw/embodied/command/builder.py:23` `_BIMANUAL["so101"]`. |

---

## 2. Fork delta vs upstream

<!-- BEGIN AUTOGEN: section regenerable via scripts queries below -->

| Metric | Value |
|--------|-------|
| Commits ahead of `upstream/main` | **26** |
| Files touched by fork commits | **13** |
| LOC added | **1487** |
| LOC removed | **46** |
| First fork commit | `0968774` (2026-04-22 13:26) |
| Latest fork commit | `fabdd50` (2026-05-07 16:38) |

Per-area churn (vs first-fork-commit `0968774~1`):

| Area | Files | LOC + | LOC − |
|------|-------|------:|------:|
| `Dockerfile` + `docker-compose.yml` | 2 | 115 | 34 |
| `scripts/*.{ps1,sh}` | 5 | 565 | 0 |
| `docs/*.md` | 4 | 693 | 9 |
| `pyproject.toml` | 1 | 2 | 2 |
| `AGENTS.md` (= `CLAUDE.md` symlink) | 1 | 113 | 2 |

No edits under `roboclaw/`, `tests/`, `ui/`, `bridge/`, or `roboclaw/embodied/engine/` (the LeRobot fork submodule remains pinned at upstream's reference).

<!-- END AUTOGEN -->

---

## 3. Validation status by component

Each row is reader-runnable: the probe column is a command an operator can paste in to verify the claim independently. Probes were last run against HEAD `fabdd50` on 2026-05-08.

| Component | Status | Probe (paste in `wsl -d Ubuntu-roboclaw`) | Expected |
|-----------|:-----:|--------------------------------------------|----------|
| Dedicated WSL2 distro | ✅ | `wsl --list --verbose` (PowerShell) | `Ubuntu-roboclaw  Running  2` |
| Docker Engine in distro | ✅ | `docker version --format '{{.Server.Os}}/{{.Server.Arch}}'` | `linux/amd64` |
| Container running | ✅ | `docker compose ps \| grep roboclaw-web` | `Up ... (healthy)` |
| `/api/health` | ✅ | `curl -fsS http://localhost:8765/api/health` | `{"status":"ok","channel":"web"}` |
| USB passthrough — arms | ✅ | `ls /dev/serial/by-id/usb-1a86_*` | 4 entries (`5A68009448`, `5A68009540`, `5A68011258`, `5A68011529`) |
| USB passthrough — camera | ✅ | `ls /dev/video*` | `/dev/video0` (and `/dev/video1` metadata node) |
| WSLInterop guard | ✅ | `systemctl is-active wsl-interop-guard.timer` (in Ubuntu) | `active` |
| OAuth token persisted | ✅ | `ls /home/hafnium/.roboclaw-local-share/oauth-cli-kit/auth/codex.json` | exists, 0600 |
| Agent smoke test | ✅ | `docker compose exec roboclaw-web roboclaw agent -m hi` | text response (model `gpt-5.2`) |
| Embodied identification | ⏳ | `cat /home/hafnium/.roboclaw/workspace/embodied/manifest.json \| jq '.arms \| length'` | currently `0`; target `4` |
| Embodied calibration | ⏳ | (interactive — operator drives arms to endstops) | calibration JSON per arm |
| Bimanual teleop @ 30 Hz | ⏳ | (dashboard-driven from `http://localhost:8765/`) | leaders drive followers, no dropped frames |

✅ = verified at HEAD `fabdd50`. ⏳ = deferred / requires operator-in-the-loop.

---

## 4. Architecture

```
Windows 11 host
├─ usbipd-win service  ─────► (TCP USB/IP)
│                                │
├─ Ubuntu (existing distro)      │   ◄─ separate, unrelated so101-rl-deploy project
│  └─ wsl-interop-guard.timer    │
│                                │
└─ Ubuntu-roboclaw  (wsl --import, dedicated)
   ├─ /etc/wsl.conf [boot] systemd=true
   ├─ wsl-interop-guard.timer
   ├─ udev (CH343 + v4l rules)
   └─ Docker Engine (in-distro, NOT Docker Desktop)
      └─ container roboclaw-web
         ├─ devices: /dev/ttyACM0..3, /dev/video0
         ├─ device_cgroup_rules: c 81/166/188/189 rmw
         ├─ volumes:
         │   /home/hafnium/.roboclaw  →  /root/.roboclaw           (config + workspace)
         │   /home/hafnium/.roboclaw-local-share → /root/.local/share  (OAuth)
         │   /dev/serial → /dev/serial:ro                          (by-id symlinks)
         │   roboclaw_hf_cache → /root/.cache/huggingface          (named)
         ├─ ports: 127.0.0.1:8765 (dashboard), 127.0.0.1:1455 (OAuth callback)
         └─ healthcheck: GET /api/health
```

Why Docker Engine in the distro and not Docker Desktop: Docker Desktop's daemon runs in the LinuxKit-based `docker-desktop` distro, which lacks the `vhci-hcd` / `usbip-core` kernel modules and udev — so `usbipd attach --wsl docker-desktop` cannot expose `/dev/ttyACM*` to containers, and even if it could, no `/dev/serial/by-id/` symlinks would be populated. Full discussion in [WSL2_DOCKER_DEPLOYMENT.md](./WSL2_DOCKER_DEPLOYMENT.md).

Why a dedicated `Ubuntu-roboclaw` distro instead of reusing the existing `Ubuntu`: the host already runs an unrelated `so101-rl-deploy` project against `Ubuntu` via a separate `attach_usb_wsl.ps1`. RoboClaw gets its own USB router (`scripts/attach_usb_roboclaw.ps1` → `--wsl Ubuntu-roboclaw`) so the two projects can coexist without fighting over BUSIDs.

---

## 5. Commit timeline — by theme

The fork has 26 commits across 4 active days (2026-04-22 → 05-07). Themes overlap — the same commit can appear in multiple buckets. Themes already covered exhaustively by [WSL2_DOCKER_DEPLOYMENT.md §13](./WSL2_DOCKER_DEPLOYMENT.md) (commit chain) are summarized here in one line each; full hash lists are kept only for themes §13 does not cover.

### 5.1 Distro automation (NEW — not in §13)

The Windows-side and Linux-side scripts that turn first-time bring-up into ~3 admin commands.

- `0629c8b` — `attach_usb_roboclaw.ps1` first-landing (Windows USB router targeting `--wsl Ubuntu-roboclaw`).
- `e0a7f91` — refactor: cgroup rules in `docker-compose.yml` annotated; `attach_usb` adds `usbipd detach --all` pre-pass to clear stale sessions after a `usbipd` service restart.
- `576969c` — `bootstrap_distro.ps1` + `provision_distro.sh` — Windows orchestrator + in-distro provisioner (rootfs download, `wsl --import`, Docker Engine via `get.docker.com`, udev rules).
- `1bef013` — PSScriptAnalyzer cleanup: rename internal helpers to approved verbs (`Require-Admin` → `Assert-Admin`, etc.).
- `3bf3da5` — provisioner detects Docker Desktop's CLI shim on `$PATH` and refuses to mistake it for real Docker Engine.

### 5.2 Compose volume + device evolution (NEW — not in §13)

The compose file went through three iterations as bring-up uncovered missing bind mounts.

- `f345e5a` — initial `docker-compose.yml` rewrite alongside Dockerfile (services, healthcheck, log rotation).
- `e0a7f91` — `device_cgroup_rules` formalized: `c 81:* rmw` (video), `c 166:* rmw` (CDC-ACM), `c 188:* rmw` (USB-serial), `c 189:* rmw` (raw libusb — flagged trusted-host-only because it grants firmware-reprogram capability).
- `0adc679` — added `/home/hafnium/.roboclaw-local-share:/root/.local/share` (OAuth tokens survive `--rm`) and `/dev/serial:/dev/serial:ro` (`by-id/` symlinks visible inside container).

### 5.3 Documentation expansion (partial overlap with §13 commit chain)

Docs went from upstream's single Chinese `AGENTS.md` to a 4-file `docs/` set with a decision table and a session commit chain.

- `0968774` — first fork commit. Expanded `AGENTS.md` (= `CLAUDE.md` symlink) with commands + architecture overview in English.
- `4e7d20f` — `docs/WSL2_DOCKER_DEPLOYMENT.md` first-landing.
- `1bcdbf4` — automate WSL2 distro setup section in WSL2 doc.
- `27579ff` — troubleshooting rows for interop + evdev + torchcodec.
- `a9cc9fd` — refresh INSTALLATION.md + DOCKERINSTALLATION.md + WSL2_DOCKER_DEPLOYMENT.md with a 3-row decision table at the top of each.
- `6f4320f` — fix stale onboard description + add USB-first ordering.
- `38dedc2` — record session findings (OAuth persistence, USB by-id, operational lessons).
- `fabdd50` — enhance install guides with LeRobot fork rationale; add `docs/SO101_BIMANUAL_DRIVER.md` (driver-source decision: upstream LeRobot vs RoboClaw fork vs phosphobot).

### 5.4 Dockerfile fix-of-fix iteration (covered by §13)

Eight commits hit `Dockerfile`. Reading [§13 of WSL2_DOCKER_DEPLOYMENT.md](./WSL2_DOCKER_DEPLOYMENT.md) gives the full story; one-line summary per commit:

`f345e5a` initial multi-stage refactor → `88960f7` `npm install` (no lockfile) → `b7a8f24` install `git` in node:20-slim builder → `433d3bb` `COPY bridge/` before `uv pip install` (force-include guard) → `53c555c` i18n COPY + submodule guard + CPU torch + `PYTHONDONTWRITEBYTECODE` → `f4a7591` `linux-libc-dev` + ffmpeg shared libs; drop `[pi]` extra → `2fe2a28` add `build-essential` (gcc for evdev source build).

### 5.5 Interop guard (covered by §13)

`81664a8` — `scripts/install-interop-guard.sh` (systemd oneshot + 30s timer that re-registers `:WSLInterop:M::MZ::/init:FP` if wiped by cross-distro `wsl.exe` exit cleanup). `49d8e6e` — provisioner runs the guard installer; bootstrap tar-packs the script. The guard survived a full PC reboot cleanly (timer `enabled` persists via `timers.target.wants` symlink; zero journal fires post-reboot, indicating interop stayed up).

### 5.6 deploy.sh promotion (covered by §13)

`d848a1a` — promoted off-repo bring-up script into `scripts/deploy.sh`; `/root/bootstrap/` becomes single-source-of-truth inside the distro; marker-file-versioned skip (`/etc/roboclaw/provisioned.v2`). `53a5eb8` — `deploy.sh` bypasses `docker compose run` for `onboard` because compose's hard `devices:` requirement fails before USB is attached on first run; uses plain `docker run` instead. `d7f732e` — `deploy.sh` chowns `~/.roboclaw` after the container-as-root onboard so the host user can read its own config.

---

## 6. Churn signals

Patterns mined from `git log` that prose alone wouldn't surface.

### File-touch heatmap (top 8)

| Touches | File |
|--------:|------|
| 7 | `docs/WSL2_DOCKER_DEPLOYMENT.md` |
| 7 | `Dockerfile` |
| 4 | `scripts/provision_distro.sh` |
| 4 | `scripts/bootstrap_distro.ps1` |
| 3 | `scripts/deploy.sh` |
| 3 | `scripts/attach_usb_roboclaw.ps1` |
| 3 | `docker-compose.yml` |
| 2 | `docs/INSTALLATION.md` |

The two hottest files (`Dockerfile`, `WSL2_DOCKER_DEPLOYMENT.md`) reflect the deployment's reality: every Dockerfile change hit a build error not predicted by the original multi-stage design, and the deployment doc absorbed the lessons commit-by-commit.

### Cadence by date

| Date | Commits | Notes |
|------|--------:|-------|
| 2026-04-22 | 7 | Day 1 — initial fork commit, Dockerfile/compose rewrite, USB script, WSL2 doc, distro automation. |
| 2026-04-23 | 12 | Day 2 — heaviest day. 5 commits in the 17:00 hour alone (Dockerfile fix-of-fix iteration + interop guard + deploy.sh promotion). |
| 2026-04-24 | 6 | Day 3 — bring-up debugging (deploy.sh fixes, compose volumes, doc fix-ups). |
| 2026-05-07 | 1 | Day 4 — `fabdd50` after a 13-day gap (driver-source decision doc). |

The 2026-04-23 17:00 cluster is the canonical "5 commits per hour" red flag — 5 fixes pushed inside 60 minutes is what fix-of-fix iteration looks like when a multi-stage Docker build is failing in slightly-different ways each rebuild.

### Fix-of-fix patterns

Subjects matching `fix.*Dockerfile` / `fix.*deploy` (8 commits):

```
d7f732e fix(deploy.sh): chown ~/.roboclaw after container-as-root onboard
53a5eb8 fix(deploy.sh): bypass compose for onboard to avoid device dependency
2fe2a28 fix(Dockerfile): replace curl with build-essential for improved package installation
f4a7591 fix(Dockerfile,pyproject): build-dep linux headers + ffmpeg libs; drop [pi] extra
53c555c fix(Dockerfile): i18n copy, submodule guard, CPU torch, no bytecode
433d3bb fix(Dockerfile): copy bridge/ source before uv pip install
b7a8f24 fix(Dockerfile): install git in node:20-slim builder stages
88960f7 fix(Dockerfile): use npm install instead of npm ci for ui + bridge
```

Six of eight target `Dockerfile`. The lesson the fork didn't expect: `node:20-slim` and `python3.12-bookworm-slim` ship genuinely minimal images — every build-time native dep (`git`, kernel headers, ffmpeg shared libs, `build-essential`) had to be added back in by hand. None of these are "code bugs"; they are environment expectations the original multi-stage plan didn't validate ahead of time.

### Mean commit size

`1487+/46-` over 26 commits = **57 lines added per commit**, **2 lines removed per commit**. Heavily additive — consistent with a fork that's adding new operational scaffolding rather than refactoring existing source.

---

## 7. Reproducibility recipe

The "three-script path" — three scripts wrap most of bring-up, but four manual prerequisites are not wrapped by any script. List both honestly so a fresh operator doesn't expect three commands to produce a working stack.

### Scripts

| # | Script | Where run | What it does |
|---|--------|-----------|--------------|
| 1 | `scripts/bootstrap_distro.ps1` | Admin PowerShell (Windows) | Download Ubuntu 24.04 rootfs, `wsl --import Ubuntu-roboclaw`, ship + run provisioner inside the new distro. |
| 2 | `scripts/attach_usb_roboclaw.ps1` | Admin PowerShell (Windows, separate session) | `usbipd bind --force` (first time) + `usbipd attach --wsl Ubuntu-roboclaw` for the 7 BUSIDs (4 arms + 1 scene cam + 2 wrist cams). Default list lives at line 10; override via `-BusIds`. Post-attach verification asserts arms=4 AND cameras=3 and exits non-zero on mismatch. |
| 3 | `scripts/deploy.sh` | `wsl -d Ubuntu-roboclaw -u root` | Re-run provisioner (marker-skipped after first run), clone repo, build image, scaffold `~/.roboclaw`. |

### Hidden prerequisites (not wrapped by any script)

1. **Two separate elevated PowerShell sessions** — `bootstrap_distro.ps1` and `attach_usb_roboclaw.ps1` both require admin. The second one needs to stay open (or be re-run after PC reboot) because `usbipd attach` is not persistent across host restarts.
2. **Browser-OAuth interactive flow** — `roboclaw provider login openai-codex` opens `http://localhost:1455/auth/callback` in the Windows browser; the operator must sign in interactively. No script automates this.
3. **`ROBOCLAW_USER=hafnium` is hardcoded** — `bootstrap_distro.ps1` defaults `-User hafnium`. Override at invocation if your host user differs.
4. **Model ID gotcha** — after `onboard`, edit `~/.roboclaw/config.json` to set the model to `gpt-5.2` (no `-codex` suffix; ChatGPT-account OAuth rejects `gpt-5.2-codex`, `o3`, `o4-mini`, etc.). Not in any of the three scripts; manual step.

### Day-2 operations (after first bring-up)

A reboot loses two transient pieces of state — USB attachments and the running container. Everything else (interop guard, OAuth token, image cache, repo clone, distro itself) survives. Day-2 resume is therefore: re-run `attach_usb_roboclaw.ps1`, then `docker compose up -d roboclaw-web` inside the distro. See [roboclaw-deployment skill](../../.claude/skills/roboclaw-deployment/SKILL.md) (operator-side, not in repo) for the full state-detect runbook.

---

## 8. Open issues (currently unresolved)

Issues catalogued in [WSL2_DOCKER_DEPLOYMENT.md §10 Troubleshooting](./WSL2_DOCKER_DEPLOYMENT.md) are NOT duplicated here. This section lists only items that are open at HEAD `fabdd50` and not in the troubleshooting catalog.

- **Manifest still empty** — embodied identification ran successfully (4 arms detected by serial), but binding to manifest entries (`so101_leader/follower` × `left/right`) and per-arm calibration are deferred to operator. The interactive setup-identify subprocess shows a camera-side prompt buffering symptom that hasn't been fully diagnosed; current workaround is to skip setup-identify and use `BindArmTool` directly with serials known from the operator's records.
- **Model ID drift** — `gpt-5.2` is the current ChatGPT-account-allowed model. OpenAI rolls models off this allowlist on a few-month cadence; if `agent -m hi` starts returning a 4xx, check `roboclaw/providers/openai_codex_provider.py` for the current allowlist and re-edit `~/.roboclaw/config.json`.
- **Three-camera bandwidth ceiling under usbipd untested** — week-2 setup adds 2 wrist cams (DSJ-2062 on `4-1`, `4-2`) on top of the existing scene cam. All 3 are the same VID:PID `0c45:64ab` model. USB 2.0 isochronous ceiling is ~720p/30 MJPEG per camera; three simultaneous streams almost certainly saturate the bus through usbipd-over-TCP. Pre-validate per-camera with `v4l2-ctl --device=/dev/v4l/by-path/<path> --stream-mmap` before turning all three on simultaneously during teleop. Expect to drop scene cam to 480p YUYV if all-three-on-30Hz is required.
- **30 Hz teleop jitter unmeasured** — Embedded review flagged 5–15 ms jitter tails from `usbipd` TCP + Windows scheduler pressure; bimanual at 30 Hz is within budget on paper but unverified. The jitter probe in [WSL2_DOCKER_DEPLOYMENT.md §11](./WSL2_DOCKER_DEPLOYMENT.md) should be run before the first dataset recording session.
- **`scripts/gen_progress_stats.sh` not yet created** — the autogen markers in §2 are placeholders; the regenerator script is planned but not committed.

---

## 9. Commit hash index

26 commits, alphabetical by short hash.

<!-- BEGIN AUTOGEN: regenerate via `git log --oneline upstream/main..HEAD | sort` -->

| Hash | Date | Subject |
|------|------|---------|
| `0629c8b` | 2026-04-22 | feat: add attach_usb_roboclaw.ps1 script for USB device management in WSL2 |
| `0968774` | 2026-04-22 | docs: expand AGENTS.md with commands and architecture overview |
| `0adc679` | 2026-04-24 | fix(docker-compose.yml): add volume for local share and expose serial devices |
| `1bcdbf4` | 2026-04-23 | feat: automate WSL2 distro setup and deployment process in documentation |
| `1bef013` | 2026-04-22 | refactor: rename functions for consistency in attach_usb_roboclaw.ps1 and bootstrap_distro.ps1 |
| `27579ff` | 2026-04-23 | docs: add troubleshooting rows for interop + evdev + torchcodec fixes |
| `2fe2a28` | 2026-04-23 | fix(Dockerfile): replace curl with build-essential for improved package installation |
| `38dedc2` | 2026-04-24 | docs(WSL2_DOCKER_DEPLOYMENT): record session findings — OAuth persistence, USB by-id, operational lessons |
| `3bf3da5` | 2026-04-23 | feat: enhance Docker installation logic in provision_distro.sh to differentiate between Docker Engine and Docker Desktop shim |
| `433d3bb` | 2026-04-23 | fix(Dockerfile): copy bridge/ source before uv pip install |
| `49d8e6e` | 2026-04-23 | feat: add install-interop-guard.sh to provisioning scripts for WSLInterop management |
| `4e7d20f` | 2026-04-22 | docs: add WSL2 + Docker deployment guide for SO-101 hardware |
| `53a5eb8` | 2026-04-24 | fix(deploy.sh): bypass compose for onboard to avoid device dependency |
| `53c555c` | 2026-04-23 | fix(Dockerfile): i18n copy, submodule guard, CPU torch, no bytecode |
| `576969c` | 2026-04-22 | feat: add bootstrap_distro.ps1 and provision_distro.sh for WSL2 setup automation |
| `6f4320f` | 2026-04-24 | docs(WSL2_DOCKER_DEPLOYMENT): fix stale onboard description + add USB-first ordering |
| `81664a8` | 2026-04-23 | feat: add install-interop-guard.sh to manage WSLInterop binfmt_misc registration |
| `88960f7` | 2026-04-23 | fix(Dockerfile): use npm install instead of npm ci for ui + bridge |
| `a9cc9fd` | 2026-04-24 | docs: refresh the three install guides for multi-path clarity |
| `b7a8f24` | 2026-04-23 | fix(Dockerfile): install git in node:20-slim builder stages |
| `d7f732e` | 2026-04-24 | fix(deploy.sh): chown ~/.roboclaw after container-as-root onboard |
| `d848a1a` | 2026-04-23 | feat: add deploy.sh for non-interactive bringup and update bootstrap_distro.ps1 to stage it |
| `e0a7f91` | 2026-04-22 | refactor: enhance usb device management in attach_usb_roboclaw.ps1 and update device_cgroup_rules in docker-compose.yml |
| `f345e5a` | 2026-04-22 | refactor: update Dockerfile and docker-compose.yml for improved structure and health checks |
| `f4a7591` | 2026-04-23 | fix(Dockerfile,pyproject): build-dep linux headers + ffmpeg libs; drop [pi] extra |
| `fabdd50` | 2026-05-07 | docs: enhance installation guides with LeRobot fork rationale and add new source selection guide for SO-101 Bimanual Driver |

<!-- END AUTOGEN -->

---

## Snapshots

Append dated subsections at milestones (post-calibration, post-teleop, post-recording). The "current state" §1 stays at the top; older snapshots accumulate here.

### 2026-05-18 — wrist cameras + upstream merge

- **Upstream sync**: merged `upstream/main` (94 commits) into `main` at `ad93322`. `pyproject.toml` auto-resolved: our `lerobot[feetech,dynamixel]` (drop `[pi]`) and upstream's `editable = true` for the local-path source landed in the same file via non-overlapping line edits. Pushed to `origin/main` from outside the sandbox (the Claude shell's squid proxy at `localhost:3128` 502'd on every `CONNECT github.com:443` during the merge turn).
- **Hardware additions**: 2 wrist cameras mounted on the followers (`DSJ-2062-309`, identical model to the existing scene cam, VID:PID `0c45:64ab`). Three cameras total now. Wrist cams enumerated as USB BUSIDs `4-1` and `4-2`; the existing 4 arms (`4-3, 4-4, 5-1, 5-3`) and scene cam (`5-4`) are unchanged.
- **Scripts**: `scripts/attach_usb_roboclaw.ps1` default BUSID list grew from 5 → 7. Verification step rewritten to count arms (expect 4) AND distinct cameras (expect 3, via `/dev/v4l/by-path/*-video-index0` with `udevadm ID_PATH` fallback) and exit non-zero on mismatch. No changes to `bootstrap_distro.ps1` / `provision_distro.sh` / `deploy.sh` / `install-interop-guard.sh`.
- **Manifest binding (still deferred)**: 3 cameras need physical-port → side-label disambiguation. udev's `ID_PATH` is the stable handle since serial numbers aren't unique across identical-model UVC cameras. Binding work still requires operator-in-the-loop (which cable is left-wrist vs right-wrist vs scene).
- **Open upstream pickup**: `roboclaw/embodied/calibration/so101/auto.py` + `prober.py` arrived in the merge — these may automate the per-arm calibration walk that's been blocked since week 1. Not yet read or evaluated.
