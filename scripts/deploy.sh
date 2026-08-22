#!/usr/bin/env bash
# deploy.sh - one-shot non-interactive bringup of RoboClaw inside Ubuntu-roboclaw.
#
# Invocation (Windows PowerShell, no admin required):
#
#   wsl -d Ubuntu-roboclaw -u root -- bash /root/bootstrap/deploy.sh
#
# Reads the provisioner scripts from /root/bootstrap/ (canonical in-distro
# location). `bootstrap_distro.ps1` is the sole writer to that directory;
# deploy.sh is a reader only.
#
# Phases (all idempotent, all skippable when marker says they're current):
#   1. Provision — user, wsl.conf, Docker Engine, udev rules, interop guard.
#       Gated on /etc/roboclaw/provisioned.v<N>: skip if marker matches
#       PROVISION_SCHEMA. Bump PROVISION_SCHEMA whenever provision_distro.sh
#       gains or removes a step that must reach existing distros.
#   2. Clone / pull the RoboClaw repo as the target user into ~/RoboClaw.
#   3. docker compose build roboclaw-web.
#   4. docker compose run --rm roboclaw-web onboard.
#
# Interactive steps deploy.sh does NOT run (operator must do afterward):
#   - docker compose run --rm roboclaw-web provider login openai-codex
#   - docker compose up -d roboclaw-web
#   - attach_usb_roboclaw.ps1 (Windows-side, admin PS)

set -euo pipefail

# Bump this when provision_distro.sh gains new install steps so deploy.sh
# re-runs provisioning on existing distros. v2 = initial provisioner +
# install-interop-guard.
PROVISION_SCHEMA=2

: "${ROBOCLAW_USER:=hafnium}"
: "${ROBOCLAW_REPO:=https://github.com/hafnium49/RoboClaw.git}"
: "${BOOTSTRAP_DIR:=/root/bootstrap}"

MARKER_DIR=/etc/roboclaw
MARKER_FILE="${MARKER_DIR}/provisioned.v${PROVISION_SCHEMA}"
REPO_DIR="/home/${ROBOCLAW_USER}/RoboClaw"

LOG()  { printf '\n[deploy %s] %s\n' "$(date +%H:%M:%S)" "$*"; }
FAIL() { printf '\n[deploy][ERROR] %s\n' "$*" >&2; exit 1; }

[[ "$(id -u)" -eq 0 ]] || FAIL "deploy.sh must be invoked as root (wsl -d Ubuntu-roboclaw -u root --)"

REQUIRED=( provision_distro.sh setup-udev.sh install-interop-guard.sh )
for f in "${REQUIRED[@]}"; do
    [[ -f "${BOOTSTRAP_DIR}/${f}" ]] || FAIL "missing ${BOOTSTRAP_DIR}/${f} - re-run bootstrap_distro.ps1 from Windows to refresh /root/bootstrap/"
done

# ---------------------------------------------------------------
# 1. Provision (skippable via marker file).
# ---------------------------------------------------------------
if [[ -f "${MARKER_FILE}" ]]; then
    LOG "Step 1/4 - provisioning SKIPPED (marker ${MARKER_FILE} present)"
else
    LOG "Step 1/4 - provisioning distro (schema v${PROVISION_SCHEMA})"
    # Strip CRLF + normalize perms before invoking the provisioner.
    # (Scripts arrive through bootstrap_distro.ps1's tar; CRLF should never
    # appear, but tolerate if the operator hand-copied.)
    sed -i 's/\r$//' "${BOOTSTRAP_DIR}"/*.sh
    chmod +x "${BOOTSTRAP_DIR}"/*.sh

    ROBOCLAW_USER="${ROBOCLAW_USER}" "${BOOTSTRAP_DIR}/provision_distro.sh"

    mkdir -p "${MARKER_DIR}"
    printf 'provisioned=v%d\ntimestamp=%s\n' "${PROVISION_SCHEMA}" "$(date -u +%FT%TZ)" > "${MARKER_FILE}"
    LOG "provisioning complete; marker written to ${MARKER_FILE}"
fi

# ---------------------------------------------------------------
# 2. Clone/pull the repo as the target user (idempotent).
# ---------------------------------------------------------------
LOG "Step 2/4 - syncing RoboClaw repo as ${ROBOCLAW_USER}"
if [[ -d "${REPO_DIR}/.git" ]]; then
    LOG "repo already present at ${REPO_DIR}; pulling latest"
    sudo -u "${ROBOCLAW_USER}" -H git -C "${REPO_DIR}" pull --recurse-submodules --rebase
else
    sudo -u "${ROBOCLAW_USER}" -H git clone --recurse-submodules "${ROBOCLAW_REPO}" "${REPO_DIR}"
fi
sudo -u "${ROBOCLAW_USER}" -H git -C "${REPO_DIR}" rev-parse --short HEAD

# ---------------------------------------------------------------
# 3. Build the Docker image (can take 5-10 minutes first run).
# ---------------------------------------------------------------
LOG "Step 3/4 - building roboclaw-web image"
# `docker` group membership only takes effect in a fresh login shell, so wrap
# with `sg docker` to acquire the supplementary group for this command.
sudo -u "${ROBOCLAW_USER}" -H -- bash -c \
    "cd '${REPO_DIR}' && sg docker -c 'docker compose build roboclaw-web'"

# ---------------------------------------------------------------
# 4. Onboard (creates ~/.roboclaw/{config.json,workspace/...}).
# ---------------------------------------------------------------
LOG "Step 4/4 - roboclaw onboard (scaffolds ~/.roboclaw)"
WORKSPACE="/home/${ROBOCLAW_USER}/.roboclaw"
mkdir -p "${WORKSPACE}"
chown -R "${ROBOCLAW_USER}:${ROBOCLAW_USER}" "${WORKSPACE}"

# Bypass `docker compose run` here: the compose service pins
# devices=[/dev/ttyACM*,...] which requires USB passthrough to be live.
# `onboard` doesn't touch hardware — it just scaffolds ~/.roboclaw — so
# invoke the already-built image directly via `docker run`, and only
# mount the workspace volume.
sudo -u "${ROBOCLAW_USER}" -H -- bash -c \
    "sg docker -c 'docker run --rm -v ${WORKSPACE}:/root/.roboclaw roboclaw:local onboard'"

# The container ran as root, so scaffolded files are root-owned. Re-chown
# to the target user so they can edit config.json without sudo.
chown -R "${ROBOCLAW_USER}:${ROBOCLAW_USER}" "${WORKSPACE}"

LOG "deploy.sh completed successfully"
cat <<EOF

========================================================================
Next steps (interactive; operator must run):

  1. OAuth provider login (Windows browser will be triggered):
     wsl -d Ubuntu-roboclaw
     cd ~/RoboClaw
     docker compose run --rm roboclaw-web provider login openai-codex

  2. Set the chosen model in ~/.roboclaw/config.json:
     nano ~/.roboclaw/config.json   # agents.defaults.model = openai-codex/...

  3. USB passthrough (Windows admin PowerShell):
     cd <path-to-RoboClaw-on-Windows>\\scripts
     .\\attach_usb_roboclaw.ps1

  4. Bring up the service:
     wsl -d Ubuntu-roboclaw
     cd ~/RoboClaw
     docker compose up -d roboclaw-web
     curl -fsS http://localhost:8765/api/health
========================================================================
EOF
