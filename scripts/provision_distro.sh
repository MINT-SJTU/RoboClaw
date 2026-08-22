#!/usr/bin/env bash
# provision_distro.sh - idempotent Linux provisioner run inside the freshly
# imported Ubuntu-roboclaw WSL2 distro by bootstrap_distro.ps1.
#
# Responsibilities (in order, all idempotent):
#   1. Create the target user with passwordless sudo.
#   2. Write /etc/wsl.conf enabling systemd and defaulting to that user.
#   3. Install Docker Engine via get.docker.com (skip if already present).
#   4. Add the user to the docker group.
#   5. Apply the repo's setup-udev.sh rules for CH343 + video4linux.
#
# Runs as root (invoked by bootstrap_distro.ps1 with `wsl -d ... -u root`).
# Safe to re-run: existing users, groups, rules, and packages are detected.

set -euo pipefail

: "${ROBOCLAW_USER:=hafnium}"
TARGET_USER="${ROBOCLAW_USER}"

log()  { printf '[provision] %s\n' "$*"; }
fail() { printf '[provision][ERROR] %s\n' "$*" >&2; exit 1; }

if [[ "$(id -u)" -ne 0 ]]; then
    fail "must run as root"
fi

# --- 1. User + passwordless sudo -------------------------------------------
if id "${TARGET_USER}" >/dev/null 2>&1; then
    log "user '${TARGET_USER}' already exists"
else
    log "creating user '${TARGET_USER}'"
    adduser --disabled-password --gecos '' "${TARGET_USER}"
fi

# Ensure the user is in sudo (harmless if already)
usermod -aG sudo "${TARGET_USER}"

SUDOERS_FILE="/etc/sudoers.d/${TARGET_USER}"
if [[ ! -f "${SUDOERS_FILE}" ]]; then
    log "granting passwordless sudo to '${TARGET_USER}'"
    printf '%s ALL=(ALL) NOPASSWD:ALL\n' "${TARGET_USER}" > "${SUDOERS_FILE}"
    chmod 0440 "${SUDOERS_FILE}"
    visudo -cf "${SUDOERS_FILE}" >/dev/null || fail "invalid sudoers snippet at ${SUDOERS_FILE}"
fi

# --- 2. /etc/wsl.conf (systemd + default user) -----------------------------
log "writing /etc/wsl.conf"
cat > /etc/wsl.conf <<EOF
[boot]
systemd=true

[user]
default=${TARGET_USER}

[network]
generateResolvConf=true

[interop]
enabled=true
appendWindowsPath=false
EOF

# --- 3. Base packages + Docker Engine --------------------------------------
export DEBIAN_FRONTEND=noninteractive
log "apt update + base packages"
apt-get update -qq
apt-get install -y --no-install-recommends \
    ca-certificates curl git usbutils iproute2 udev

# Detect a real Docker Engine install (daemon + CLI) vs. Docker Desktop's
# WSL-integration shim (CLI-only). The shim makes `command -v docker` succeed
# but has no `dockerd`, so we key on the daemon binary.
if command -v dockerd >/dev/null 2>&1 && [[ -x /usr/bin/docker ]]; then
    log "docker already installed ($(docker --version))"
else
    if command -v docker >/dev/null 2>&1; then
        log "docker CLI present but is a Docker Desktop shim; installing real Docker Engine"
    else
        log "installing Docker Engine via get.docker.com"
    fi
    curl -fsSL https://get.docker.com -o /tmp/get-docker.sh
    sh /tmp/get-docker.sh
    rm -f /tmp/get-docker.sh
fi

# --- 4. docker group membership -------------------------------------------
if getent group docker >/dev/null 2>&1; then
    if id -nG "${TARGET_USER}" | tr ' ' '\n' | grep -qx docker; then
        log "'${TARGET_USER}' already in docker group"
    else
        log "adding '${TARGET_USER}' to docker group"
        usermod -aG docker "${TARGET_USER}"
    fi
else
    fail "docker group missing after install — unexpected"
fi

# --- 5. udev rules for CH343 + v4l ----------------------------------------
SCRIPT_DIR="$(dirname "$(readlink -f "$0")")"
UDEV_SCRIPT="${SCRIPT_DIR}/setup-udev.sh"
if [[ -x "${UDEV_SCRIPT}" ]]; then
    log "running ${UDEV_SCRIPT}"
    # setup-udev.sh uses sudo internally; running as root makes that a no-op.
    "${UDEV_SCRIPT}"
else
    log "WARNING: ${UDEV_SCRIPT} not found or not executable; skipping udev install"
fi

# --- 6. WSLInterop guard (re-registers binfmt_misc entry if wiped) --------
GUARD_SCRIPT="${SCRIPT_DIR}/install-interop-guard.sh"
if [[ -x "${GUARD_SCRIPT}" ]]; then
    log "running ${GUARD_SCRIPT}"
    "${GUARD_SCRIPT}"
else
    log "WARNING: ${GUARD_SCRIPT} not found or not executable; skipping interop guard"
fi

# --- 7. Marker file so deploy.sh can skip re-provisioning -----------------
# The marker version is set by deploy.sh when it calls this script; see
# PROVISION_SCHEMA in scripts/deploy.sh. Here we just write "current" — the
# caller is responsible for keeping the version in sync.
mkdir -p /etc/roboclaw

log "provisioning complete"
