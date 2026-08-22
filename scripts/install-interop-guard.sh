#!/usr/bin/env bash
# install-interop-guard.sh — install a systemd oneshot + timer that re-registers
# the WSLInterop binfmt_misc entry whenever it is wiped.
#
# Why: cross-distro wsl.exe calls leaving a target distro can unregister the
# shared WSLInterop entry, which breaks `wsl.exe`, `pwsh.exe`, `usbipd.exe` and
# any other Windows PE invocation from WSL until the kernel-level entry is
# restored. The register file is root-only-writable, so unprivileged shells
# cannot recover on their own. This guard fires every 30 seconds (idle) and
# re-registers the exact bytes WSL itself would write.
#
# Safe to run multiple times; unit files are overwritten verbatim.

set -euo pipefail

[ "$(id -u)" = 0 ] || { echo "install-interop-guard.sh: run as root" >&2; exit 1; }

if ! command -v systemctl >/dev/null 2>&1; then
    echo "install-interop-guard.sh: systemctl not available; systemd is required" >&2
    exit 1
fi

SERVICE=/etc/systemd/system/wsl-interop-guard.service
TIMER=/etc/systemd/system/wsl-interop-guard.timer

cat > "$SERVICE" <<'EOF'
[Unit]
Description=Re-register WSLInterop binfmt_misc if missing
After=systemd-binfmt.service
Documentation=https://github.com/hafnium49/RoboClaw

[Service]
Type=oneshot
# Idempotency check lives inside the command (not ConditionPathExists) so we
# emit no journal line when interop is healthy. EEXIST / EBUSY on concurrent
# write are swallowed so TOCTOU races don't flip the unit to `failed`.
ExecStart=/bin/sh -c '[ -e /proc/sys/fs/binfmt_misc/WSLInterop ] && exit 0; printf ":WSLInterop:M::MZ::/init:FP\n" > /proc/sys/fs/binfmt_misc/register 2>/dev/null; [ -e /proc/sys/fs/binfmt_misc/WSLInterop ]'
SuccessExitStatus=0 1

# Hardening — service has exactly one job: write to one procfs path.
ProtectSystem=strict
ProtectHome=yes
PrivateTmp=yes
NoNewPrivileges=yes
ReadWritePaths=/proc/sys/fs/binfmt_misc
CapabilityBoundingSet=CAP_SYS_ADMIN
SystemCallFilter=@system-service
LogLevelMax=warning
EOF

cat > "$TIMER" <<'EOF'
[Unit]
Description=Poll WSLInterop binfmt_misc availability

[Timer]
OnBootSec=5s
OnUnitInactiveSec=30s
AccuracySec=1s
Unit=wsl-interop-guard.service

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now wsl-interop-guard.timer

echo "wsl-interop-guard installed."
echo "  service: $SERVICE"
echo "  timer:   $TIMER"
echo "  active:  $(systemctl is-active wsl-interop-guard.timer)"
