# bootstrap_distro.ps1 - one-shot Windows orchestrator for the dedicated
# RoboClaw WSL2 distro + in-distro Docker Engine.
#
# Run ONCE from an elevated PowerShell window. Idempotent: re-running simply
# skips any step that is already complete.
#
# What this does:
#   1. Download the Ubuntu 24.04 WSL rootfs (cached under %USERPROFILE%\wsl).
#   2. Import it as the WSL2 distro specified by -Distro (default Ubuntu-roboclaw).
#   3. Copy the sibling provision_distro.sh and setup-udev.sh into the distro
#      and execute the provisioner. That script creates the user, configures
#      /etc/wsl.conf, installs Docker Engine via get.docker.com, and installs
#      the udev rules from scripts/setup-udev.sh.
#   4. Terminate the distro so the [user] default + [boot] systemd entries in
#      /etc/wsl.conf take effect on the next launch.
#
# After this finishes, the full RoboClaw deployment is:
#   .\scripts\bootstrap_distro.ps1                # this script (one time)
#   .\scripts\attach_usb_roboclaw.ps1             # per session / on log-on
#   wsl -d Ubuntu-roboclaw                        # then build + compose up

[CmdletBinding()]
param(
    [string]$Distro  = "Ubuntu-roboclaw",
    [string]$User    = "hafnium",
    [string]$WslRoot = (Join-Path $env:USERPROFILE "wsl\ubuntu-roboclaw"),
    [string]$RootfsUrl = "https://cloud-images.ubuntu.com/wsl/releases/24.04/current/ubuntu-noble-wsl-amd64-24.04-lts.rootfs.tar.gz"
)

$ErrorActionPreference = "Stop"

function Assert-Admin {
    $principal = New-Object Security.Principal.WindowsPrincipal(
        [Security.Principal.WindowsIdentity]::GetCurrent())
    if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
        throw "Run from an elevated PowerShell (Run as Administrator)."
    }
}

function Assert-CommandOnPath($name) {
    if (-not (Get-Command $name -ErrorAction SilentlyContinue)) {
        throw "Required command '$name' not found on PATH."
    }
}

function Test-Distro($name) {
    $lines = wsl --list --quiet 2>$null
    $names = @()
    foreach ($l in ($lines -split "`r?`n")) { if ($l) { $names += $l.Trim() } }
    return $names -contains $name
}

Assert-Admin
Assert-CommandOnPath wsl

# ------------------------------------------------------------------------
# 1. Rootfs download (cached)
# ------------------------------------------------------------------------
New-Item -ItemType Directory -Force -Path $WslRoot | Out-Null
$rootfs = Join-Path $WslRoot "rootfs.tar.gz"

if (Test-Path $rootfs) {
    Write-Host "[1/4] Rootfs already downloaded at $rootfs" -ForegroundColor DarkGray
} else {
    Write-Host "[1/4] Downloading Ubuntu 24.04 WSL rootfs (~500 MB)..."
    try {
        Start-BitsTransfer -Source $RootfsUrl -Destination $rootfs -ErrorAction Stop
    } catch {
        # Fallback for environments where BITS is disabled
        Write-Host "      BITS unavailable; falling back to Invoke-WebRequest..."
        Invoke-WebRequest -Uri $RootfsUrl -OutFile $rootfs -UseBasicParsing
    }
    Write-Host ("      Download OK: {0} MB" -f [int]((Get-Item $rootfs).Length / 1MB))
}

# ------------------------------------------------------------------------
# 2. Import WSL distro (idempotent)
# ------------------------------------------------------------------------
if (Test-Distro $Distro) {
    Write-Host "[2/4] WSL distro '$Distro' already exists; skipping import" -ForegroundColor DarkGray
} else {
    Write-Host "[2/4] Importing rootfs as WSL distro '$Distro'..."
    wsl --import $Distro $WslRoot $rootfs --version 2
    if ($LASTEXITCODE -ne 0) { throw "wsl --import failed (exit $LASTEXITCODE)" }
    Write-Host "      Import OK"
}

# ------------------------------------------------------------------------
# 3. Provision inside the distro (user, wsl.conf, Docker Engine, udev)
# ------------------------------------------------------------------------
$scriptDir        = Split-Path -Parent $MyInvocation.MyCommand.Path
$provisionLocal   = Join-Path $scriptDir "provision_distro.sh"
$udevLocal        = Join-Path $scriptDir "setup-udev.sh"

if (-not (Test-Path $provisionLocal)) { throw "Missing provisioner at $provisionLocal" }
if (-not (Test-Path $udevLocal))      { throw "Missing udev script at $udevLocal" }

Write-Host "[3/4] Provisioning '$Distro' (user, Docker Engine, udev)..."

# Ship the two scripts into the distro via tar-over-stdin. This avoids any
# dependence on where the repo lives on Windows and sidesteps path-quoting
# issues across the WSL interop boundary.
$tmpTar = [System.IO.Path]::GetTempFileName() + ".tar"
tar -cf $tmpTar -C $scriptDir provision_distro.sh setup-udev.sh install-interop-guard.sh deploy.sh
if ($LASTEXITCODE -ne 0) { throw "Failed to tar provisioner scripts" }

Get-Content -Raw -Encoding Byte $tmpTar |
    wsl -d $Distro -u root -- bash -c "mkdir -p /root/bootstrap && cd /root/bootstrap && tar -xf - && chmod +x provision_distro.sh setup-udev.sh install-interop-guard.sh deploy.sh"
if ($LASTEXITCODE -ne 0) {
    Remove-Item $tmpTar -Force -ErrorAction SilentlyContinue
    throw "Failed to ship provisioner scripts into '$Distro' (exit $LASTEXITCODE)"
}
Remove-Item $tmpTar -Force -ErrorAction SilentlyContinue

wsl -d $Distro -u root -- bash -c "ROBOCLAW_USER='$User' /root/bootstrap/provision_distro.sh"
if ($LASTEXITCODE -ne 0) {
    throw "Provisioning failed (exit $LASTEXITCODE). Re-run this script; it is idempotent."
}
Write-Host "      Provisioning OK"

# ------------------------------------------------------------------------
# 4. Apply wsl.conf (default user + systemd) by terminating the distro
# ------------------------------------------------------------------------
Write-Host "[4/4] Restarting distro to apply /etc/wsl.conf..."
wsl --terminate $Distro | Out-Null
Start-Sleep -Seconds 2

Write-Host ""
Write-Host "=== Post-bootstrap verification ==="
wsl -d $Distro -- bash -lc "echo User: `$(whoami); echo Groups: `$(id -nG); docker --version 2>&1 || echo 'docker: not in PATH for this shell yet'"

# Also stage deploy.sh to the Windows-side cache for operators who prefer to
# invoke it via /mnt/c/... (still points at the same content).
$stagedDeploy = Join-Path $WslRoot "bootstrap\deploy.sh"
New-Item -ItemType Directory -Force -Path (Split-Path $stagedDeploy) | Out-Null
Copy-Item -Force (Join-Path $scriptDir "deploy.sh") $stagedDeploy

Write-Host ""
Write-Host "Bootstrap complete." -ForegroundColor Green
Write-Host "Next steps:"
Write-Host "  1. .\scripts\attach_usb_roboclaw.ps1                               # attach SO-101 + camera"
Write-Host "  2. wsl -d $Distro -u root -- bash /root/bootstrap/deploy.sh        # end-to-end bringup"
Write-Host "     (equivalently: bash $stagedDeploy)"
Write-Host "  3. docker compose run --rm roboclaw-web provider login openai-codex  # OAuth"
Write-Host "  4. docker compose up -d roboclaw-web                               # start the service"
Write-Host "  6. docker compose run --rm roboclaw-web provider login openai-codex"
Write-Host "  7. docker compose up -d roboclaw-web"
