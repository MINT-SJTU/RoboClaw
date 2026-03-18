# ROS2 Install Playbook

Use this guide when a user wants to connect a real or simulated robot and ROS2
is missing, broken, or not yet initialized on the local machine.

This file is meant to be followed by RoboClaw, not just shown to the user.

## RoboClaw Policy

- Prefer installing ROS2 yourself with shell commands when the user has already
  said they want local setup help.
- Prefer a RoboClaw-controlled step-by-step flow over a single opaque installer
  script.
- Prefer supported binary installs over source builds.
- Prefer the most stable supported distro for the user's OS instead of the
  newest short-lived release.
- Ask only for facts that change the install path:
  - operating system and version
  - whether this is native Ubuntu, WSL2, or a remote Linux machine
  - whether GUI tools like RViz are needed
  - whether root / sudo is available
- Do not improvise a custom ROS2 install from memory when this guide already
  covers the case.

## RoboClaw Guided Flow

RoboClaw should treat ROS2 installation as a deterministic prerequisite flow:

1. probe the host
2. choose the supported distro/profile
3. choose the next explicit install step
4. show the exact commands for that step
5. wait for the user to confirm the step or paste the failure output
6. verify `ros2` before continuing setup generation

This flow intentionally borrows the useful shape of fishros/install:

- detect OS codename before selecting packages
- choose a concrete ROS2 distro instead of a generic "install ROS2" step
- choose headless vs desktop explicitly
- finish by wiring shell init and validating the CLI

RoboClaw should still prefer the official ROS apt-source path below as the
default implementation. Fishros-style mirror switching is a fallback when the
official path is blocked or too unreliable for the user's region.

RoboClaw should not hard-code the fishros menu structure or pipe
`http://fishros.com/install` directly into the shell as the default first-run
path.

## Default Decision Tree

### Preferred targets

- Ubuntu 24.04: install ROS 2 Jazzy by default.
- Ubuntu 22.04: install ROS 2 Humble by default.
- Ubuntu 24.04 + explicit request for latest regular release: install Kilted.
- WSL2 with Ubuntu 24.04: install ROS 2 Jazzy inside WSL and handle USB
  passthrough separately.

### Avoid as first choice

- Native macOS for first real-robot setup.
- Native Windows for first real-robot setup.
- Source builds unless binary packages are unavailable or the user explicitly
  needs development-from-source.

For RoboClaw's first real setup flow, the safest recommendation is usually:

1. native Ubuntu 24.04
2. remote Ubuntu 24.04
3. WSL2 Ubuntu 24.04

## Preflight Checks

Run these before choosing the install path:

```bash
uname -a
cat /etc/os-release
echo "$SHELL"
which sudo || true
python3 --version || true
printenv | grep -E 'CONDA|ROS_|RMW_' || true
```

If the machine is WSL, also run:

```bash
grep -i microsoft /proc/version && echo WSL_DETECTED
```

## Install Recipes

### A. Ubuntu 24.04 + Jazzy

Use this as the default stable path for new setups.

#### 1. System setup

```bash
sudo apt update
sudo apt install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale

sudo add-apt-repository universe
sudo apt update
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb
```

#### 2. Install ROS2

Headless / robot execution first:

```bash
sudo apt update
sudo apt upgrade -y
sudo apt install -y ros-jazzy-ros-base python3-rosdep ros-dev-tools
```

If the user explicitly wants RViz, demos, or desktop tools:

```bash
sudo apt install -y ros-jazzy-desktop python3-rosdep ros-dev-tools
```

#### 3. Initialize rosdep

```bash
sudo rosdep init || true
rosdep update
```

#### 4. Shell setup

For bash:

```bash
echo 'source /opt/ros/jazzy/setup.bash' >> ~/.bashrc
source /opt/ros/jazzy/setup.bash
```

For zsh:

```bash
echo 'source /opt/ros/jazzy/setup.zsh' >> ~/.zshrc
source /opt/ros/jazzy/setup.zsh
```

#### 5. Validate

```bash
printenv | grep -i ROS
which ros2
ros2 --help
```

If demo packages are installed, you can also use the official talker/listener
smoke test:

```bash
ros2 run demo_nodes_cpp talker
ros2 run demo_nodes_py listener
```

### B. Ubuntu 22.04 + Humble

Use this only for Jammy machines that should stay on 22.04.

Follow the same steps as Jazzy, but replace package names and setup files:

```bash
sudo apt update
sudo apt install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
export LANG=en_US.UTF-8
locale

sudo add-apt-repository universe
sudo apt update
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo ${UBUNTU_CODENAME:-${VERSION_CODENAME}})_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

sudo apt update
sudo apt upgrade -y
sudo apt install -y ros-humble-ros-base python3-rosdep ros-dev-tools
sudo rosdep init || true
rosdep update
```

Then source:

```bash
source /opt/ros/humble/setup.bash
```

or:

```bash
source /opt/ros/humble/setup.zsh
```

### C. Ubuntu 24.04 + Kilted

Use this only when the user explicitly wants the latest regular release or a
package requires Kilted-specific binaries.

The setup is the same as Jazzy except the package names and setup path:

```bash
sudo apt install -y ros-kilted-ros-base python3-rosdep ros-dev-tools
source /opt/ros/kilted/setup.bash
```

Install `ros-kilted-desktop` only when GUI tools are truly needed.

### D. WSL2 + Ubuntu

For WSL2, install ROS2 inside Ubuntu using the matching Ubuntu recipe above.

Then handle USB passthrough from Windows:

1. On Windows, install `usbipd-win`.
2. In an elevated PowerShell:

```powershell
usbipd list
usbipd bind --busid <BUSID>
usbipd attach --wsl --busid <BUSID>
```

3. Inside WSL, verify the device:

```bash
lsusb
ls /dev/ttyACM* /dev/ttyUSB* 2>/dev/null || true
```

If RViz crashes in WSL2, try:

```bash
export LIBGL_ALWAYS_SOFTWARE=true
rviz2
```

## Mirror And Network Fallbacks

If the official ROS apt source or GitHub release fetch is too slow or blocked,
offer a mirror fallback instead of inventing a different install method.

This is inspired by fishros/install, which explicitly separates:

1. system-source cleanup
2. ROS source selection
3. ROS package installation
4. `rosdep`
5. shell environment setup

RoboClaw should keep that separation, but keep the commands explicit and
reviewable in chat.

Common community mirror options include:

- Tsinghua
- USTC
- Huawei Cloud
- MirrorZ

Use them only as fallbacks when the official path fails repeatedly.

## Post-Install Facts RoboClaw Should Record

After a successful install, update intake or workspace notes with:

- OS and version
- ROS distro installed
- shell init file updated or not
- whether this is native Linux, remote Linux, or WSL2
- whether GUI tools were installed
- whether USB passthrough is required
- current serial device path or by-id path

## Common Failure Modes

### 1. Conda conflicts with apt Python

If using the official Ubuntu `apt` install, do not leave Conda Python ahead of
system Python in `PATH`.

Check:

```bash
printenv | grep CONDA || true
echo "$PATH"
which python3
```

If Conda is active, open a clean shell or temporarily disable Conda before
running ROS2.

### 2. ROS environment not sourced

If `ros2` exists but commands do not behave correctly, verify:

```bash
printenv | grep -i ROS
```

Expected variables include `ROS_VERSION=2` and `ROS_DISTRO=<distro>`.

### 3. ROS2 packages installed under `/opt/ros`, but `ros2` is still missing in PATH

Sometimes the installation succeeded, but the shell init file was never updated.

Check:

```bash
ls /opt/ros
grep -F "/opt/ros/" ~/.bashrc ~/.zshrc 2>/dev/null || true
```

If `/opt/ros/<distro>` exists, add the correct `source /opt/ros/<distro>/setup.*`
line and open a fresh shell before re-checking.

### 4. DDS discovery problems on shared networks

If nodes do not discover each other:

- make sure multicast is enabled on the network interface
- make sure firewall rules are not blocking DDS traffic
- set a unique `ROS_DOMAIN_ID` on shared lab networks

Example:

```bash
export ROS_DOMAIN_ID=7
```

If communication should stay on one machine only, use:

```bash
export ROS_LOCALHOST_ONLY=1
```

### 5. Wayland / RViz issues

If RViz fails on Linux desktops using Wayland:

```bash
QT_QPA_PLATFORM=xcb rviz2
```

### 6. Source vs binary mixing

Do not mix an existing sourced ROS install with a new install path in the same
shell. Use a fresh terminal before switching between binary and source installs.

## When To Use Source Builds

Use a source build only when one of the following is true:

- the OS is unsupported by ROS2 binary packages
- the user needs to patch ROS core packages
- the required robot stack only builds correctly from source

Do not default to source builds for first-run RoboClaw setup.

## Official References

- ROS 2 Jazzy installation: https://docs.ros.org/en/jazzy/Installation.html
- ROS 2 Jazzy Ubuntu deb install:
  https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html
- ROS 2 Humble Ubuntu deb install:
  https://docs.ros.org/en/humble/Installation/Ubuntu-Install-Debs.html
- ROS 2 Kilted Ubuntu deb install:
  https://docs.ros.org/en/kilted/Installation/Ubuntu-Install-Debs.html
- ROS 2 environment setup:
  https://docs.ros.org/en/jazzy/Tutorials/Beginner-CLI-Tools/Configuring-ROS2-Environment.html
- ROS 2 rosdep guide:
  https://docs.ros.org/en/jazzy/Tutorials/Intermediate/Rosdep.html
- ROS 2 installation troubleshooting:
  https://docs.ros.org/en/jazzy/How-To-Guides/Installation-Troubleshooting.html
- ROS 2 release/support matrix:
  https://www.ros.org/reps/rep-2000.html
- Microsoft WSL USB passthrough:
  https://learn.microsoft.com/en-us/windows/wsl/connect-usb
