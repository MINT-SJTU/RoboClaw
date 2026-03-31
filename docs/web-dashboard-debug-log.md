# Web Dashboard 遥操作/录制 Debug 日志

> 2026-04-01 深夜调试记录

## 最终状态

遥操作和录制均已跑通。核心架构：`session.py` 通过 `SO101Controller` 构建 LeRobot CLI 命令，`subprocess.Popen` 启动子进程执行。

## 遇到的 Bug 及根因

### 1. LeRobot import 路径错误

**现象**: 点击 Connect → `ModuleNotFoundError: No module named 'lerobot.robots.so101_follower'`

**根因**: rdlwicked 的 `session.py` 直接 import LeRobot 内部 config classes（`SO101FollowerConfig`），但 LeRobot 0.5.0 的模块路径是 `so_follower` 而不是 `so101_follower`。

**修复**: 重写 `session.py`，不再直接 import LeRobot 内部类，改为复用 `SO101Controller` 构建 CLI 命令 + subprocess 执行。

### 2. 只支持单臂遥操作

**现象**: 4 臂配置（2 follower + 2 leader）只走单臂路径。

**根因**: `_build_teleop_argv` 只处理了 `len(followers) == 1` 的情况。

**修复**: 加 bimanual 分支，使用 `SO101Controller.teleoperate_bimanual()`。

### 3. bimanual 校准提示导致 EOFError

**现象**: `EOFError: EOF when reading a line` — leader 连接时 LeRobot 调 `input("Press ENTER...")` 但子进程没有 stdin。

**根因**: `subprocess.Popen` 默认 stdin=None，LeRobot 的 `so_leader.calibrate()` 内部有交互式 `input()` 提示。

**修复**: 通过 `stdin=subprocess.PIPE` 创建子进程，写入 `\n\n\n\n` 自动确认校准。

### 4. bimanual 临时校准目录提前删除

**现象**: 子进程启动后找不到校准文件。

**根因**: 使用了 `_bimanual_cal_dirs` context manager，`with` 块结束后 temp dir 被删，但子进程还需要读取。

**修复**: 手动创建 temp dirs，存储在 `self._temp_dirs`，在 `stop/disconnect` 时才清理。

### 5. headless_patch 在无 TTY 环境 crash

**现象**: `termios.error: (25, 'Inappropriate ioctl for device')`

**根因**: `lerobot_wrapper.py` 调 `apply_headless_patch()`，其中 `TTYKeyboardListener.start()` 尝试 `termios.tcgetattr(stdin.fileno())`，但 web 子进程没有 TTY。

**修复**: 在 `start()` 开头加 `if not os.isatty(self._fd): return`。

### 6. Camera preview 占用摄像头设备（最顽固的问题）

**现象**: LeRobot 子进程 `TimeoutError: Timed out waiting for frame from camera`

**根因**: Dashboard 的 `/api/dashboard/camera-preview/{alias}` 端点每次请求都 `cv2.VideoCapture(port)` 打开摄像头拍一帧再 `cap.release()`。但：
- 前端每 1.5 秒轮询 3 个摄像头 = 频繁 open/close
- Linux V4L2 设备释放有延迟，`release()` 不保证立即释放 fd
- 浏览器旧标签页的 JS 代码继续 polling（前端 state 改了但旧代码还在跑）
- OpenCV VideoCapture 对象被 Python GC 延迟回收

**修复（多层防御）**:
- 前端：camera preview 默认关闭（`useState(false)`）
- 前端：teleop/record loading 时自动 `setCamerasEnabled(false)`
- 后端：`camera-preview` 端点检查 `session.cameras_locked`，在 preparing/teleoperating/recording 状态返回 503

### 7. Servo positions 轮询占用串口

**现象**: `ConnectionError: Failed to write 'Lock' on id_=N` — follower 电机通信失败

**根因**: `/api/embodied/servo-positions` 每 500ms 打开串口读所有 follower 电机位置。和 LeRobot 子进程争抢同一个串口设备。虽然加了 state 检查，但 `_read_servo_positions` 在线程中运行，和 `start_recording/start_teleop` 之间有竞态。

**修复**:
- servo-positions 获取 `session._lock` 后才读串口，和 start_teleop/start_recording 互斥
- 引入 `preparing` 状态：先释放 lock → 等 5 秒让 in-flight 读取完成 → 再 lock 启动子进程

### 8. dataset_root 路径错误

**现象**: `FileExistsError: [Errno 17] File exists: '/home/zhaobo/.roboclaw/workspace/embodied/datasets/local'`

**根因**: `_dataset_path()` 返回完整路径 `datasets/local/name`，但代码错误地用了 `.parent` 导致 `dataset_root` 指向 `datasets/local`。

**修复**: 去掉 `.parent`，直接传完整路径。

### 9. repo_id 格式错误

**现象**: `ValueError: not enough values to unpack (expected 2, got 1)`

**根因**: LeRobot 要求 `repo_id` 格式为 `owner/name`（如 `local/dataset`），但传的只有 `dataset`。

**修复**: 加 `local/` 前缀。

### 10. 重复 dataset 名称冲突

**现象**: 第二次录制同名 dataset → `FileExistsError`

**修复**: dataset 名自动追加时间戳 `_{YYYYMMDD_HHMMSS}`。

## 潜在风险

1. **设备释放延迟**: Linux V4L2/ttyUSB 释放不是原子操作。`preparing` 状态的 5 秒等待是经验值，极端情况可能不够。
2. **子进程 crash 后 state 不回退**: 如果 LeRobot 子进程 crash，session state 停留在 `teleoperating`/`recording`，需要用户手动点 Stop 才能恢复。
3. **camera preview 的 OpenCV 资源泄漏**: `_capture_preview_bytes` 每次 open/close 不够可靠。长时间运行后可能累积未释放的 fd。
4. **单进程架构限制**: 所有硬件访问（preview/servo/teleop/record）在同一个 Python 进程中通过 threading 和 subprocess 协调，竞态风险高。理想方案是独立的 hardware server 进程。
