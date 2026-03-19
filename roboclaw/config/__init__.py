"""Configuration module for roboclaw."""

from __future__ import annotations

from roboclaw.config.loader import get_config_path
from roboclaw.config.paths import (
    ensure_robot_calibration_file,
    get_bridge_install_dir,
    get_calibration_root,
    get_cli_history_path,
    get_cron_dir,
    get_data_dir,
    get_legacy_sessions_dir,
    get_logs_dir,
    get_media_dir,
    get_robot_calibration_dir,
    get_runtime_subdir,
    get_workspace_path,
    resolve_active_serial_device_path,
    resolve_serial_by_id_path,
)

__all__ = [
    "Config",
    "load_config",
    "get_config_path",
    "get_data_dir",
    "get_calibration_root",
    "get_robot_calibration_dir",
    "ensure_robot_calibration_file",
    "get_runtime_subdir",
    "get_media_dir",
    "get_cron_dir",
    "get_logs_dir",
    "get_workspace_path",
    "get_cli_history_path",
    "get_bridge_install_dir",
    "get_legacy_sessions_dir",
    "resolve_serial_by_id_path",
    "resolve_active_serial_device_path",
]


def __getattr__(name: str):
    if name == "Config":
        from roboclaw.config.schema import Config

        return Config
    if name == "load_config":
        from roboclaw.config.loader import load_config

        return load_config
    raise AttributeError(f"module 'roboclaw.config' has no attribute {name!r}")
