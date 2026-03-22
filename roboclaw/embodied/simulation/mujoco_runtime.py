"""MuJoCo physics runtime for embodied simulation."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any


class MujocoRuntime:
    """Minimal synchronous MuJoCo runtime."""

    def __init__(self, model_path: str, joint_mapping: dict[str, str] | None = None) -> None:
        self.model_path = model_path
        self.joint_mapping = joint_mapping or {}
        self._model: Any | None = None
        self._data: Any | None = None

    def _import_mujoco(self) -> Any:
        try:
            return importlib.import_module("mujoco")
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError("Python package 'mujoco' is not installed.") from exc

    def _require_running(self) -> tuple[Any, Any]:
        if not self.is_running:
            raise RuntimeError("MuJoCo runtime is not running.")
        return self._model, self._data

    def _alias_joint_name(self, joint_name: str) -> str:
        for alias, mapped_name in self.joint_mapping.items():
            if mapped_name == joint_name:
                return alias
        return joint_name

    @staticmethod
    def _zero_values(values: Any) -> None:
        for idx in range(len(values)):
            values[idx] = 0.0

    @property
    def is_running(self) -> bool:
        return self._model is not None and self._data is not None

    def start(self) -> None:
        if self.is_running:
            return
        model_path = Path(self.model_path).resolve()
        if not model_path.exists():
            raise FileNotFoundError(f"MuJoCo model file not found: {model_path}")
        mujoco = self._import_mujoco()
        import os
        old_cwd = os.getcwd()
        try:
            os.chdir(model_path.parent)
            self._model = mujoco.MjModel.from_xml_path(str(model_path))
        finally:
            os.chdir(old_cwd)
        self._data = mujoco.MjData(self._model)

    def stop(self) -> None:
        self._model = None
        self._data = None

    def step(self) -> None:
        model, data = self._require_running()
        self._import_mujoco().mj_step(model, data)

    def reset(self) -> None:
        _, data = self._require_running()
        self._zero_values(data.qpos)
        if hasattr(data, "qvel"):
            self._zero_values(data.qvel)
        if hasattr(data, "ctrl"):
            self._zero_values(data.ctrl)
        self.step()

    def get_joint_positions(self) -> dict[str, float]:
        model, data = self._require_running()
        mujoco = self._import_mujoco()
        joint_positions: dict[str, float] = {}
        for idx in range(int(model.njnt)):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, idx) or f"joint_{idx}"
            joint_positions[self._alias_joint_name(joint_name)] = float(data.qpos[model.jnt_qposadr[idx]])
        return joint_positions

    def set_joint_targets(self, targets: dict[str, float]) -> None:
        model, data = self._require_running()
        mujoco = self._import_mujoco()
        for joint_name, value in targets.items():
            actuator_name = self.joint_mapping.get(joint_name, joint_name)
            actuator_id = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name))
            if actuator_id < 0:
                raise KeyError(f"MuJoCo actuator not found for joint '{joint_name}' ({actuator_name}).")
            data.ctrl[actuator_id] = float(value)
