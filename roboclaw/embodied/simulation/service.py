"""Isolated service layer for simulation-first navigation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from roboclaw.embodied.simulation.doctor import run_simulation_doctor
from roboclaw.embodied.simulation.lifecycle import SimulationLifecycle
from roboclaw.embodied.simulation.profiles import get_profile
from roboclaw.embodied.simulation.state import (
    get_simulation_state_path,
    load_simulation_state,
    save_simulation_state,
    sync_from_doctor_manifest,
)


DoctorRunner = Callable[[str | None], dict[str, Any]]


def _default_doctor_runner(profile_id: str | None) -> dict[str, Any]:
    profile = get_profile(profile_id)
    return run_simulation_doctor(profile=profile)


class SimulationService:
    """Standalone simulation service that avoids EmbodiedService."""

    def __init__(
        self,
        *,
        lifecycle: SimulationLifecycle | None = None,
        doctor_runner: DoctorRunner | None = None,
        state_path: str | Path | None = None,
    ) -> None:
        self._lifecycle = lifecycle or SimulationLifecycle()
        self._doctor_runner = doctor_runner or _default_doctor_runner
        self._state_path = Path(state_path).expanduser() if state_path is not None else get_simulation_state_path()

    @property
    def state_path(self) -> Path:
        return self._state_path

    @property
    def lifecycle(self) -> SimulationLifecycle:
        return self._lifecycle

    def state_show(self) -> dict[str, Any]:
        return {
            "action": "state_show",
            "ok": True,
            "state_path": str(self._state_path),
            "lifecycle": self._lifecycle.status(),
            "state": self._load_state(),
        }

    def doctor(self, *, profile_id: str | None = None) -> dict[str, Any]:
        manifest = self._run_doctor(profile_id)
        state = self._sync_manifest(manifest)
        return {
            "action": "doctor",
            "ok": True,
            "state_path": str(self._state_path),
            "lifecycle": self._lifecycle.status(),
            "manifest": manifest,
            "state": state,
        }

    def bringup(
        self,
        *,
        profile_id: str | None = None,
        mode: str = "nav",
        map_path: str | Path | None = None,
        world_launch: str | None = None,
        model: str | None = None,
        ros_domain_id: int | None = None,
        rviz: bool = True,
    ) -> dict[str, Any]:
        manifest = self._run_doctor(profile_id)
        state = self._sync_manifest(manifest)
        if not manifest["status"].get("environment_installed", False):
            return {
                "action": "bringup",
                "ok": False,
                "message": "Simulation bringup blocked: ROS 2 simulation dependencies are not ready.",
                "state_path": str(self._state_path),
                "lifecycle": self._lifecycle.status(),
                "manifest": manifest,
                "state": state,
            }
        if any(error.get("category") == "discovery" for error in manifest.get("errors", [])):
            return {
                "action": "bringup",
                "ok": False,
                "message": "Simulation bringup blocked: ROS 2 graph discovery is failing.",
                "state_path": str(self._state_path),
                "lifecycle": self._lifecycle.status(),
                "manifest": manifest,
                "state": state,
            }

        requested_mode = mode.strip().lower()
        lifecycle_mode = requested_mode
        if requested_mode == "nav" and manifest["status"].get("nav_ready", False):
            return {
                "action": "bringup",
                "ok": True,
                "already_running": True,
                "message": "Navigation stack already appears ready; not starting a duplicate bringup.",
                "state_path": str(self._state_path),
                "lifecycle": self._lifecycle.status(),
                "manifest": manifest,
                "state": state,
            }
        if requested_mode == "nav" and manifest["status"].get("runtime_up", False):
            lifecycle_mode = "nav-only"

        result = self._lifecycle.bringup(
            mode=lifecycle_mode,
            map_path=map_path,
            world_launch=world_launch,
            model=model,
            ros_domain_id=ros_domain_id,
            rviz=rviz,
        )
        updated = self._update_paths(
            state,
            launch="robotics/scripts/run_sim.sh",
            map_path=map_path,
            world_launch=world_launch,
        )
        saved = save_simulation_state(updated, self._state_path)
        return {
            "action": "bringup",
            "ok": bool(result.get("ok")),
            "requested_mode": requested_mode,
            "lifecycle_mode": lifecycle_mode,
            "message": result.get("message", ""),
            "state_path": str(self._state_path),
            "lifecycle": result.get("process", self._lifecycle.status()),
            "manifest": manifest,
            "state": saved,
        }

    def shutdown(self) -> dict[str, Any]:
        result = self._lifecycle.shutdown()
        return {
            "action": "shutdown",
            "ok": bool(result.get("ok")),
            "message": result.get("message", ""),
            "state_path": str(self._state_path),
            "lifecycle": self._lifecycle.status(),
            "details": result,
        }

    def reset_world(
        self,
        *,
        service_name: str = "/reset_simulation",
        timeout_s: float = 10.0,
    ) -> dict[str, Any]:
        result = self._lifecycle.reset_world(service_name=service_name, timeout_s=timeout_s)
        return {
            "action": "reset_world",
            "ok": bool(result.get("ok")),
            "message": result.get("message", ""),
            "state_path": str(self._state_path),
            "lifecycle": self._lifecycle.status(),
            "details": result,
        }

    def _load_state(self) -> dict[str, Any]:
        return load_simulation_state(self._state_path)

    def _run_doctor(self, profile_id: str | None) -> dict[str, Any]:
        return self._doctor_runner(profile_id)

    def _sync_manifest(self, manifest: dict[str, Any]) -> dict[str, Any]:
        synced = sync_from_doctor_manifest(manifest, self._load_state())
        return save_simulation_state(synced, self._state_path)

    def _update_paths(
        self,
        state: dict[str, Any],
        *,
        launch: str,
        map_path: str | Path | None,
        world_launch: str | None,
    ) -> dict[str, Any]:
        updated = dict(state)
        paths = dict(updated.get("paths", {}))
        paths["launch"] = launch
        if map_path is not None:
            paths["map"] = str(map_path)
        if world_launch is not None:
            paths["world"] = world_launch
        updated["paths"] = paths
        return updated
