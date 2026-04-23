"""Isolated service layer for simulation-first navigation workflows."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

from roboclaw.embodied.simulation.doctor import run_simulation_doctor
from roboclaw.embodied.simulation.lifecycle import SimulationLifecycle
from roboclaw.embodied.simulation.maps import resolve_simulation_map_path
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
        map_id: str | None = None,
        map_path: str | Path | None = None,
        world_launch: str | None = None,
        model: str | None = None,
        ros_domain_id: int | None = None,
        rviz: bool = True,
    ) -> dict[str, Any]:
        try:
            resolved_map_path, selected_map = resolve_simulation_map_path(
                map_id=map_id,
                map_path=map_path,
            )
        except ValueError as exc:
            return {
                "action": "bringup",
                "ok": False,
                "message": str(exc),
                "state_path": str(self._state_path),
                "lifecycle": self._lifecycle.status(),
                "map": self._map_selection(
                    map_id=map_id,
                    map_path=map_path,
                    world_launch=world_launch,
                ),
            }
        requested_world_launch = str(world_launch).strip() if world_launch is not None else ""
        resolved_world_launch = requested_world_launch or (
            selected_map.world_launch if selected_map is not None else None
        )

        manifest = self._run_doctor(profile_id)
        state = self._sync_manifest(manifest)
        map_selection = self._map_selection(
            map_id=map_id,
            map_path=map_path,
            world_launch=world_launch,
            resolved_map_path=resolved_map_path,
            resolved_world_launch=resolved_world_launch,
            selected_map=selected_map,
        )
        if not manifest["status"].get("environment_installed", False):
            return {
                "action": "bringup",
                "ok": False,
                "message": "Simulation bringup blocked: ROS 2 simulation dependencies are not ready.",
                "state_path": str(self._state_path),
                "lifecycle": self._lifecycle.status(),
                "map": map_selection,
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
                "map": map_selection,
                "manifest": manifest,
                "state": state,
            }

        requested_mode = mode.strip().lower()
        lifecycle_mode = requested_mode
        if requested_mode == "nav" and manifest["status"].get("nav_ready", False):
            has_runtime_selection = resolved_map_path is not None or resolved_world_launch is not None
            if has_runtime_selection and not self._active_runtime_matches(
                state,
                resolved_map_path,
                selected_map,
                resolved_world_launch,
            ):
                return {
                    "action": "bringup",
                    "ok": False,
                    "already_running": True,
                    "message": (
                        "Navigation stack is already ready, but the requested map/world is not the recorded active "
                        "runtime. Shutdown and bring up again with the requested selection to prove it is applied."
                    ),
                    "state_path": str(self._state_path),
                    "lifecycle": self._lifecycle.status(),
                    "map": {
                        **map_selection,
                        "applied": False,
                        "active_map": dict(state.get("paths", {})),
                    },
                    "manifest": manifest,
                    "state": state,
                }
            return {
                "action": "bringup",
                "ok": True,
                "already_running": True,
                "message": "Navigation stack already appears ready; not starting a duplicate bringup.",
                "state_path": str(self._state_path),
                "lifecycle": self._lifecycle.status(),
                "map": {
                    **map_selection,
                    "applied": not has_runtime_selection
                    or self._active_runtime_matches(
                        state,
                        resolved_map_path,
                        selected_map,
                        resolved_world_launch,
                    ),
                    "active_map": dict(state.get("paths", {})),
                },
                "manifest": manifest,
                "state": state,
            }
        if requested_mode == "nav" and manifest["status"].get("runtime_up", False):
            has_runtime_selection = resolved_map_path is not None or resolved_world_launch is not None
            if (
                has_runtime_selection
                and self._requires_proven_runtime_world(selected_map, resolved_world_launch)
                and not self._active_runtime_matches(
                    state,
                    resolved_map_path,
                    selected_map,
                    resolved_world_launch,
                )
            ):
                return {
                    "action": "bringup",
                    "ok": False,
                    "already_running": True,
                    "message": (
                        "Simulation runtime is already up, but the requested map/world is not the recorded active "
                        "runtime. Shutdown and bring up again with the requested selection so Gazebo uses the "
                        "correct world."
                    ),
                    "state_path": str(self._state_path),
                    "lifecycle": self._lifecycle.status(),
                    "map": {
                        **map_selection,
                        "applied": False,
                        "active_map": dict(state.get("paths", {})),
                    },
                    "manifest": manifest,
                    "state": state,
                }
            lifecycle_mode = "nav-only"

        result = self._lifecycle.bringup(
            mode=lifecycle_mode,
            map_path=resolved_map_path,
            world_launch=resolved_world_launch,
            model=model,
            ros_domain_id=ros_domain_id,
            rviz=rviz,
        )
        updated = self._update_paths(
            state,
            launch="robotics/scripts/run_sim.sh",
            map_id=selected_map.map_id if selected_map is not None else map_id,
            map_path=resolved_map_path,
            world_launch=resolved_world_launch,
            semantic_graph_path=selected_map.semantic_graph_path if selected_map is not None else None,
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
            "map": {
                **map_selection,
                "applied": bool(result.get("ok"))
                and (resolved_map_path is not None or resolved_world_launch is not None),
            },
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
        map_id: str | None,
        map_path: str | Path | None,
        world_launch: str | None,
        semantic_graph_path: str | None = None,
    ) -> dict[str, Any]:
        updated = dict(state)
        paths = dict(updated.get("paths", {}))
        paths["launch"] = launch
        if map_id is not None:
            paths["map_id"] = str(map_id)
        if map_path is not None:
            paths["map"] = str(map_path)
        if world_launch is not None:
            paths["world"] = world_launch
        if semantic_graph_path is not None:
            paths["semantic_graph"] = semantic_graph_path
        updated["paths"] = paths
        return updated

    @staticmethod
    def _map_selection(
        *,
        map_id: str | None,
        map_path: str | Path | None,
        world_launch: str | None = None,
        resolved_map_path: str | Path | None = None,
        resolved_world_launch: str | None = None,
        selected_map: Any | None = None,
    ) -> dict[str, Any]:
        explicit_map_path = str(map_path).strip() if map_path is not None else ""
        explicit_world_launch = str(world_launch).strip() if world_launch is not None else ""
        return {
            "requested_map_id": map_id,
            "explicit_map_path": explicit_map_path or None,
            "explicit_world_launch": explicit_world_launch or None,
            "selected_map_id": selected_map.map_id if selected_map is not None else None,
            "selected_map_description": selected_map.description if selected_map is not None else None,
            "selected_world_launch": selected_map.world_launch if selected_map is not None else None,
            "selected_semantic_graph": (
                selected_map.semantic_graph_path if selected_map is not None else None
            ),
            "resolved_map_path": str(resolved_map_path) if resolved_map_path is not None else None,
            "resolved_world_launch": resolved_world_launch,
        }

    @staticmethod
    def _active_runtime_matches(
        state: dict[str, Any],
        resolved_map_path: str | Path | None,
        selected_map: Any | None,
        resolved_world_launch: str | None,
    ) -> bool:
        paths = state.get("paths", {})
        if not isinstance(paths, dict):
            return False

        map_matches = True
        if resolved_map_path is not None:
            map_matches = False
            if selected_map is not None and paths.get("map_id") == selected_map.map_id:
                map_matches = True
            else:
                active = str(paths.get("map", "") or "")
                requested = str(resolved_map_path)
                map_matches = bool(active) and (
                    active == requested or active.endswith(requested) or requested.endswith(active)
                )

        world_matches = True
        if resolved_world_launch is not None:
            active_world = str(paths.get("world", "") or "")
            world_matches = bool(active_world) and active_world == resolved_world_launch

        return map_matches and world_matches

    @staticmethod
    def _requires_proven_runtime_world(selected_map: Any | None, resolved_world_launch: str | None) -> bool:
        if selected_map is not None and selected_map.map_id != "default":
            return True
        return resolved_world_launch not in (None, "turtlebot3_world.launch.py")
