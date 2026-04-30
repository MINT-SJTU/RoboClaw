"""Simulation doctor for ROS 2 navigation environments.

This module is the Phase 2 aggregation layer: it asks ROS 2 discovery what is
installed and what is currently running, then returns a machine-readable
capability manifest. It does not install dependencies, launch simulators, send
navigation goals, or write simulation state.
"""

from __future__ import annotations

import json
from typing import Any, Sequence

from roboclaw.embodied.ros2.discovery import Ros2Discovery
from roboclaw.embodied.simulation.profiles import (
    DEFAULT_PROFILE,
    SimulationProfile,
    TransformCheck,
)


SimulationDoctorProfile = SimulationProfile


class SimulationDoctor:
    """Aggregate ROS 2 discovery checks into a capability manifest."""

    def __init__(
        self,
        discovery: Ros2Discovery | None = None,
        *,
        profile: SimulationDoctorProfile = DEFAULT_PROFILE,
    ) -> None:
        self._discovery = discovery or Ros2Discovery()
        self._profile = profile

    def check(self) -> dict[str, Any]:
        """Return the current simulation capability manifest."""
        ros2_cli = self._discovery.ros2_cli_available()
        if not ros2_cli:
            return self._manifest(
                system={"ros2_cli": False},
                packages={package: False for package in self._profile.packages},
                nodes={node: False for node in self._profile.nodes},
                topics={topic: False for topic in self._profile.topics},
                actions={action: False for action in self._profile.actions},
                services={service: False for service in self._profile.services},
                tf={transform.key: False for transform in self._profile.transforms},
                errors=[{
                    "category": "environment",
                    "message": "ros2 CLI is not available on PATH. Source ROS 2 or install ROS 2 first.",
                }],
            )

        packages = self._discovery.packages_present(self._profile.packages)
        if not _all_ok(packages):
            errors = self._errors(
                packages=packages,
                nodes={node: False for node in self._profile.nodes},
                topics={topic: False for topic in self._profile.topics},
                actions={action: False for action in self._profile.actions},
                services={service: False for service in self._profile.services},
                tf={transform.key: False for transform in self._profile.transforms},
            )
            return self._manifest(
                system={"ros2_cli": True},
                packages=packages,
                nodes={node: False for node in self._profile.nodes},
                topics={topic: False for topic in self._profile.topics},
                actions={action: False for action in self._profile.actions},
                services={service: False for service in self._profile.services},
                tf={transform.key: False for transform in self._profile.transforms},
                errors=errors,
            )

        nodes, node_error, _ = self._presence_from_list(
            self._profile.nodes,
            rich_method="node_list",
            plain_method="nodes",
            query_name="nodes",
        )
        topics, topic_error, discovered_topics = self._presence_from_list(
            self._profile.topics,
            rich_method="topic_list",
            plain_method="topics",
            query_name="topics",
        )
        actions, action_error, _ = self._presence_from_list(
            self._profile.actions,
            rich_method="action_list",
            plain_method="actions",
            query_name="actions",
        )
        services, service_error, _ = self._presence_from_list(
            self._profile.services,
            rich_method="service_list",
            plain_method="services",
            query_name="services",
        )
        query_errors = [
            error for error in (node_error, topic_error, action_error, service_error)
            if error is not None
        ]
        failed_queries = {error["query"] for error in query_errors}
        tf = self._tf_checks(
            self._profile.transforms,
            discovered_topics=discovered_topics,
            topic_query_failed=topic_error is not None,
        )

        errors = self._errors(
            packages=packages,
            nodes=nodes,
            topics=topics,
            actions=actions,
            services=services,
            tf=tf,
            query_errors=query_errors,
            failed_queries=failed_queries,
        )
        return self._manifest(
            system={"ros2_cli": True},
            packages=packages,
            nodes=nodes,
            topics=topics,
            actions=actions,
            services=services,
            tf=tf,
            errors=errors,
        )

    def check_json(self) -> str:
        """Return the current simulation capability manifest as formatted JSON."""
        return json.dumps(self.check(), indent=2, ensure_ascii=False)

    def _tf_checks(
        self,
        expected_transforms: Sequence[TransformCheck],
        *,
        discovered_topics: set[str],
        topic_query_failed: bool,
    ) -> dict[str, bool]:
        if topic_query_failed or not ({"/tf", "/tf_static"} & discovered_topics):
            return {transform.key: False for transform in expected_transforms}
        return {
            transform.key: self._discovery.transform_available(
                transform.target_frame,
                transform.source_frame,
            )
            for transform in expected_transforms
        }

    def _presence_from_list(
        self,
        expected: Sequence[str],
        *,
        rich_method: str,
        plain_method: str,
        query_name: str,
    ) -> tuple[dict[str, bool], dict[str, str] | None, set[str]]:
        rich = getattr(self._discovery, rich_method, None)
        if callable(rich):
            result = rich()
            discovered = set(result.items)
            error = None
            if not result.ok:
                error = {
                    "category": "discovery",
                    "query": query_name,
                    "message": f"Failed to query ROS 2 {query_name}: {result.error}",
                    "command": " ".join(result.command),
                }
            return {name: name in discovered for name in expected}, error, discovered

        plain = getattr(self._discovery, plain_method)
        discovered = set(plain())
        return {name: name in discovered for name in expected}, None, discovered

    def _manifest(
        self,
        *,
        system: dict[str, bool],
        packages: dict[str, bool],
        nodes: dict[str, bool],
        topics: dict[str, bool],
        actions: dict[str, bool],
        services: dict[str, bool],
        tf: dict[str, bool],
        errors: list[dict[str, str]],
    ) -> dict[str, Any]:
        environment_installed = bool(system.get("ros2_cli")) and _all_ok(packages)
        runtime_up = environment_installed and _all_ok(topics)
        tf_ready = runtime_up and _all_ok(tf)
        nav_ready = (
            runtime_up
            and tf_ready
            and _all_ok(actions)
            and _all_ok(nodes)
            and _all_ok(services)
        )
        status = {
            "environment_installed": environment_installed,
            "runtime_up": runtime_up,
            "tf_ready": tf_ready,
            "nav_ready": nav_ready,
        }
        decision, next_steps = self._decision(status, errors)
        return {
            "profile_id": self._profile.profile_id,
            "mode": self._profile.mode,
            "robot": self._profile.robot,
            "simulator": self._profile.simulator,
            "checks": {
                "system": system,
                "packages": packages,
                "nodes": nodes,
                "topics": topics,
                "actions": actions,
                "services": services,
                "tf": tf,
            },
            "status": status,
            "errors": errors,
            "decision": decision,
            "next_steps": next_steps,
        }

    @staticmethod
    def _errors(
        *,
        packages: dict[str, bool],
        nodes: dict[str, bool],
        topics: dict[str, bool],
        actions: dict[str, bool],
        services: dict[str, bool],
        tf: dict[str, bool],
        query_errors: list[dict[str, str]] | None = None,
        failed_queries: set[str] | None = None,
    ) -> list[dict[str, str]]:
        errors: list[dict[str, str]] = list(query_errors or [])
        failed_queries = failed_queries or set()
        missing_packages = _missing(packages)
        if missing_packages:
            errors.append({
                "category": "environment",
                "message": f"Missing ROS 2 packages: {', '.join(missing_packages)}",
            })
            return errors

        missing_topics = [] if "topics" in failed_queries else _missing(topics)
        if missing_topics:
            errors.append({
                "category": "runtime",
                "message": f"Simulation runtime topics are missing: {', '.join(missing_topics)}",
            })

        missing_tf = _missing(tf)
        if missing_tf and "topics" not in failed_queries:
            errors.append({
                "category": "tf",
                "message": f"Required TF transforms are missing: {', '.join(missing_tf)}",
            })

        missing_actions = [] if "actions" in failed_queries else _missing(actions)
        missing_nodes = [] if "nodes" in failed_queries else _missing(nodes)
        missing_services = [] if "services" in failed_queries else _missing(services)
        missing_nav = missing_nodes + missing_actions + missing_services
        if missing_nav:
            errors.append({
                "category": "navigation",
                "message": f"Navigation capabilities are missing: {', '.join(missing_nav)}",
            })

        return errors

    @staticmethod
    def _decision(status: dict[str, bool], errors: list[dict[str, str]]) -> tuple[str, list[str]]:
        if not status["environment_installed"]:
            return "blocked", [
                "Install or source the required ROS 2 simulation dependencies.",
                "Re-run the simulation doctor.",
            ]
        if any(error.get("category") == "discovery" for error in errors):
            return "blocked", [
                "Fix ROS 2 graph discovery errors shown in errors.",
                "Re-run the simulation doctor.",
            ]
        if not status["runtime_up"]:
            return "blocked", [
                "Start the simulator, robot bringup, and navigation stack.",
                "Re-run the simulation doctor.",
            ]
        if not status["tf_ready"]:
            return "needs_reconfiguration", [
                "Fix the localization or robot_state_publisher TF chain.",
                "Re-run the simulation doctor.",
            ]
        if not status["nav_ready"]:
            return "needs_reconfiguration", [
                "Start or fix the missing Nav2 nodes, actions, or services.",
                "Re-run the simulation doctor.",
            ]
        return "ready_for_smoke_test", [
            "Run the navigation smoke test.",
        ]


def run_simulation_doctor(
    *,
    discovery: Ros2Discovery | None = None,
    profile: SimulationDoctorProfile = DEFAULT_PROFILE,
) -> dict[str, Any]:
    """Convenience function for callers that do not need a doctor instance."""
    return SimulationDoctor(discovery=discovery, profile=profile).check()


def run_simulation_doctor_json(
    *,
    discovery: Ros2Discovery | None = None,
    profile: SimulationDoctorProfile = DEFAULT_PROFILE,
) -> str:
    """Convenience function that returns the manifest as formatted JSON."""
    return SimulationDoctor(discovery=discovery, profile=profile).check_json()


def _all_ok(checks: dict[str, bool]) -> bool:
    return all(checks.values()) if checks else True


def _missing(checks: dict[str, bool]) -> list[str]:
    return [name for name, ok in checks.items() if not ok]
