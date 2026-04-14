"""Dashboard route registration — one file per API area."""

from __future__ import annotations

from typing import Any, Callable

from fastapi import FastAPI

from roboclaw.embodied.service import EmbodiedService


def register_all_routes(
    app: FastAPI,
    web_channel: Any,
    service: EmbodiedService,
    get_config: Callable[[], tuple[str, int]],
) -> None:
    """Register every dashboard API group on *app*."""
    from roboclaw.http.routes.session import register_session_routes
    from roboclaw.http.routes.hardware import register_hardware_routes
    from roboclaw.http.routes.setup import register_setup_routes
    from roboclaw.http.routes.devices import register_device_routes
    from roboclaw.http.routes.calibrate import register_calibrate_routes
    from roboclaw.http.routes.datasets import register_dataset_routes
    from roboclaw.http.routes.policies import register_policy_routes
    from roboclaw.http.routes.troubleshoot import register_troubleshoot_routes
    from roboclaw.http.routes.network import register_network_routes
    from roboclaw.http.routes.replay import register_replay_routes
    from roboclaw.http.routes.train import register_train_routes
    from roboclaw.http.routes.infer import register_infer_routes
    from roboclaw.http.routes.hub import register_hub_routes

    register_session_routes(app, service)
    register_hardware_routes(app, service)
    register_setup_routes(app, service)
    register_device_routes(app, service)
    register_calibrate_routes(app, service)
    register_dataset_routes(app, service)
    register_policy_routes(app, service)
    register_troubleshoot_routes(app, service)
    register_network_routes(app, get_config)
    register_replay_routes(app, service)
    register_train_routes(app, service)
    register_infer_routes(app, service)
    register_hub_routes(app, service)
