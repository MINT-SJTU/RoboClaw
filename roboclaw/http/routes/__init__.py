"""Dashboard route registration — one file per API area."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Callable

from fastapi import FastAPI
from loguru import logger

if TYPE_CHECKING:
    from roboclaw.embodied.service import EmbodiedService

_OPTIONAL_DATA_ROUTE_DEPENDENCIES = {"cv2", "numpy", "pandas", "pyarrow"}


def register_all_routes(
    app: FastAPI,
    web_channel: Any,
    service: EmbodiedService,
    get_config: Callable[[], tuple[str, int]],
    *,
    include_data_routes: bool = True,
) -> None:
    """Register every dashboard API group on *app*."""
    from roboclaw.http.routes.session import register_session_routes
    from roboclaw.http.routes.hardware import register_hardware_routes
    from roboclaw.http.routes.setup import register_setup_routes
    from roboclaw.http.routes.devices import register_device_routes
    from roboclaw.http.routes.calibrate import register_calibrate_routes
    from roboclaw.http.routes.datasets import register_dataset_routes
    from roboclaw.http.routes.policies import register_policy_routes
    from roboclaw.http.routes.recovery import register_recovery_routes
    from roboclaw.http.routes.network import register_network_routes
    from roboclaw.http.routes.replay import register_replay_routes
    from roboclaw.http.routes.train import register_train_routes
    from roboclaw.http.routes.infer import register_infer_routes
    from roboclaw.http.routes.hub import register_hub_routes
    from roboclaw.http.routes.chat_uploads import register_chat_upload_routes

    register_chat_upload_routes(app)
    register_session_routes(app, service)
    register_hardware_routes(app, service)
    register_setup_routes(app, service)
    register_device_routes(app, service)
    register_calibrate_routes(app, service)
    register_dataset_routes(app, service)
    register_policy_routes(app, service)
    register_recovery_routes(app)
    register_network_routes(app, get_config)
    register_replay_routes(app, service)
    register_train_routes(app, service)
    register_infer_routes(app, service)
    register_hub_routes(app, service)

    if include_data_routes:
        register_data_routes(app)


def register_data_routes(app: FastAPI) -> None:
    """Register dataset curation/explorer routes that do not require embodied service."""
    from roboclaw.http.routes.explorer import register_explorer_routes

    register_explorer_routes(app)

    try:
        from roboclaw.http.routes.curation import register_curation_routes
    except ModuleNotFoundError as exc:
        if exc.name not in _OPTIONAL_DATA_ROUTE_DEPENDENCIES:
            raise
        logger.warning("Curation routes disabled; optional dependency missing: {}", exc)
        return
    register_curation_routes(app)
