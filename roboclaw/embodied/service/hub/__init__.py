"""HubService — upload/download datasets and policies to/from HuggingFace Hub."""

from __future__ import annotations

import asyncio
from typing import TYPE_CHECKING, Any

from roboclaw.embodied.board.channels import CH_HUB
from roboclaw.embodied.command.helpers import policy_path
from roboclaw.embodied.service.hub.progress import make_tqdm_class

if TYPE_CHECKING:
    from roboclaw.embodied.embodiment.manifest import Manifest
    from roboclaw.embodied.service import EmbodiedService


class HubService:
    """Upload/download datasets and policies to/from HuggingFace Hub.

    NOT a Session subclass — pure Python API via huggingface_hub,
    no subprocess, no hardware interaction.
    """

    def __init__(self, parent: EmbodiedService) -> None:
        self._parent = parent

    # ── Config ───────────────────────────────────────────────────────

    def _hf_defaults(self) -> dict[str, str]:
        """Load HuggingFace config (endpoint, token, proxy) from config.json."""
        from roboclaw.config.loader import load_runtime_config
        cfg = load_runtime_config().huggingface
        return {"endpoint": cfg.endpoint, "token": cfg.token, "proxy": cfg.proxy}

    # ── Datasets ─────────────────────────────────────────────────────

    async def push_dataset(
        self, manifest: Manifest, kwargs: dict[str, Any], tty_handoff: Any,
    ) -> str:
        repo_id = kwargs["repo_id"]
        dataset_id = kwargs["dataset_id"]
        defaults = self._hf_defaults()
        token = kwargs.get("token", "") or defaults["token"]
        private = kwargs.get("private", False)

        return await asyncio.to_thread(
            self._parent.datasets.push_dataset,
            dataset_id,
            repo_id=repo_id,
            token=token,
            private=private,
            endpoint=defaults["endpoint"],
            proxy=defaults["proxy"],
        )

    async def pull_dataset(
        self, manifest: Manifest, kwargs: dict[str, Any], tty_handoff: Any,
    ) -> str:
        repo_id = kwargs["repo_id"]
        dataset_id = kwargs.get("dataset_id", "")
        defaults = self._hf_defaults()
        token = kwargs.get("token", "") or defaults["token"]

        tqdm_cls = make_tqdm_class(self._parent.board, f"pull_dataset:{dataset_id or repo_id}")
        dataset = await asyncio.to_thread(
            self._parent.datasets.pull_dataset,
            repo_id=repo_id,
            dataset_id=dataset_id,
            token=token,
            tqdm_class=tqdm_cls,
            endpoint=defaults["endpoint"],
            proxy=defaults["proxy"],
        )
        self._emit_done(f"pull_dataset:{dataset.id}")
        return f"Dataset '{dataset.label}' downloaded from {repo_id}"

    # ── Policies ─────────────────────────────────────────────────────

    async def push_policy(
        self, manifest: Manifest, kwargs: dict[str, Any], tty_handoff: Any,
    ) -> str:
        repo_id = kwargs["repo_id"]
        name = kwargs["name"]
        defaults = self._hf_defaults()
        token = kwargs.get("token", "") or defaults["token"]
        private = kwargs.get("private", False)

        local = policy_path(manifest, name)
        if not local.is_dir():
            raise ValueError(f"Policy '{name}' not found")

        from roboclaw.embodied.service.hub.transfer import push_folder

        url = await asyncio.to_thread(
            push_folder,
            local_path=local,
            repo_id=repo_id,
            repo_type="model",
            token=token,
            private=private,
            endpoint=defaults["endpoint"],
            proxy=defaults["proxy"],
        )
        return f"Policy '{name}' pushed to {repo_id}\n{url}"

    async def pull_policy(
        self, manifest: Manifest, kwargs: dict[str, Any], tty_handoff: Any,
    ) -> str:
        repo_id = kwargs["repo_id"]
        name = kwargs.get("name", "") or repo_id.rsplit("/", 1)[-1]
        defaults = self._hf_defaults()
        token = kwargs.get("token", "") or defaults["token"]

        local = policy_path(manifest, name)
        tqdm_cls = make_tqdm_class(self._parent.board, f"pull_policy:{name}")

        from roboclaw.embodied.service.hub.transfer import pull_repo

        await asyncio.to_thread(
            pull_repo,
            repo_id=repo_id,
            repo_type="model",
            local_dir=local,
            token=token,
            tqdm_class=tqdm_cls,
            endpoint=defaults["endpoint"],
            proxy=defaults["proxy"],
        )
        self._emit_done(f"pull_policy:{name}")
        return f"Policy '{name}' downloaded from {repo_id}"

    # ── Helpers ──────────────────────────────────────────────────────

    def _emit_done(self, operation: str) -> None:
        self._parent.board.emit_sync(CH_HUB, {
            "operation": operation,
            "progress_percent": 100.0,
            "done": True,
        })
