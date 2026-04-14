"""Policy listing utilities for the dashboard."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def list_policies(root: Path) -> list[dict[str, Any]]:
    """Scan *root* for trained-policy directories and return summary dicts.

    Accepts both layouts:
      - ``{root}/{name}/checkpoints/last/pretrained_model/`` (lerobot training output)
      - ``{root}/{name}/pretrained_model/``                  (downloaded or manually placed)
    """
    if not root.is_dir():
        return []

    policies: list[dict[str, Any]] = []
    for policy_dir in sorted(root.iterdir()):
        if not policy_dir.is_dir():
            continue
        checkpoint = _resolve_checkpoint(policy_dir)
        if checkpoint is None:
            continue
        entry: dict[str, Any] = {
            "name": policy_dir.name,
            "checkpoint": str(checkpoint),
        }
        _enrich(entry, checkpoint)
        policies.append(entry)
    return policies


def _resolve_checkpoint(policy_dir: Path) -> Path | None:
    for candidate in (
        policy_dir / "checkpoints" / "last" / "pretrained_model",
        policy_dir / "pretrained_model",
    ):
        if candidate.is_dir():
            return candidate
    # HuggingFace snapshot layout: safetensors directly in policy_dir
    if any(policy_dir.glob("*.safetensors")):
        return policy_dir
    return None


def _enrich(entry: dict[str, Any], checkpoint_dir: Path) -> None:
    """Pull dataset and step count out of train_config.json if readable."""
    train_config = checkpoint_dir / "train_config.json"
    if not train_config.is_file() or not os.access(train_config, os.R_OK):
        return
    cfg = json.loads(train_config.read_text(encoding="utf-8"))
    entry["dataset"] = cfg.get("dataset", {}).get("repo_id", "")
    entry["steps"] = cfg.get("steps", 0)
