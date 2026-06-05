"""Cloud training artifact and metric extraction helpers."""

from __future__ import annotations

import re
from typing import Any

_CLOUD_ARTIFACT_PATH_RE = re.compile(
    r"(?P<path>/(?:root/autodl-tmp|workspace|tmp)/[^\s'\"`]+?\.(?:json|txt|log))"
)


def _cloud_artifacts_from_payload(payload: dict[str, Any]) -> list[dict[str, Any]]:
    text = "\n".join(
        str(payload.get(key) or "")
        for key in ("message", "log_tail", "logTail", "error")
    )
    candidates: list[tuple[str, str]] = []
    captured = re.findall(r"__EVO_RLINF_METRICS_CAPTURED__=([^\s]+)", text)
    candidates.extend(("metrics", path.strip()) for path in captured)
    for match in _CLOUD_ARTIFACT_PATH_RE.finditer(text):
        path = match.group("path").rstrip(".,;)")
        lower = path.lower()
        if "metrics" in lower:
            kind = "metrics"
        elif "rollout_summary" in lower:
            kind = "rollout_summary"
        elif lower.endswith(".log"):
            kind = "log"
        else:
            kind = "artifact"
        candidates.append((kind, path))
    log_path = str(payload.get("log_path") or "").strip()
    if log_path:
        candidates.append(("log", log_path))
    seen: set[str] = set()
    artifacts: list[dict[str, Any]] = []
    for kind, path in candidates:
        if not path or path in seen:
            continue
        seen.add(path)
        artifacts.append({
            "kind": kind,
            "name": path.rsplit("/", 1)[-1],
            "path": path,
            "previewable": path.endswith((".json", ".txt", ".log")),
        })
    return artifacts


def _cloud_metrics_from_payload(payload: dict[str, Any]) -> dict[str, float | int]:
    text = "\n".join(
        str(payload.get(key) or "")
        for key in ("message", "log_tail", "logTail", "error")
    )
    metrics: dict[str, float | int] = {}
    for key, raw_value in re.findall(
        r"['\"]([^'\"]+)['\"]\s*:\s*(?:array\()?([+-]?\d+(?:\.\d*)?(?:[eE][+-]?\d+)?)",
        text,
    ):
        try:
            value = float(raw_value)
        except ValueError:
            continue
        metrics[key] = int(value) if value.is_integer() else value
    return metrics


def _attach_cloud_artifacts(payload: dict[str, Any]) -> dict[str, Any]:
    artifacts = _cloud_artifacts_from_payload(payload)
    metrics = _cloud_metrics_from_payload(payload)
    if not artifacts and not metrics:
        return payload
    result = dict(payload)
    if artifacts:
        result["artifacts"] = artifacts
    if metrics:
        result["metrics"] = metrics
        result["metricsSource"] = "log_tail"
    for artifact in artifacts:
        if artifact.get("kind") == "metrics":
            result["metricsPath"] = artifact.get("path")
            break
    return result
