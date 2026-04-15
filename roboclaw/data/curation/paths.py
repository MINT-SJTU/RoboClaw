from __future__ import annotations

from pathlib import Path


def datasets_root() -> Path:
    from roboclaw.embodied.embodiment.manifest.helpers import get_roboclaw_home, get_manifest_path
    import json

    manifest_path = get_manifest_path()
    if manifest_path.exists():
        data = json.loads(manifest_path.read_text(encoding="utf-8"))
        root = data.get("datasets", {}).get("root", "")
        if root:
            return Path(root).expanduser()
    return get_roboclaw_home() / "workspace" / "embodied" / "datasets"
