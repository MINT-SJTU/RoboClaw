"""Developer CLI utilities — ``roboclaw dev …``."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console

dev_app = typer.Typer(no_args_is_help=True)
_console = Console()


@dev_app.command()
def reset(
    model: Optional[str] = typer.Option(None, help="Override the default model in config.json."),
    provider: Optional[str] = typer.Option(None, help="Provider name to set in config.json."),
    api_base: Optional[str] = typer.Option(None, help="API base URL to set."),
    api_key: Optional[str] = typer.Option(None, help="API key to set."),
    keep_calibration: bool = typer.Option(False, "--keep-calibration", help="Preserve calibration files."),
    keep_setup: bool = typer.Option(False, "--keep-setup", help="Preserve manifest.json (arms, cameras)."),
    workspace_only: bool = typer.Option(False, "--workspace-only", help="Only reset workspace, keep config.json."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Skip confirmation prompt."),
) -> None:
    """Delete workspace and re-run onboard non-interactively.

    Useful for returning to a clean state during development.
    Optionally patches config.json with --model / --provider / --api-base / --api-key.
    Use --keep-calibration and/or --keep-setup to preserve hardware state.
    """
    from roboclaw.config.loader import get_config_path
    from roboclaw.config.paths import get_workspace_path

    workspace = get_workspace_path()
    config_path = get_config_path()

    if not yes:
        _console.print(f"[yellow]This will delete:[/yellow]")
        _console.print(f"  - {workspace}")
        if not workspace_only:
            _console.print(f"  - {config_path}")
        if keep_calibration:
            _console.print(f"  [dim](keeping calibration files)[/dim]")
        if keep_setup:
            _console.print(f"  [dim](keeping manifest.json)[/dim]")
        if workspace_only:
            _console.print(f"  [dim](keeping config.json)[/dim]")
        if not typer.confirm("Continue?"):
            raise typer.Abort()

    # 1. Delete workspace and config, preserving selected files
    if workspace.exists():
        _clean_workspace(workspace, keep_calibration=keep_calibration, keep_setup=keep_setup)
        _console.print(f"[green]✓[/green] Cleaned {workspace}")
    if not workspace_only and config_path.exists():
        config_path.unlink()
        _console.print(f"[green]✓[/green] Deleted {config_path}")

    # 2. Re-run onboard (non-interactive: always creates fresh config)
    from roboclaw.cli.commands import run_onboard_core

    run_onboard_core(interactive=False, skip_config=workspace_only)

    # 3. Patch config.json if model params given
    _patch_config(config_path, model=model, provider=provider, api_base=api_base, api_key=api_key)

    _console.print("\n[green]Dev reset complete.[/green]")


def _clean_workspace(workspace: Path, *, keep_calibration: bool, keep_setup: bool) -> None:
    """Delete workspace contents, optionally preserving calibration and setup."""
    embodied = workspace / "embodied"
    cal_dir = embodied / "calibration"
    setup_file = embodied / "manifest.json"

    # Save files we want to keep
    saved_cal = None
    saved_setup = None
    if keep_calibration and cal_dir.exists():
        import tempfile
        saved_cal = Path(tempfile.mkdtemp()) / "calibration"
        shutil.copytree(cal_dir, saved_cal)
    if keep_setup and setup_file.exists():
        saved_setup = setup_file.read_bytes()

    # Nuke workspace
    shutil.rmtree(workspace)

    # Restore saved files
    if saved_cal:
        cal_dir.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(saved_cal, cal_dir)
        shutil.rmtree(saved_cal.parent)
    if saved_setup is not None:
        setup_file.parent.mkdir(parents=True, exist_ok=True)
        setup_file.write_bytes(saved_setup)


def _patch_config(
    config_path: Path,
    *,
    model: str | None,
    provider: str | None,
    api_base: str | None,
    api_key: str | None,
) -> None:
    """Patch config.json with optional overrides."""
    if not any([model, provider, api_base, api_key]):
        return
    if not config_path.exists():
        return

    data = json.loads(config_path.read_text(encoding="utf-8"))
    agents = data.setdefault("agents", {}).setdefault("defaults", {})

    if model:
        agents["model"] = model
        _console.print(f"  [dim]model → {model}[/dim]")

    provider_name = provider or _infer_provider(model)
    if provider_name and (api_key or api_base):
        providers = data.setdefault("providers", {})
        prov = providers.setdefault(provider_name, {})
        if api_key:
            prov["apiKey"] = api_key
            _console.print(f"  [dim]providers.{provider_name}.apiKey → (set)[/dim]")
        if api_base:
            prov["apiBase"] = api_base
            _console.print(f"  [dim]providers.{provider_name}.apiBase → {api_base}[/dim]")

    config_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _infer_provider(model: str | None) -> str | None:
    """Best-effort provider name from a model string (e.g. 'openai/gpt-4o' → 'openai')."""
    if not model:
        return None
    if "/" in model:
        return model.split("/", 1)[0]
    return None
