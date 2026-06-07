"""Dataset CLI commands."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import httpx
import typer
from rich.console import Console

from roboclaw.data.dataset_push import (
    DatasetPushSummary,
    dataset_handle,
    format_bytes,
    push_dataset_to_local_catalog,
    scan_dataset_path,
)

dataset_app = typer.Typer(
    name="dataset",
    help="Collect, register, and share robot datasets.",
    no_args_is_help=True,
)
console = Console()


@dataset_app.command("push")
def push_dataset(
    path: Path = typer.Argument(..., help="Local robot dataset directory to push."),
    dataset_id: str = typer.Option("", "--dataset-id", "-d", help="Dataset id to register."),
    username: str = typer.Option("", "--username", "-u", help="Owner username. Defaults to EVOMIND_USER or ROBOCLAW_USERNAME."),
    visibility: str = typer.Option("private", "--visibility", help="private or public."),
    server: str = typer.Option("", "--server", help="Optional RoboClaw HTTP server, for example http://127.0.0.1:8766."),
    force: bool = typer.Option(False, "--force", help="Overwrite an existing local dataset target."),
    dry_run: bool = typer.Option(False, "--dry-run", help="Only scan and print the push plan."),
) -> None:
    """Push a local robot dataset into Evo Studio's data catalog."""
    owner = _resolve_username(username)
    if not owner:
        raise typer.BadParameter(
            "username is required until EvoMind session auth is enabled; pass --username or set EVOMIND_USER."
        )

    try:
        summary = scan_dataset_path(path, dataset_id=dataset_id)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from exc

    _print_summary(summary, owner)
    if dry_run:
        console.print("[yellow]Dry run only. No dataset was registered.[/yellow]")
        return

    target_server = server.strip() or os.environ.get("ROBOCLAW_SERVER_URL", "").strip()
    if target_server:
        payload = _push_via_server(summary, owner=owner, visibility=visibility, server=target_server, force=force)
        _print_server_result(payload, summary, owner)
        return

    try:
        ref = push_dataset_to_local_catalog(
            summary,
            username=owner,
            visibility=visibility,
            force=force,
        )
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from exc

    console.print("[green]Dataset registered in the local Evo Studio catalog.[/green]")
    console.print(f"dataset: [bold]{dataset_handle(owner, ref.id)}[/bold]")
    console.print("credits: pending quality review")
    console.print(f"next: roboclaw train \"use {dataset_handle(owner, ref.id)} for my robot task\"")


def _resolve_username(username: str) -> str:
    return (
        username.strip()
        or os.environ.get("EVOMIND_USER", "").strip()
        or os.environ.get("ROBOCLAW_USERNAME", "").strip()
    )


def _print_summary(summary: DatasetPushSummary, owner: str) -> None:
    console.print("[bold]Dataset push plan[/bold]")
    console.print(f"owner: {owner}")
    console.print(f"dataset: {summary.dataset_id}")
    console.print(f"source: {summary.source_path}")
    console.print(f"files: {summary.file_count}")
    console.print(f"size: {format_bytes(summary.total_bytes)}")
    console.print(f"episodes: {summary.total_episodes or 'unknown'}")
    if summary.total_frames:
        console.print(f"frames: {summary.total_frames}")
    if summary.robot_type:
        console.print(f"robot: {summary.robot_type}")
    if not summary.has_manifest:
        console.print("[yellow]meta/info.json not found; RoboClaw will create minimal catalog metadata.[/yellow]")


def _push_via_server(
    summary: DatasetPushSummary,
    *,
    owner: str,
    visibility: str,
    server: str,
    force: bool,
) -> dict[str, Any]:
    url = server.rstrip("/") + "/api/datasets/ingest"
    request_payload = {
        "dataset_id": summary.dataset_id,
        "username": owner,
        "owner_username": owner,
        "contribution_source": "self_collected",
        "source_kind": "mounted_path",
        "source_uri": str(summary.source_path),
        "source_auth_ref": "local",
        "storage_mode": "managed",
        "include_videos": True,
        "force": force,
    }
    try:
        response = httpx.post(
            url,
            json=request_payload,
            headers={"x-evo-studio-user": owner},
            timeout=60.0,
        )
        response.raise_for_status()
        payload = response.json()
    except httpx.HTTPStatusError as exc:
        detail = _response_detail(exc.response)
        hint = _server_ingest_hint(detail, summary.source_path)
        raise typer.BadParameter(f"dataset push failed: {detail}{hint}") from exc
    except (httpx.HTTPError, ValueError) as exc:
        raise typer.BadParameter(f"dataset push failed: {exc}") from exc
    if not isinstance(payload, dict):
        raise typer.BadParameter("dataset push failed: server returned a non-object response")
    return payload


def _response_detail(response: httpx.Response) -> str:
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip() or response.reason_phrase
    detail = payload.get("detail") if isinstance(payload, dict) else payload
    if isinstance(detail, dict):
        return str(detail.get("message") or detail)
    return str(detail or payload)


def _server_ingest_hint(detail: str, source_path: Path) -> str:
    if "ROBOCLAW_DATASET_INGEST_ROOTS" not in detail:
        return ""
    return (
        "\nHint: allow the server to read this parent directory, then restart the web service:\n"
        f"  export ROBOCLAW_DATASET_INGEST_ROOTS={source_path.parent}"
    )


def _print_server_result(payload: dict[str, Any], summary: DatasetPushSummary, owner: str) -> None:
    status = str(payload.get("status") or "registered")
    dataset = payload.get("dataset") if isinstance(payload.get("dataset"), dict) else {}
    dataset_id = str(dataset.get("id") or summary.dataset_id)
    quality = payload.get("quality") if isinstance(payload.get("quality"), dict) else {}
    console.print(f"[green]Dataset {status} by Evo Studio.[/green]")
    console.print(f"dataset: [bold]{dataset_handle(owner, dataset_id)}[/bold]")
    console.print(f"quality: {quality.get('status', 'pending')}")
    console.print("credits: pending quality review")
    console.print(f"next: roboclaw train \"use {dataset_handle(owner, dataset_id)} for my robot task\"")
