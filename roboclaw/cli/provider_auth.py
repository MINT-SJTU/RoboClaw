"""OAuth provider login commands."""

from __future__ import annotations

import asyncio
from collections.abc import Callable

import typer
from rich.console import Console

from roboclaw import __logo__

console = Console()
provider_app = typer.Typer(help="Manage providers")
_LOGIN_HANDLERS: dict[str, Callable[..., None]] = {}


# ============================================================================
# OAuth Login
# ============================================================================





def _register_login(name: str):
    def decorator(fn):
        _LOGIN_HANDLERS[name] = fn
        return fn
    return decorator


def _oauth_print(s: str) -> None:
    """Print with Rich, but use plain print for URLs to avoid wrapping."""
    if s.startswith("http://") or s.startswith("https://"):
        print(s)
    else:
        console.print(s)


@provider_app.command("login")
def provider_login(
    provider: str = typer.Argument(..., help="OAuth provider (e.g. 'openai-codex', 'github-copilot')"),
    force: bool = typer.Option(False, "--force", "-f", help="Force re-authentication even if already logged in"),
):
    """Authenticate with an OAuth provider."""
    from roboclaw.providers.registry import PROVIDERS

    key = provider.replace("-", "_")
    spec = next((s for s in PROVIDERS if s.name == key and s.is_oauth), None)
    if not spec:
        names = ", ".join(s.name.replace("_", "-") for s in PROVIDERS if s.is_oauth)
        console.print(f"[red]Unknown OAuth provider: {provider}[/red]  Supported: {names}")
        raise typer.Exit(1)

    handler = _LOGIN_HANDLERS.get(spec.name)
    if not handler:
        console.print(f"[red]Login not implemented for {spec.label}[/red]")
        raise typer.Exit(1)

    console.print(f"{__logo__} OAuth Login - {spec.label}\n")
    handler(force=force)


@_register_login("openai_codex")
def _login_openai_codex(force: bool = False) -> None:
    try:
        from oauth_cli_kit import get_token, login_oauth_interactive
    except ImportError:
        console.print("[red]oauth_cli_kit not installed. Run: pip install oauth-cli-kit[/red]")
        raise typer.Exit(1)

    if not force:
        try:
            token = get_token()
        except RuntimeError:
            token = None
        if token and token.access:
            console.print(f"[green]✓ Already authenticated[/green]  [dim]{token.account_id}[/dim]")
            console.print("[dim]Use --force to re-authenticate[/dim]")
            return

    console.print("[cyan]Starting interactive OAuth login...[/cyan]\n")
    token = login_oauth_interactive(
        print_fn=_oauth_print,
        prompt_fn=lambda s: typer.prompt(s),
        originator="roboclaw",
    )
    if not (token and token.access):
        console.print("[red]✗ Authentication failed[/red]")
        raise typer.Exit(1)
    console.print(f"[green]✓ Authenticated with OpenAI Codex[/green]  [dim]{token.account_id}[/dim]")


@_register_login("github_copilot")
def _login_github_copilot(force: bool = False) -> None:
    # GitHub Copilot uses device flow via LiteLLM — no local token cache to check
    import asyncio

    console.print("[cyan]Starting GitHub Copilot device flow...[/cyan]\n")

    async def _trigger():
        from litellm import acompletion
        await acompletion(model="github_copilot/gpt-4o", messages=[{"role": "user", "content": "hi"}], max_tokens=1)

    try:
        asyncio.run(_trigger())
        console.print("[green]✓ Authenticated with GitHub Copilot[/green]")
    except Exception as e:
        console.print(f"[red]Authentication error: {e}[/red]")
        raise typer.Exit(1)
