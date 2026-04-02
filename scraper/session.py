"""Session management for the SF Superior Court website.

Prompts the user to paste a session ID in the terminal if one isn't
provided via SFTC_SESSION_ID env var. Also prompts again on expiry.
"""

from __future__ import annotations

import logging

from rich.console import Console
from rich.prompt import Prompt

from scraper.config import ScraperConfig

logger = logging.getLogger(__name__)
console = Console()

PLACEHOLDERS = {"", "your-session-id-here", "your_session_id_here"}

INSTRUCTIONS = (
    "[bold]How to get a session ID:[/bold]\n"
    "  1. Open [link=https://webapps.sftc.org/cc/CaseCalendar.dll]"
    "https://webapps.sftc.org/cc/CaseCalendar.dll[/link] in your browser\n"
    "  2. Copy the hex string after SessionID= from the URL"
)


class SessionExpiredError(Exception):
    """Raised when the court site returns -1 (session expired)."""


def get_session_id(config: ScraperConfig) -> str:
    """Get a session ID from env var or prompt the user interactively."""
    sid = config.session_id.strip()
    if sid and sid.lower() not in PLACEHOLDERS:
        logger.info("Using session ID from .env: %s...", sid[:12])
        return sid

    return _prompt_for_session_id()


def prompt_refresh() -> str:
    """Prompt the user for a fresh session ID after expiry."""
    console.print("\n[bold red]Session expired.[/bold red]")
    return _prompt_for_session_id()


def _prompt_for_session_id() -> str:
    """Show instructions and prompt for a session ID."""
    console.print(f"\n{INSTRUCTIONS}\n")
    while True:
        sid = Prompt.ask("[bold]Paste your SessionID[/bold]").strip()
        if sid and sid.lower() not in PLACEHOLDERS:
            logger.info("Using session ID: %s...", sid[:12])
            return sid
        console.print("[red]That doesn't look like a valid session ID. Try again.[/red]")
