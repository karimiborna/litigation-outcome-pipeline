"""Keepalive thread — pings the court API every 60s to prevent session expiry."""

import threading
import time
from datetime import date

import requests

from scraper.config import BASE_URL, CALENDAR_PATH, SMALL_CLAIMS_TYPE, ScraperConfig

PING_INTERVAL = 60
_config = ScraperConfig()

# Mutable session holder so the keepalive thread always uses the latest session ID
_session = {"id": ""}


def start_keepalive(session_id: str) -> None:
    """Update the active session ID and ensure the keepalive thread is running."""
    _session["id"] = session_id

    # Only start one thread — subsequent calls just update _session["id"]
    if not hasattr(start_keepalive, "_started"):
        start_keepalive._started = True

        def loop() -> None:
            while True:
                sid = _session["id"]
                if sid:
                    url = (
                        f"{BASE_URL}{CALENDAR_PATH}"
                        f"/datasnap/rest/TServerMethods1/GetCases2"
                        f"/{date.today().isoformat()}/{SMALL_CLAIMS_TYPE}/{sid}"
                    )
                    try:
                        requests.get(
                            url,
                            headers={"User-Agent": _config.user_agent},
                            timeout=10,
                        )
                    except Exception:
                        pass
                time.sleep(PING_INTERVAL)

        t = threading.Thread(target=loop, daemon=True)
        t.start()
        print(f"Keepalive started (ping every {PING_INTERVAL}s).")
    else:
        print("Keepalive updated with new session ID.")
