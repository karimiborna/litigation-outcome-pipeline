"""
Simple session ID management.
Read SFTC_SESSION_ID from .env or environment.

To get a session ID:
1. Visit https://webapps.sftc.org/cc/CaseCalendar.dll in your browser
2. Copy the SessionID= value from the URL
3. Paste it into .env as SFTC_SESSION_ID=<value>
"""

import threading
import time

import requests

from config import BASE_URL, CALENDAR_PATH, SMALL_CLAIMS_TYPE, USER_AGENT

PING_INTERVAL = 2 * 60
COURT_HOME = f"{BASE_URL}{CALENDAR_PATH}"


def start_keepalive(session_id: str):
    """Ping the court API every 2 minutes to prevent session expiry."""
    def loop():
        while True:
            time.sleep(PING_INTERVAL)
            from datetime import date
            url = (
                f"{BASE_URL}{CALENDAR_PATH}"
                f"/datasnap/rest/TServerMethods1/GetCases2"
                f"/{date.today().isoformat()}/{SMALL_CLAIMS_TYPE}/{session_id}"
            )
            try:
                requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=10)
            except Exception:
                pass

    t = threading.Thread(target=loop, daemon=True)
    t.start()
    print(f"Keepalive started (ping every {PING_INTERVAL // 60} min).")
