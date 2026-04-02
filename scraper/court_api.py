"""Client for the SF Superior Court DataSnap REST API.

The court runs a Delphi DataSnap backend that exposes JSON endpoints
behind CaseCalendar.dll and CaseInfo.dll. These were reverse-engineered
from the site's lookup.js and cslookup.js scripts.
"""

from __future__ import annotations

import json
import logging
import re

import requests

from scraper.config import (
    BASE_URL,
    CALENDAR_PATH,
    CASE_PATH,
    SMALL_CLAIMS_TYPE,
    ScraperConfig,
)
from scraper.session import SessionExpiredError

logger = logging.getLogger(__name__)


def get_cases(session_id: str, date_str: str, config: ScraperConfig) -> list[dict]:
    """Fetch all small claims cases for a given court date.

    Calls: GET /cc/CaseCalendar.dll/datasnap/rest/TServerMethods1/GetCases2/{date}/{type}/{sid}

    Returns a list of case dicts with keys: CASE_NUMBER, CASETITLE,
    COURTDATE, COURT_TIME, ROOM, LOCATIONMAPPED.

    Note: CASE_NUMBER contains raw HTML with an anchor tag — use
    parse_case_number() to extract the bare case number.
    """
    url = (
        f"{BASE_URL}{CALENDAR_PATH}"
        f"/datasnap/rest/TServerMethods1/GetCases2"
        f"/{date_str}/{SMALL_CLAIMS_TYPE}/{session_id}"
    )
    resp = requests.get(
        url,
        headers={"User-Agent": config.user_agent},
        timeout=config.request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    if data["result"][0] == -1:
        raise SessionExpiredError(
            "Session expired. Get a fresh SessionID from your browser."
        )
    if data["result"][0] == 0:
        return []

    return json.loads(data["result"][1])


def get_documents(case_num: str, session_id: str, config: ScraperConfig) -> list[dict]:
    """Fetch the document list for a specific case.

    Calls: GET /ci/CaseInfo.dll/datasnap/rest/TServerMethods1/GetDocuments/{case}/{sid}/

    Returns a list of document dicts with keys: FILEDATE, DESCRIPTION, URL.
    The URL is a time-limited signed link (~10 min expiry) that proxies
    through CaseInfo.dll to imgquery.sftc.org.
    """
    url = (
        f"{BASE_URL}{CASE_PATH}"
        f"/datasnap/rest/TServerMethods1/GetDocuments"
        f"/{case_num}/{session_id}/"
    )
    resp = requests.get(
        url,
        headers={"User-Agent": config.user_agent},
        timeout=config.request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    if data["result"][0] == -1:
        raise SessionExpiredError("Session expired.")
    if data["result"][0] == 0:
        return []

    return json.loads(data["result"][1])


def get_roa(case_num: str, session_id: str, config: ScraperConfig) -> list[dict]:
    """Fetch the Register of Actions for a case (lightweight existence check).

    Calls: GET /ci/CaseInfo.dll/datasnap/rest/TServerMethods1/GetROA/{case}/{sid}/

    Returns a list of ROA entry dicts if the case exists, empty list otherwise.
    Cheaper than GetDocuments for probing since it doesn't generate signed URLs.
    """
    url = (
        f"{BASE_URL}{CASE_PATH}"
        f"/datasnap/rest/TServerMethods1/GetROA"
        f"/{case_num}/{session_id}/"
    )
    resp = requests.get(
        url,
        headers={"User-Agent": config.user_agent},
        timeout=config.request_timeout,
    )
    resp.raise_for_status()
    data = resp.json()

    if data["result"][0] == -1:
        raise SessionExpiredError("Session expired.")
    if data["result"][0] == 0:
        return []

    return json.loads(data["result"][1])


def probe_case_exists(
    case_num: str, session_id: str, config: ScraperConfig
) -> int:
    """Check if a case number exists. Returns document count (0 = not found).

    Tries GetROA first (lightweight). Falls back to GetDocuments if ROA
    doesn't differentiate between "no case" and "no ROA entries".
    """
    try:
        roa = get_roa(case_num, session_id, config)
        if roa:
            return len(roa)
        docs = get_documents(case_num, session_id, config)
        return len(docs)
    except SessionExpiredError:
        raise
    except Exception:
        logger.debug("Probe failed for %s", case_num)
        return 0


def parse_case_number(case_num_html: str) -> str | None:
    """Extract the bare case number (e.g. CSM26871146) from the HTML anchor tag.

    The API returns CASE_NUMBER as raw HTML like:
    <A HREF="...?CaseNum=CSM26871146&SessionID=...">CSM-26-871146</A>
    """
    match = re.search(r"CaseNum=(\w+)&", case_num_html)
    return match.group(1) if match else None


def sanitize_description(desc: str, max_len: int = 40) -> str:
    """Convert a document description into a safe filename component."""
    return re.sub(r"[^\w\-]", "_", desc)[:max_len]


def download_pdf(
    doc_url: str,
    dest_path: str | object,
    session: requests.Session,
    timeout: float = 60.0,
) -> bool:
    """Download a PDF from a signed court URL. Returns True on success."""
    try:
        resp = session.get(str(doc_url), timeout=timeout, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            logger.warning("Skipping non-PDF response (%s)", content_type)
            return False

        from pathlib import Path

        path = Path(dest_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        logger.info("Downloaded: %s (%d KB)", path.name, path.stat().st_size // 1024)
        return True

    except Exception:
        logger.exception("Download failed for %s", doc_url)
        return False
