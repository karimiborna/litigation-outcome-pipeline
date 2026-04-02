"""
Simple SF small claims scraper.
Downloads up to MAX_PDFS PDFs for a given date and extracts text via NVIDIA API.

Usage:
    export SFTC_SESSION_ID=<your session id from browser>
    export NVIDIA_API_KEY=<your key>
    python scrape.py --date 2026-04-01

Session ID: visit https://webapps.sftc.org/cc/CaseCalendar.dll in your browser,
then copy the SessionID from the URL.
"""

import argparse
import json
import os
import re
import time
from datetime import date
from pathlib import Path

import requests

from config import (
    BASE_URL, CALENDAR_PATH, CASE_PATH,
    SMALL_CLAIMS_TYPE, USER_AGENT, REQUEST_DELAY_SECS,
)
from nvidia_extractor import extract_text

MAX_PDFS = 5

RAW_DIR = Path(__file__).parent.parent / "data" / "raw" / "pdfs"
PROCESSED_DIR = Path(__file__).parent.parent / "data" / "processed" / "extracted"


class SessionExpiredError(Exception):
    pass


def get_session_id() -> str:
    sid = os.environ.get("SFTC_SESSION_ID", "").strip()
    if not sid:
        raise SystemExit(
            "ERROR: Set SFTC_SESSION_ID env var.\n"
            "Get it by visiting the court calendar in your browser and copying "
            "the SessionID from the URL."
        )
    return sid


def get_cases(session_id: str, date_str: str) -> list:
    url = (
        f"{BASE_URL}{CALENDAR_PATH}"
        f"/datasnap/rest/TServerMethods1/GetCases2"
        f"/{date_str}/{SMALL_CLAIMS_TYPE}/{session_id}"
    )
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data["result"][0] == -1:
        raise SessionExpiredError(
            "Session expired. Get a fresh SessionID from your browser."
        )
    if data["result"][0] == 0:
        return []

    return json.loads(data["result"][1])


def parse_case_num(case_num_html: str) -> str:
    """Extract bare case number (e.g. CSM26871146) from the HTML anchor tag."""
    match = re.search(r"CaseNum=(\w+)&", case_num_html)
    return match.group(1) if match else None


def get_documents(case_num: str, session_id: str) -> list:
    url = (
        f"{BASE_URL}{CASE_PATH}"
        f"/datasnap/rest/TServerMethods1/GetDocuments"
        f"/{case_num}/{session_id}/"
    )
    resp = requests.get(url, headers={"User-Agent": USER_AGENT}, timeout=30)
    resp.raise_for_status()
    data = resp.json()

    if data["result"][0] == -1:
        raise SessionExpiredError("Session expired.")
    if data["result"][0] == 0:
        return []

    return json.loads(data["result"][1])


def download_pdf(doc_url: str, dest: Path, session: requests.Session) -> bool:
    """Download a PDF to dest. Returns True on success."""
    try:
        resp = session.get(doc_url, timeout=60, stream=True)
        resp.raise_for_status()

        content_type = resp.headers.get("content-type", "")
        if "pdf" not in content_type.lower():
            print(f"  Skipping non-PDF response ({content_type})")
            return False

        dest.parent.mkdir(parents=True, exist_ok=True)
        with open(dest, "wb") as f:
            for chunk in resp.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"  Downloaded: {dest.name} ({dest.stat().st_size // 1024} KB)")
        return True

    except Exception as e:
        print(f"  Download failed: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Scrape SF small claims PDFs")
    parser.add_argument(
        "--date",
        default=date.today().isoformat(),
        help="Court date to scrape (YYYY-MM-DD). Defaults to today.",
    )
    args = parser.parse_args()

    session_id = get_session_id()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

    http = requests.Session()
    http.headers.update({"User-Agent": USER_AGENT})

    print(f"\nFetching small claims cases for {args.date} ...")
    cases = get_cases(session_id, args.date)
    print(f"Found {len(cases)} cases on calendar.")

    if not cases:
        print("No cases found. Try a different date (weekdays only).")
        return

    downloaded = 0

    for case in cases:
        if downloaded >= MAX_PDFS:
            break

        case_num = parse_case_num(case.get("CASE_NUMBER", ""))
        if not case_num:
            continue

        title = case.get("CASETITLE", "Unknown")
        print(f"\nCase {case_num}: {title}")

        time.sleep(REQUEST_DELAY_SECS)

        docs = get_documents(case_num, session_id)
        if not docs:
            print("  No documents found.")
            continue

        print(f"  {len(docs)} document(s) available.")

        for doc in docs:
            if downloaded >= MAX_PDFS:
                break

            desc = doc.get("DESCRIPTION", "doc")
            doc_url = doc.get("URL", "")
            if not doc_url:
                continue

            # Filename: <case_num>_<sanitized_desc>.pdf
            safe_desc = re.sub(r"[^\w\-]", "_", desc)[:40]
            pdf_path = RAW_DIR / f"{case_num}_{safe_desc}.pdf"

            if pdf_path.exists():
                print(f"  Already exists, skipping: {pdf_path.name}")
                downloaded += 1
                continue

            print(f"  Downloading '{desc}' ...")
            time.sleep(REQUEST_DELAY_SECS)

            if download_pdf(doc_url, pdf_path, http):
                downloaded += 1

                # Extract text if NVIDIA key is available
                if os.environ.get("NVIDIA_API_KEY"):
                    txt_path = PROCESSED_DIR / f"{pdf_path.stem}.txt"
                    print(f"  Extracting text via NVIDIA API ...")
                    text = extract_text(pdf_path)
                    txt_path.write_text(text, encoding="utf-8")
                    print(f"  Saved: {txt_path.name}")
                else:
                    print(f"  Skipping extraction (no NVIDIA_API_KEY set).")

    print(f"\nDone. Downloaded {downloaded} PDF(s).")
    print(f"  Raw PDFs : {RAW_DIR}")
    print(f"  Extracted: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
