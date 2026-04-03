"""
Standalone scraper — enumerate case numbers then download their PDFs.

Usage:
    # Enumerate case numbers starting where Borna left off
    python scraper/scrape.py --enumerate --start CSM25870100 --end CSM25879999

    # Download all found cases (no text extraction)
    python scraper/scrape.py --download --no-extract

    # Enumerate then immediately download
    python scraper/scrape.py --enumerate --start CSM25870100 --end CSM25879999 --download

Session ID: set SFTC_SESSION_ID in .env, or you'll be prompted interactively.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env", override=True)

from scraper.config import ScraperConfig
from scraper.court_api import download_pdf, get_documents, sanitize_description
from scraper.enumerator import CaseEnumerator, ValidCasesStore, parse_case_range
from scraper.extractor import extract_text
from scraper.manifest import load_manifest, save_manifest
from scraper.rate_limiter import RateLimiter
from scraper.session import SessionExpiredError, get_session_id, prompt_refresh
from scraper.session_manager import start_keepalive

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

MANIFEST_PATH = Path("scraper/state/manifest.json")


def run_enumerate(start: str, end: str, config: ScraperConfig, session_id: str) -> None:
    """Probe a range of case numbers and save valid ones to valid_cases.json."""
    case_numbers = parse_case_range(start, end)
    logger.info("Enumerating %d case numbers: %s → %s", len(case_numbers), start, end)

    store = ValidCasesStore()
    enumerator = CaseEnumerator(config, session_id, store, probe_delay=1.0)
    stats = enumerator.enumerate(case_numbers)

    logger.info(
        "Done. Probed: %d | Found: %d | Total valid: %d",
        stats["probed"], stats["found"], store.valid_count,
    )


def run_download(config: ScraperConfig, session_id: str, extract: bool) -> None:
    """Download PDFs for all valid cases in valid_cases.json."""
    import requests as req

    store = ValidCasesStore()
    valid = store.valid_cases

    if not valid:
        logger.error("No valid cases found. Run --enumerate first.")
        sys.exit(1)

    manifest = load_manifest(MANIFEST_PATH)
    rate_limiter = RateLimiter(
        min_delay=config.rate_limit_seconds,
        max_daily=config.max_daily_requests,
    )

    http = req.Session()
    http.headers.update({"User-Agent": config.user_agent})

    logger.info("Downloading PDFs for %d cases ...", len(valid))
    total_pdfs = 0
    cases_done = 0

    for case_num in sorted(valid.keys()):
        if manifest.is_case_scraped(case_num):
            logger.info("  %s: already done, skipping", case_num)
            cases_done += 1
            continue

        try:
            rate_limiter.wait()
            docs = get_documents(case_num, session_id, config)
        except SessionExpiredError:
            session_id = prompt_refresh()
            start_keepalive(session_id)
            rate_limiter.wait()
            docs = get_documents(case_num, session_id, config)

        case_raw_dir = config.raw_dir / case_num
        case_proc_dir = config.processed_dir / case_num
        case_raw_dir.mkdir(parents=True, exist_ok=True)
        case_proc_dir.mkdir(parents=True, exist_ok=True)

        if not docs:
            logger.info("  %s: no documents", case_num)
            manifest.mark_case_scraped(case_num, "", date.today(), 0)
            cases_done += 1
            continue

        logger.info("  %s: %d document(s)", case_num, len(docs))
        pdf_count = 0

        for doc in docs:
            desc = doc.get("DESCRIPTION", "doc")
            doc_url = doc.get("URL", "")
            if not doc_url:
                continue

            safe_desc = sanitize_description(desc)
            pdf_path = case_raw_dir / f"{safe_desc}.pdf"

            if pdf_path.exists():
                pdf_count += 1
                continue

            logger.info("    Downloading '%s' ...", desc)
            rate_limiter.wait()

            if download_pdf(doc_url, pdf_path, http, config.pdf_download_timeout):
                pdf_count += 1

                if extract and config.nvidia_api_key:
                    txt_path = case_proc_dir / f"{safe_desc}.txt"
                    if not txt_path.exists():
                        text = extract_text(pdf_path, config.nvidia_api_key)
                        if text:
                            txt_path.write_text(text, encoding="utf-8")
                            logger.info("    Extracted: %s", txt_path.name)

        manifest.mark_case_scraped(case_num, "", date.today(), pdf_count)
        if extract and pdf_count > 0:
            manifest.mark_case_extracted(case_num, pdf_count)

        total_pdfs += pdf_count
        cases_done += 1

        if cases_done % 10 == 0:
            save_manifest(manifest, MANIFEST_PATH)
            logger.info("Progress: %d/%d cases, %d PDFs", cases_done, len(valid), total_pdfs)

    save_manifest(manifest, MANIFEST_PATH)
    logger.info("Done. Cases: %d | PDFs: %d", cases_done, total_pdfs)


def main() -> None:
    parser = argparse.ArgumentParser(description="SF Small Claims scraper — case number mode")
    parser.add_argument("--enumerate", action="store_true", help="Probe case number range")
    parser.add_argument("--start", default="CSM25870100", help="Start case number (default: CSM25870100)")
    parser.add_argument("--end", default="CSM25879999", help="End case number (default: CSM25879999)")
    parser.add_argument("--download", action="store_true", help="Download PDFs for all valid cases")
    parser.add_argument("--no-extract", action="store_true", help="Skip text extraction")
    args = parser.parse_args()

    if not args.enumerate and not args.download:
        parser.print_help()
        sys.exit(1)

    config = ScraperConfig()
    session_id = get_session_id(config)
    start_keepalive(session_id)

    try:
        if args.enumerate:
            run_enumerate(args.start, args.end, config, session_id)

        if args.download:
            run_download(config, session_id, extract=not args.no_extract)

    except KeyboardInterrupt:
        logger.info("\nInterrupted. Progress is saved — re-run to resume.")
        sys.exit(0)


if __name__ == "__main__":
    main()
