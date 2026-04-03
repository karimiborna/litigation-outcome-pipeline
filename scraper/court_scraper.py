"""Main orchestrator — scrapes cases, downloads PDFs, extracts text.

Uses Ernie's reverse-engineered DataSnap REST API rather than HTML scraping.
Synchronous (requests + time.sleep) since we rate limit to 2.5s anyway.
"""

from __future__ import annotations

import logging
from datetime import date, timedelta
from pathlib import Path

import requests

from scraper.config import ScraperConfig
from scraper.court_api import (
    download_pdf,
    get_cases,
    get_documents,
    parse_case_number,
    sanitize_description,
)
from scraper.extractor import extract_text
from scraper.manifest import ScrapeManifest, save_manifest
from scraper.rate_limiter import RateLimiter
from scraper.session import SessionExpiredError, get_session_id, prompt_refresh

logger = logging.getLogger(__name__)


class CourtScraper:
    """End-to-end scraper for SF Superior Court small claims cases."""

    def __init__(
        self,
        config: ScraperConfig,
        manifest: ScrapeManifest,
        manifest_path: Path,
    ):
        self._config = config
        self._manifest = manifest
        self._manifest_path = manifest_path
        self._rate_limiter = RateLimiter(
            min_delay=config.rate_limit_seconds,
            max_daily=config.max_daily_requests,
        )
        self._session_id = get_session_id(config)
        self._http = requests.Session()
        self._http.headers.update({"User-Agent": config.user_agent})

    def scrape_date_range(
        self,
        start_date: date,
        end_date: date,
        extract: bool = True,
    ) -> dict:
        """Scrape all small claims cases between start_date and end_date.

        Returns a summary dict with counts.
        """
        stats = {
            "dates_processed": 0,
            "cases_scraped": 0,
            "pdfs_downloaded": 0,
            "errors": 0,
        }
        current = start_date

        while current <= end_date:
            if current.weekday() >= 5:
                current += timedelta(days=1)
                continue

            if self._manifest.is_date_completed(current):
                logger.info("Skipping %s (already completed)", current)
                current += timedelta(days=1)
                continue

            try:
                day_stats = self._scrape_date(current, extract)
                stats["dates_processed"] += 1
                stats["cases_scraped"] += day_stats["cases"]
                stats["pdfs_downloaded"] += day_stats["pdfs"]
            except RuntimeError as e:
                if "Daily request cap" in str(e):
                    logger.warning("Daily cap hit, stopping. Resume tomorrow.")
                    break
                raise
            except SessionExpiredError:
                self._session_id = prompt_refresh()
                continue
            except Exception:
                logger.exception("Failed to scrape date %s", current)
                stats["errors"] += 1

            save_manifest(self._manifest, self._manifest_path)
            current += timedelta(days=1)

        save_manifest(self._manifest, self._manifest_path)
        return stats

    def _scrape_date(self, court_date: date, extract: bool) -> dict:
        """Fetch and scrape all cases for a single court date."""
        logger.info("Scraping cases for %s", court_date)
        stats = {"cases": 0, "pdfs": 0}

        self._rate_limiter.wait()
        date_str = court_date.strftime("%Y-%m-%d")
        cases = get_cases(self._session_id, date_str, self._config)

        self._manifest.mark_date_searched(court_date, len(cases))
        logger.info("Found %d cases for %s", len(cases), court_date)

        if not cases:
            self._manifest.mark_date_completed(court_date)
            return stats

        for case in cases:
            case_num = parse_case_number(case.get("CASE_NUMBER", ""))
            if not case_num:
                continue

            if self._manifest.is_case_scraped(case_num):
                logger.debug("Skipping %s (already scraped)", case_num)
                continue

            max_pdfs = self._config.max_pdfs_per_run
            if max_pdfs > 0 and stats["pdfs"] >= max_pdfs:
                logger.info("Hit max PDFs per run (%d), stopping.", max_pdfs)
                break

            try:
                pdf_count = self._scrape_case(case, case_num, court_date, extract)
                stats["cases"] += 1
                stats["pdfs"] += pdf_count
            except SessionExpiredError:
                raise
            except Exception:
                logger.exception("Failed to scrape case %s", case_num)

        self._manifest.mark_date_completed(court_date)
        return stats

    def _scrape_case(
        self,
        case: dict,
        case_num: str,
        court_date: date,
        extract: bool,
    ) -> int:
        """Scrape a single case: fetch docs, download PDFs, extract text."""
        title = case.get("CASETITLE", "Unknown")
        logger.info("Case %s: %s", case_num, title)

        self._rate_limiter.wait()
        docs = get_documents(case_num, self._session_id, self._config)

        if not docs:
            logger.info("  No documents found for %s", case_num)
            self._manifest.mark_case_scraped(case_num, title, court_date, 0)
            return 0

        logger.info("  %d document(s) available", len(docs))

        pdf_dir = self._config.raw_dir / "pdfs"
        txt_dir = self._config.processed_dir / "extracted"
        pdf_dir.mkdir(parents=True, exist_ok=True)
        txt_dir.mkdir(parents=True, exist_ok=True)

        pdf_count = 0
        for doc in docs:
            desc = doc.get("DESCRIPTION", "doc")
            doc_url = doc.get("URL", "")
            if not doc_url:
                continue

            safe_desc = sanitize_description(desc)
            pdf_path = pdf_dir / f"{case_num}_{safe_desc}.pdf"

            if pdf_path.exists():
                logger.info("  Already exists, skipping: %s", pdf_path.name)
                pdf_count += 1
                continue

            self._rate_limiter.wait()
            logger.info("  Downloading '%s' ...", desc)

            if download_pdf(doc_url, pdf_path, self._http, self._config.pdf_download_timeout):
                pdf_count += 1

                if extract:
                    txt_path = txt_dir / f"{pdf_path.stem}.txt"
                    if not txt_path.exists():
                        text = extract_text(pdf_path, self._config.nvidia_api_key)
                        if text:
                            txt_path.write_text(text, encoding="utf-8")
                            logger.info("  Saved extracted text: %s", txt_path.name)

        self._manifest.mark_case_scraped(case_num, title, court_date, pdf_count)
        if extract and pdf_count > 0:
            self._manifest.mark_case_extracted(case_num, pdf_count)

        return pdf_count


def build_date_range(days_back: int = 120) -> tuple[date, date]:
    """Return (start_date, end_date) covering the last N days of court data."""
    end = date.today()
    start = end - timedelta(days=days_back)
    return start, end
