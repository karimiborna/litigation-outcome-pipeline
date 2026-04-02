"""Scrape manifest for resume support and progress tracking.

Tracks which dates have been searched and which cases have been scraped,
so the scraper never re-scrapes or re-extracts.
"""

from __future__ import annotations

import json
import logging
from datetime import date, datetime
from pathlib import Path

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

DEFAULT_MANIFEST_PATH = Path("scraper/state/manifest.json")


class CaseRecord(BaseModel):
    case_number: str
    case_title: str = ""
    filing_date: date
    metadata_saved: bool = False
    pdfs_downloaded: int = 0
    pdfs_extracted: int = 0
    scraped_at: datetime | None = None
    extraction_completed: bool = False


class DateRecord(BaseModel):
    filing_date: date
    total_cases_found: int = 0
    cases_scraped: int = 0
    completed: bool = False
    searched_at: datetime | None = None


class ScrapeManifest(BaseModel):
    """Persistent manifest tracking all scraping progress."""

    dates: dict[str, DateRecord] = Field(default_factory=dict)
    cases: dict[str, CaseRecord] = Field(default_factory=dict)
    last_updated: datetime = Field(default_factory=datetime.now)

    def is_date_completed(self, filing_date: date) -> bool:
        key = filing_date.isoformat()
        record = self.dates.get(key)
        return record is not None and record.completed

    def is_case_scraped(self, case_number: str) -> bool:
        record = self.cases.get(case_number)
        return record is not None and record.metadata_saved

    def is_case_extracted(self, case_number: str) -> bool:
        record = self.cases.get(case_number)
        return record is not None and record.extraction_completed

    def mark_date_searched(self, filing_date: date, total_cases: int) -> None:
        key = filing_date.isoformat()
        self.dates[key] = DateRecord(
            filing_date=filing_date,
            total_cases_found=total_cases,
            searched_at=datetime.now(),
        )

    def mark_date_completed(self, filing_date: date) -> None:
        key = filing_date.isoformat()
        if key in self.dates:
            self.dates[key].completed = True
            self.dates[key].cases_scraped = self.dates[key].total_cases_found

    def mark_case_scraped(
        self,
        case_number: str,
        case_title: str,
        filing_date: date,
        pdfs_downloaded: int,
    ) -> None:
        self.cases[case_number] = CaseRecord(
            case_number=case_number,
            case_title=case_title,
            filing_date=filing_date,
            metadata_saved=True,
            pdfs_downloaded=pdfs_downloaded,
            scraped_at=datetime.now(),
        )
        date_key = filing_date.isoformat()
        if date_key in self.dates:
            self.dates[date_key].cases_scraped += 1

    def mark_case_extracted(self, case_number: str, pdfs_extracted: int) -> None:
        if case_number in self.cases:
            self.cases[case_number].pdfs_extracted = pdfs_extracted
            self.cases[case_number].extraction_completed = True

    def summary(self) -> dict:
        total_dates = len(self.dates)
        completed_dates = sum(1 for d in self.dates.values() if d.completed)
        total_cases = len(self.cases)
        extracted = sum(1 for c in self.cases.values() if c.extraction_completed)
        return {
            "dates_searched": total_dates,
            "dates_completed": completed_dates,
            "cases_scraped": total_cases,
            "cases_extracted": extracted,
        }


def load_manifest(path: Path = DEFAULT_MANIFEST_PATH) -> ScrapeManifest:
    """Load manifest from disk, or return a fresh one if not found."""
    if path.exists():
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            return ScrapeManifest.model_validate(raw)
        except Exception:
            logger.exception("Failed to load manifest from %s, starting fresh", path)
    return ScrapeManifest()


def save_manifest(manifest: ScrapeManifest, path: Path = DEFAULT_MANIFEST_PATH) -> None:
    """Persist the manifest to disk."""
    path.parent.mkdir(parents=True, exist_ok=True)
    manifest.last_updated = datetime.now()
    path.write_text(manifest.model_dump_json(indent=2), encoding="utf-8")
    logger.debug("Manifest saved to %s", path)
