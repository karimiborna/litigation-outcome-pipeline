"""Local filesystem storage helpers for the data pipeline.

Handles saving/loading case metadata, PDFs, extracted text, and raw HTML.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from data.schemas.case import CaseMetadata, ExtractedText

logger = logging.getLogger(__name__)

DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_PROCESSED_DIR = Path("data/processed")


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def case_raw_dir(case_number: str, base: Path = DEFAULT_RAW_DIR) -> Path:
    return _ensure_dir(base / case_number)


def case_processed_dir(case_number: str, base: Path = DEFAULT_PROCESSED_DIR) -> Path:
    return _ensure_dir(base / case_number)


def save_metadata(metadata: CaseMetadata, base: Path = DEFAULT_RAW_DIR) -> Path:
    """Persist case metadata JSON to data/raw/<case_number>/metadata.json."""
    dest = case_raw_dir(metadata.case_number, base) / "metadata.json"
    dest.write_text(
        metadata.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.debug("Saved metadata for %s to %s", metadata.case_number, dest)
    return dest


def save_pdf(case_number: str, filename: str, content: bytes, base: Path = DEFAULT_RAW_DIR) -> Path:
    """Save a downloaded PDF to data/raw/<case_number>/<filename>."""
    dest = case_raw_dir(case_number, base) / filename
    dest.write_bytes(content)
    logger.debug("Saved PDF %s for case %s", filename, case_number)
    return dest


def save_raw_html(
    case_number: str, page_name: str, html: str, base: Path = DEFAULT_RAW_DIR
) -> Path:
    """Store raw HTML for re-parsing without re-scraping."""
    dest = case_raw_dir(case_number, base) / f"{page_name}.html"
    dest.write_text(html, encoding="utf-8")
    return dest


def save_extracted_text(extracted: ExtractedText, base: Path = DEFAULT_PROCESSED_DIR) -> Path:
    """Save extracted text JSON to data/processed/<case_number>/<doc>.json."""
    dest = case_processed_dir(extracted.case_number, base) / f"{extracted.document_filename}.json"
    dest.write_text(
        extracted.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.debug(
        "Saved extracted text for %s/%s",
        extracted.case_number,
        extracted.document_filename,
    )
    return dest


def load_metadata(case_number: str, base: Path = DEFAULT_RAW_DIR) -> CaseMetadata | None:
    path = case_raw_dir(case_number, base) / "metadata.json"
    if not path.exists():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return CaseMetadata.model_validate(raw)
    except Exception:
        logger.exception("Failed to load metadata for %s", case_number)
        return None


def list_scraped_cases(base: Path = DEFAULT_RAW_DIR) -> list[str]:
    """Return case numbers that already have metadata saved."""
    if not base.exists():
        return []
    return [d.name for d in sorted(base.iterdir()) if d.is_dir() and (d / "metadata.json").exists()]


def case_pdfs(case_number: str, base: Path = DEFAULT_RAW_DIR) -> list[Path]:
    """List PDF files already downloaded for a case."""
    case_dir = base / case_number
    if not case_dir.exists():
        return []
    return sorted(case_dir.glob("*.pdf"))
