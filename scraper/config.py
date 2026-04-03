"""Scraper configuration — court API endpoints, NVIDIA settings, and env vars."""

from __future__ import annotations

from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings

# SF Superior Court endpoints (DataSnap REST API behind Delphi .dll)
BASE_URL = "https://webapps.sftc.org"
CALENDAR_PATH = "/cc/CaseCalendar.dll"
CASE_PATH = "/ci/CaseInfo.dll"

# DataSnap REST paths (appended to the .dll endpoints above)
CALENDAR_API = "/datasnap/rest/TServerMethods1/GetCases2/{date}/{case_type}/{session_id}"
CASE_DOCS_API = "/datasnap/rest/TServerMethods1/GetDocuments/{case_num}/{session_id}/"
CASE_ROA_API = "/datasnap/rest/TServerMethods1/GetROA/{case_num}/{session_id}/"

# Small claims case type value from the calendar dropdown
SMALL_CLAIMS_TYPE = "M//CSM"

# Document types worth downloading — substring-matched against the API's DESCRIPTION field.
# Everything else (proof of service, continuances, scheduling, etc.) is skipped.
DOC_TYPE_WHITELIST = {
    "CLAIM_OF_PLAINTIFF",
    "DEFENDANT_S_CLAIM",
    "JUDGMENT",
    "ORDER",
    "DISMISSAL",
    "Notice_of_Entry_of_Judgment",
    "DECLARATION_OF_APPEARANCE",
    "STIPULATION",
    "COURT_JUDGMENT",
}


def is_doc_type_wanted(description: str) -> bool:
    """Check if a document description matches any whitelisted type."""
    desc_upper = description.upper()
    return any(w.upper() in desc_upper for w in DOC_TYPE_WHITELIST)


# NVIDIA vision model for scanned PDF extraction
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
PDF_IMAGE_DPI = 150


class ScraperConfig(BaseSettings):
    """All scraper settings, loaded from env vars or .env file."""

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }

    # Session ID — must be obtained manually from browser (CAPTCHA-gated)
    session_id: str = Field(default="", alias="SFTC_SESSION_ID")

    # NVIDIA API key for vision-based OCR on scanned PDFs
    nvidia_api_key: str = Field(default="", alias="NVIDIA_API_KEY")

    user_agent: str = Field(
        default=(
            "MSDS603-Research-Scraper/1.0 "
            "(SF Small Claims academic study; contact: msds603-team@usfca.edu)"
        ),
        alias="SCRAPER_USER_AGENT",
    )

    # Rate limiting
    rate_limit_seconds: float = Field(default=2.5, alias="SCRAPER_RATE_LIMIT_SECONDS")
    max_daily_requests: int = Field(default=200, alias="SCRAPER_MAX_DAILY_REQUESTS")

    # Retry
    max_retries: int = 4
    retry_backoff_base: float = 2.0

    # Timeouts (seconds)
    request_timeout: float = 30.0
    pdf_download_timeout: float = 60.0

    # Caps
    max_pdfs_per_run: int = Field(default=0, alias="SCRAPER_MAX_PDFS")
    nvidia_daily_cap_per_key: int = 490

    # Data paths
    raw_dir: Path = Field(default=Path("data/raw"), alias="DATA_RAW_DIR")
    processed_dir: Path = Field(default=Path("data/processed"), alias="DATA_PROCESSED_DIR")
