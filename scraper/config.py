"""
Scraper configuration constants.
All secrets come from environment variables — never hardcoded here.
"""

# SF Superior Court endpoints
BASE_URL = "https://webapps.sftc.org"
CALENDAR_PATH = "/cc/CaseCalendar.dll"
CASE_PATH = "/ci/CaseInfo.dll"

# DataSnap REST paths (relative to each .dll)
CALENDAR_API = "/datasnap/rest/TServerMethods1/GetCases2/{date}/{case_type}/{session_id}"
CASE_DOCS_API = "/datasnap/rest/TServerMethods1/GetDocuments/{case_num}/{session_id}/"
CASE_ROA_API  = "/datasnap/rest/TServerMethods1/GetROA/{case_num}/{session_id}/"

# Small claims case type value from the calendar dropdown
SMALL_CLAIMS_TYPE = "M//CSM"

# Respectful scraping parameters
REQUEST_DELAY_SECS = 2.5       # min seconds between court site requests
MAX_RETRIES = 4                # total attempts before giving up
BACKOFF_BASE = 2               # exponential backoff base (seconds)
DAILY_REQUEST_CAP = 200        # max HTTP requests to court site per day

# User-Agent — identify as academic research scraper
USER_AGENT = (
    "MSDS603-Research-Scraper/1.0 "
    "(SF Small Claims academic study; contact: msds603-team@usfca.edu)"
)

# NVIDIA API
NVIDIA_API_BASE = "https://integrate.api.nvidia.com/v1"
NVIDIA_VISION_MODEL = "meta/llama-3.2-90b-vision-instruct"
NVIDIA_DAILY_CAP_PER_KEY = 490  # free tier, leaving 10-request buffer
PDF_IMAGE_DPI = 150             # resolution when rendering PDF pages to images

# Data paths (relative to repo root — callers should resolve to absolute)
RAW_DATA_DIR = "data/raw/pdfs"
PROCESSED_DATA_DIR = "data/processed/extracted"
MANIFEST_DB = "data/manifest.db"
