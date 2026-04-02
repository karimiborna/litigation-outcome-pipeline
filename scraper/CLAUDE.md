# Scraper Module

Scrapes SF Superior Court small claims case data via reverse-engineered DataSnap REST API, downloads case PDFs, and extracts text using PyMuPDF with NVIDIA vision API fallback.

## Data Source

The SF Superior Court runs a legacy Delphi DataSnap REST backend behind Cloudflare. Two `.dll` endpoints:

| Endpoint | Purpose |
|---|---|
| `/cc/CaseCalendar.dll` | Search cases by date, name, or case number |
| `/ci/CaseInfo.dll` | Fetch documents, parties, and ROA for a specific case |

### Reverse-Engineered API Endpoints

**Get cases by date:**
```
GET /cc/CaseCalendar.dll/datasnap/rest/TServerMethods1/GetCases2/{date}/{case_type}/{session_id}
```
- `date` — `YYYY-MM-DD` format
- `case_type` — `M//CSM` for Small Claims
- Returns JSON: `{"result": [count, "[{...case objects...}]"]}`
- `result[0] == -1` → session expired; `== 0` → no cases found

**Get documents for a case:**
```
GET /ci/CaseInfo.dll/datasnap/rest/TServerMethods1/GetDocuments/{case_num}/{session_id}/
```
- Document URLs are **time-limited signed links** (~10 min expiry)
- PDFs must be downloaded immediately — URLs cannot be cached

### Session Management

Session IDs are Cloudflare-issued and require browser CAPTCHA. **No programmatic acquisition possible.** User must:
1. Visit `https://webapps.sftc.org/cc/CaseCalendar.dll` in browser
2. Copy SessionID from the URL
3. Export as `SFTC_SESSION_ID` env var

Sessions expire after ~10 minutes of inactivity.

## Architecture

```
scraper/
├── config.py          # Constants, env-based settings (ScraperConfig)
├── session.py         # Session ID validation and error types
├── court_api.py       # DataSnap REST API client (get_cases, get_documents, download_pdf)
├── extractor.py       # PDF text extraction (pymupdf + NVIDIA vision fallback)
├── court_scraper.py   # Main orchestrator (CourtScraper class)
├── cli.py             # Click CLI entry point
├── rate_limiter.py    # Request delay + daily cap enforcement
└── manifest.py        # Resume support (tracks scraped dates/cases)
```

## PDF Text Extraction

Two-tier strategy:
1. **PyMuPDF** (free, instant) — extracts selectable text. If >100 chars found, done.
2. **NVIDIA Vision API** (fallback) — renders pages to JPEG at 150 DPI, sends to `meta/llama-3.2-90b-vision-instruct` for OCR. Uses OpenAI-compatible client.

NVIDIA free tier: ~500 requests/day per key. Most court PDFs have selectable text so quota is preserved.

## Key Dependencies

- `requests` — HTTP client (synchronous, rate-limited to 2.5s between calls)
- `pymupdf` (fitz) — PDF text extraction and page rendering
- `openai` — NVIDIA vision API client (OpenAI-compatible endpoint)

## Output

```
data/raw/pdfs/            # Downloaded PDFs (immutable)
data/processed/extracted/ # Extracted .txt files (one per PDF)
scraper/state/            # Manifest JSON for resume support
```

Filename convention: `{case_num}_{sanitized_description}.pdf`

## Respectful Scraping

- 2.5-second minimum delay between requests
- 200 requests/day cap (configurable)
- Academic research User-Agent header
- Skip weekends (no court dates)
- Resume support via manifest — never re-scrapes or re-downloads
- Graceful handling of session expiry
