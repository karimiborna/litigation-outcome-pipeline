# Scraper Module

Harvests SF Superior Court small claims case data via reverse-engineered DataSnap REST API.

## Entry Points

```bash
# CLI (preferred)
python -m scraper.cli scrape --start 2025-12-01 --end 2026-04-01
python -m scraper.cli status
python -m scraper.cli enumerate --range CSM25870000-CSM25879999
python -m scraper.cli download-cases

# Legacy standalone script
python scraper/scrape.py --date 2026-04-01
python scraper/scrape.py --backfill   # last 120 days
```

## File Map

| File | Purpose |
|---|---|
| `cli.py` | Click CLI — commands: scrape, status, extract, enumerate, download-cases |
| `court_api.py` | DataSnap REST client — get_cases, get_documents, get_roa, probe_case_exists |
| `court_scraper.py` | Orchestrator — CourtScraper class, scrape_date_range, build_date_range |
| `enumerator.py` | Brute-force case discovery — CaseEnumerator, ValidCasesStore |
| `manifest.py` | Resume support — ScrapeManifest, CaseRecord, DateRecord (JSON-backed) |
| `rate_limiter.py` | Token bucket — 2.5s min delay, 200 req/day cap |
| `session.py` | Session ID — read from env or prompt user interactively |
| `extractor.py` | PDF text extraction — pymupdf first, NVIDIA vision API fallback |
| `scrape.py` | Legacy standalone scraper with --backfill support |
| `config.py` | Constants + ScraperConfig (pydantic-settings) |
| `state/valid_cases.json` | Persisted valid case numbers from enumeration runs |

## Court Site Architecture

**Base endpoint:** `https://webapps.sftc.org`

Two DLL services expose DataSnap REST APIs:
- `/cc/CaseCalendar.dll` — search cases by hearing date
- `/ci/CaseInfo.dll` — case detail, documents, parties, ROA

**Session IDs** are Cloudflare-gated (CAPTCHA) — must be obtained from a real browser. Sessions expire from inactivity. The keepalive thread in session_manager.py pings every 2 minutes to prevent expiry.

**API endpoints (reverse-engineered from lookup.js / cslookup.js):**
```
GET /cc/CaseCalendar.dll/datasnap/rest/TServerMethods1/GetCases2/{date}/M//CSM/{session_id}
GET /ci/CaseInfo.dll/datasnap/rest/TServerMethods1/GetDocuments/{case_num}/{session_id}/
GET /ci/CaseInfo.dll/datasnap/rest/TServerMethods1/GetROA/{case_num}/{session_id}/
GET /ci/CaseInfo.dll/datasnap/rest/TServerMethods1/GetParties/{case_num}/{session_id}/
```

Result format: `{"result": [count, "[{...json array...}]"]}`
- `result[0] == -1` → session expired
- `result[0] == 0` → no results
- `result[0] > 0` → count, parse `result[1]` as JSON

**Document URLs are time-limited (~10 min signed URLs)** — download PDFs immediately after fetching the document list, never queue for later.

**DATA EXPIRES AFTER 120 DAYS** — the court site only serves filings from the last 120 days. Run `--backfill` ASAP to capture historical data before it disappears.

## PDF Extraction Strategy

All court PDFs are scanned image PDFs (Fujitsu PaperStream scanner, no embedded text):
1. **pymupdf** — try selectable text extraction first (free, instant, no quota)
2. **NVIDIA vision API** — fallback when <100 chars found (for scanned/image PDFs)

NVIDIA model: `meta/llama-3.2-90b-vision-instruct` via `integrate.api.nvidia.com/v1`
Free tier: ~500 extractions/day per key. Each team member uses their own key (~1,500/day total).

## Output Structure

```
data/raw/<case_number>/
    metadata.json              # case number, title, hearing date, parties
    CLAIM_OF_PLAINTIFF.pdf
    PROOF_OF_SERVICE.pdf
    ...

data/processed/<case_number>/
    CLAIM_OF_PLAINTIFF.txt     # extracted text (pymupdf or NVIDIA)
    PROOF_OF_SERVICE.txt
    ...

data/scraped_dates.txt         # resume log — one date per line
data/manifest.json             # full scrape manifest (ScrapeManifest)
scraper/state/valid_cases.json # case numbers found via enumeration
```

## Respectful Scraping Rules

- 2.5s minimum delay between all court site requests
- Exponential backoff on 429/5xx (base 2, max 4 retries)
- 200 requests/day cap to court site
- Academic User-Agent header
- Never re-scrape — manifest and scraped_dates.txt track completed dates
- NVIDIA daily cap: 490/day per key (10-request buffer)

## Getting a Session ID

```
1. Visit https://webapps.sftc.org/cc/CaseCalendar.dll in your browser
2. Copy SessionID=... from the redirected URL
3. Paste into .env:
   SFTC_SESSION_ID=<value>
   NVIDIA_API_KEY=<value>
```
