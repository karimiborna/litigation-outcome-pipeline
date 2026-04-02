# Scraper Module

Respectful scraper for SF Superior Court civil case data, with NVIDIA NeMo Retriever for PDF text extraction.

## Court Site Structure

**Base endpoint:** `https://webapps.sftc.org/ci/CaseInfo.dll?&SessionID=<SESSION_ID>`

The site uses a session ID in the URL. Tabs navigate via `?CaseNum=<NUM>&SessionID=<SID>#` anchors on the same page.

### Step 1: Search by New Filings (date search)

- Third tab on the search page: "Search by New Filings"
- Form field: `Filing Date` in `YYYY-MM-DD` format (e.g., `2026-04-01`)
- Submit the form → returns paginated results (10 per page, can be 100+ entries per day)
- Results table columns: **Case Number** (clickable link), **Case Title**
- Pagination at bottom: Previous / 1 / 2 / ... / N / Next

### Step 2: Case Detail Page

Clicking a case number loads a detail page with 6 tabs:

1. **Register of Actions** — Date, Proceedings text, Document ("View" link → PDF), Fee
2. **Parties** — Party name, Party Type (PLAINTIFF/DEFENDANT/APPELLANT/RESPONDENT), Attorneys, Filings links
3. **Attorneys** — Name, Bar Number, Address/Phone, Parties Represented
4. **Calendar** — hearing dates (not yet explored)
5. **Payments** — fee info (not yet explored)
6. **Documents** — consolidated list: Date, Description (clickable links to PDFs)

### Step 3: Download PDFs

- "View" links in Register of Actions and document links in Documents tab point to PDFs
- **CRITICAL: Document links expire ~10 minutes after page generation** (red warning banner on page)
- Must download PDFs immediately after loading the case detail page — cannot queue for later

### URL Pattern

Case detail pages: `?CaseNum=<CASE_NUMBER>&SessionID=<SID>#`
Example: `?CaseNum=APP26008959&SessionID=97F9B63ED07FAE5E8FBB26E84ADF8CAB59F7D04F#`

Note: case number in URL strips the dash (APP-26-008959 → APP26008959).

## Scraping Flow

```
For each date in range:
  1. POST/GET search form with Filing Date
  2. Parse result table → list of (case_number, case_title)
  3. Paginate through all pages
  4. For each case:
     a. Load case detail page
     b. Scrape header metadata (case number, title, cause of action)
     c. Scrape Parties tab (party names, types, attorneys)
     d. Scrape Register of Actions (proceedings text, dates)
     e. Scrape Documents tab → get PDF links
     f. IMMEDIATELY download all PDFs (links expire in ~10 min)
     g. Save metadata as JSON, PDFs to data/raw/
  5. Rate limit between requests (2-3 sec minimum)
```

## What to Scrape Per Case

| Field | Source Tab | Notes |
|---|---|---|
| Case number | Header | e.g., APP26008959 |
| Case title | Header | Full plaintiff vs defendant string |
| Cause of action | Header | e.g., TRAFFIC APPEAL, SMALL CLAIMS, etc. |
| Filing date | Search context | The date used to find this case |
| Parties | Parties tab | Name, type (plaintiff/defendant), pro per or attorney |
| Attorneys | Attorneys tab | Name, bar number, firm, party represented |
| Proceedings | Register of Actions | Date + proceedings text for each entry |
| Documents (PDFs) | Documents tab / ROA "View" links | Download immediately, links expire |

## PDF Document Characteristics

Court documents are **scanned image PDFs** — not digital/searchable PDFs. Key details:

- Created by Fujitsu document scanners (PaperStream Capture)
- No embedded text layer — `pdftotext` returns nothing
- Typically 1-3 pages, ~50-70KB each
- Page size: ~614 x 818 pts
- **OCR is required** for all text extraction — there is no digital text to parse
- Expect OCR noise: stamps, handwriting, poor scan quality, skewed pages
- Post-extraction text cleaning will be necessary before feeding into the features pipeline

## NVIDIA NeMo Retriever (PDF Extraction)

**What:** NeMo Retriever Extraction (nv-ingest) — extracts text, tables, charts, and images from PDFs. Runs OCR under the hood, which is essential since all court documents are scanned images.

**Library mode (cloud-hosted, no self-hosting needed):**

```python
from nv_ingest_client.client.interface import Ingestor

ingestor = Ingestor().files("path/to/case.pdf")
result = ingestor.extract(
    extract_text=True,
    text_depth="page",
    extract_tables=True,
    extract_charts=True,
    extract_images=True
).ingest()
```

**Authentication:**
```bash
export NVIDIA_BUILD_API_KEY=nvapi-<your key>
export NVIDIA_API_KEY=nvapi-<your key>
```

- Get API keys at: https://org.ngc.nvidia.com/setup/api-keys (select NGC Catalog + Public API Endpoints)
- Cloud endpoint: `integrate.api.nvidia.com`
- Free tier: ~500+ extractions/day per key
- Library mode suited for <100 docs per batch
- Requires Python 3.12

**Scaling:** Each team member generates their own NVIDIA API key. Scraper supports multiple keys and rotates between them, tracking per-key daily usage.

**Docs:** https://docs.nvidia.com/nemo/retriever/25.6.2/extraction/overview/index.html

## Respectful Scraping Rules

These are non-negotiable:

- **Rate limiting** — enforce delays between requests (minimum 2-3 seconds between court site hits)
- **Retry with backoff** — exponential backoff on 429/5xx responses, do not hammer the server
- **Respect robots.txt** — check and honor it
- **User-Agent** — identify as a research/academic scraper, not a generic bot
- **Session management** — reuse sessions, don't create excessive new ones; handle session expiry gracefully
- **Error tracking** — log all failures, do not silently retry in tight loops
- **Daily caps** — configurable max requests per day to the court site
- **NVIDIA API key rotation** — support multiple keys, distribute load, track per-key usage against ~500/day limit
- **Resume support** — track what's already been scraped so we never re-scrape or re-extract

## Output

- Raw PDFs → `data/raw/<case_number>/`
- Case metadata (JSON) → `data/raw/<case_number>/metadata.json`
- Extracted text → `data/processed/<case_number>/`
- Scrape manifest/log tracking: date scraped, cases found, PDFs downloaded, extraction status

## Key Considerations

- **DATA EXPIRES AFTER 120 DAYS** — the court site only serves filings from the last 120 days. Scraping must start from the oldest available date (120 days ago) and work forward to the present. Any date older than 120 days is gone forever. This makes the scraper time-sensitive — every day we delay, we lose a day of historical data off the back end.
- Document links expire ~10 minutes after page load — download PDFs immediately, never queue
- Session IDs may rotate — scraper must detect expiry and re-acquire
- Court site uses DataTables (jQuery plugin) for pagination — may need to handle client-side rendering or find the underlying data source
- Store raw HTML responses before parsing so we can re-parse without re-scraping
- Keep all credentials (NVIDIA API keys, session tokens) out of code — use environment variables
- The scraper is the entry point of the entire pipeline — data quality here cascades everywhere
