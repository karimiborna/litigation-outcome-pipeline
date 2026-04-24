# Scraper Module

**Victor's Angels — MSDS 603 MLOps**
Team: Alexander Mohun, Borna Karimi, Ernesto Diaz

This module scrapes small claims court case data from the SF Superior Court public website, downloads case PDFs, and extracts text from them using the NVIDIA vision API. It is the entry point of the entire ML pipeline — data quality here cascades into every downstream step.

---

## What Was Built

### Files

```
scraper/
├── config.py            # All constants and settings (no secrets)
├── scrape.py            # Main scraper script
├── nvidia_extractor.py  # PDF text extraction (pymupdf + NVIDIA fallback)
└── requirements.txt     # Python dependencies
```

### Output

```
data/
├── raw/pdfs/            # Downloaded PDFs (immutable — never modified)
└── processed/extracted/ # Extracted .txt files (one per PDF)
```

---

## How to Run

```bash
# From the repo root (litigation-outcome-pipeline/)
pip3 install -r scraper/requirements.txt

# Required: session ID from your browser
export SFTC_SESSION_ID=<paste from browser URL>

# Optional: only needed if PDFs are image-based (scanned)
export NVIDIA_API_KEY=<your key from build.nvidia.com>

python3 scraper/scrape.py --date 2026-04-01
```

**Getting the Session ID:**
1. Visit `https://webapps.sftc.org/cc/CaseCalendar.dll` in your browser
2. The URL will redirect to something like `...?=&SessionID=XXXXXXXX`
3. Copy the `SessionID` value and export it as shown above

Sessions expire after ~10 minutes of inactivity. When they do, you'll see:
```
SessionExpiredError: Session expired. Get a fresh SessionID from your browser.
```
Just get a new one from your browser and re-export.

---

## How It Works — Deep Dive

### The Court Website

The SF Superior Court runs a legacy Delphi DataSnap REST backend behind Cloudflare. The public-facing site at `webapps.sftc.org` exposes two main `.dll` endpoints:

| Endpoint | Purpose |
|---|---|
| `/cc/CaseCalendar.dll` | Search cases by date, name, or case number |
| `/ci/CaseInfo.dll` | Fetch details, documents, parties, and ROA for a specific case |

Session IDs are issued by Cloudflare and embedded in every URL. They require CAPTCHA to acquire (must be done in a real browser) and expire after inactivity. **There is no way to get a session programmatically.**

### Reverse-Engineering the API

The site's frontend uses jQuery + DataTables and makes AJAX calls to a DataSnap REST API. By inspecting `lookup.js` and `cslookup.js` (the site's own JavaScript files), we identified these JSON endpoints:

**1. Get cases by date:**
```
GET /cc/CaseCalendar.dll/datasnap/rest/TServerMethods1/GetCases2/{date}/{case_type}/{session_id}
```
- `date` — `YYYY-MM-DD` format
- `case_type` — `M//CSM` for Small Claims (discovered from the dropdown HTML)
- Returns JSON: `{"result": [count, "[{...case objects...}]"]}`
- `result[0] == -1` means session expired; `== 0` means no cases found

Each case object contains:
```json
{
  "CASE_NUMBER": "<A HREF=\"...?CaseNum=CSM26871146&SessionID=...\">CSM-26-871146</A>",
  "CASETITLE": "CHUNHONG MENG VS. JOHN ZHOU ET AL",
  "COURTDATE": "2026-04-01 08:30:00",
  "COURT_TIME": "8:30 AM",
  "ROOM": "506",
  "LOCATIONMAPPED": "400 MCALLISTER ST"
}
```

Note: `CASE_NUMBER` is raw HTML with an anchor tag. We parse the bare case number (`CSM26871146`) out of it with a regex on the `CaseNum=` URL parameter.

**2. Get documents for a case:**
```
GET /ci/CaseInfo.dll/datasnap/rest/TServerMethods1/GetDocuments/{case_num}/{session_id}/
```
Returns a list of document objects:
```json
{
  "FILEDATE": "2026-02-05 11:33:28",
  "DESCRIPTION": "CLAIM OF PLAINTIFF",
  "URL": "https://webapps.sftc.org/ci/CaseInfo.dll?SessionID=...&URL=https://imgquery.sftc.org/..."
}
```

The `URL` field is a **time-limited signed link** (valid ~10 minutes from page load) that proxies through `CaseInfo.dll` to the actual PDF stored on `imgquery.sftc.org`. This means PDFs must be downloaded immediately after fetching the document list — URLs cannot be cached for later.

### scrape.py — Main Script

**`get_session_id()`** — reads `SFTC_SESSION_ID` from env, exits with a clear error if missing.

**`get_cases(session_id, date_str)`** — hits the `GetCases2` endpoint and returns the list of case dicts. Raises `SessionExpiredError` if the session has timed out.

**`parse_case_num(case_num_html)`** — uses a regex (`CaseNum=(\w+)&`) to extract the bare case number from the HTML anchor tag in the `CASE_NUMBER` field.

**`get_documents(case_num, session_id)`** — hits the `GetDocuments` endpoint for a specific case. Returns a list of documents with `DESCRIPTION`, `FILEDATE`, and `URL`.

**`download_pdf(doc_url, dest, session)`** — downloads the PDF using streaming chunked reads (8 KB chunks) so large files don't fill memory. Validates the `Content-Type` header to confirm it's actually a PDF before saving. Skips silently if the content type is wrong. Returns `True` on success.

**`main()`** — orchestrates everything:
1. Reads session ID and creates output directories
2. Fetches the case list for the target date
3. For each case, sleeps 2.5 seconds (rate limiting), fetches documents
4. For each document, sleeps 2.5 seconds, downloads the PDF
5. Stops at `MAX_PDFS = 5` total downloads
6. If `NVIDIA_API_KEY` is set, calls `extract_text()` immediately after each download
7. Skips files that already exist on disk (simple resume support)

**Rate limiting:** Every request to the court site is preceded by a `time.sleep(2.5)`. This is enforced in two places — before fetching documents for a case, and before downloading each PDF. The `User-Agent` header identifies the scraper as an academic research tool.

**Filename convention:**
```
data/raw/pdfs/CSM26871146_CLAIM_OF_PLAINTIFF.pdf
```
Format: `{case_num}_{description_sanitized}.pdf`. The description is stripped of special characters and truncated to 40 characters.

### nvidia_extractor.py — Text Extraction

This module has two strategies, tried in order:

**Strategy 1 — pymupdf (free, instant):**
Opens the PDF with `fitz` (PyMuPDF), calls `page.get_text()` on each page, and joins the results. This works for any PDF with selectable/embedded text. If the combined text is longer than 100 characters, it's returned immediately — no API quota consumed.

**Strategy 2 — NVIDIA Vision API (for scanned PDFs):**
Used as a fallback when pymupdf finds no meaningful text (i.e., the PDF is a scanned image). Each page is:
1. Rendered to a JPEG at 150 DPI using `fitz.Matrix` and `page.get_pixmap()`
2. Base64-encoded
3. Sent to `meta/llama-3.2-90b-vision-instruct` on NVIDIA's API (`integrate.api.nvidia.com/v1`) via the OpenAI-compatible client

The prompt instructs the model to extract all text exactly as it appears, prioritizing party names, dates, claim amounts, and rulings. Temperature is set to 0 for deterministic output. Pages are joined with `--- PAGE BREAK ---` separators.

**NVIDIA API quota:** The free tier gives ~500 requests/day per key. Since each PDF page = 1 request, a multi-page scanned PDF uses multiple requests. The team has 3 keys (one per member) for ~1,500 requests/day total. Most SF court PDFs have selectable text, so pymupdf handles them and quota is preserved.

### config.py — Configuration

All constants in one place, zero secrets:

| Constant | Value | Purpose |
|---|---|---|
| `SMALL_CLAIMS_TYPE` | `M//CSM` | Case type code for the dropdown |
| `REQUEST_DELAY_SECS` | `2.5` | Seconds between court site requests |
| `MAX_RETRIES` | `4` | Retry attempts on failure |
| `DAILY_REQUEST_CAP` | `200` | Max daily court site requests |
| `NVIDIA_VISION_MODEL` | `meta/llama-3.2-90b-vision-instruct` | Model for scanned PDFs |
| `PDF_IMAGE_DPI` | `150` | Page render resolution |
| `NVIDIA_DAILY_CAP_PER_KEY` | `490` | Free tier limit with buffer |

---

## What Has Been Done Beyond the Scraper

### Project Documentation (all CLAUDE.md files)
Every module in the pipeline has a detailed CLAUDE.md spec covering responsibilities, key decisions, and constraints:
- `scraper/` — this module
- `data/` — ingestion, validation, raw vs processed separation
- `features/` — LLM feature extraction into structured JSON
- `models/` — binary classifier (win/loss) + regression (claim amount)
- `mlflow/` — experiment tracking and model registry
- `api/` — FastAPI inference service
- `retrieval/` — vector similarity search (sentence-transformers + FAISS/ChromaDB/Pinecone)
- `counterfactual/` — "what if" feature perturbation analysis
- `docker/` — containerization
- `infra/` — cloud deployment (AWS or GCP)
- `tests/` — unit and integration testing strategy
- `.github/workflows/` — CI/CD pipeline

### Architecture Decisions Locked In
- LLM is for feature extraction only — not prediction
- Raw data in `data/raw/` is immutable
- Model artifacts tracked via MLflow, never committed to git
- Secrets in env vars only
- Two-model approach: binary classifier (win probability + std dev) + regressor (claim amount)
- Hybrid retrieval: ML model + NLP similarity search together

### What's Still Pending
- `features/` — LLM prompt to convert extracted text → structured JSON features
- `models/` — training the binary classifier and regressor
- `mlflow/` — experiment tracking setup
- `api/` — FastAPI inference service
- `retrieval/` — embedding generation and vector store
- `counterfactual/` — feature perturbation analysis
- Docker + infra + CI/CD

---

## Dependencies

```
requests==2.32.3   # HTTP client for scraping
pymupdf==1.24.11   # PDF text extraction (fitz)
openai==1.51.0     # NVIDIA API client (OpenAI-compatible)
```

Install: `pip3 install -r scraper/requirements.txt`
