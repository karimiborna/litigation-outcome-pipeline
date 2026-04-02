# Scraper Module

Respectful scraper for SF Superior Court small claims case data, with NVIDIA API-based PDF extraction.

## Data Source

- **Court endpoint:** `https://webapps.sftc.org/ci/CaseInfo.dll?&SessionID=97F9B63ED07FAE5E8FBB26E84ADF8CAB59F7D04F`
- **Search format:** date-based, format `YYYY-MM-DD` (e.g., `2026-04-01`)
- Session IDs may rotate — scraper must handle session expiry and re-acquisition

## PDF Extraction

- **NVIDIA API** for extracting structured text from court PDFs
- Free tier: ~500+ extractions per day per API key
- To scale: each team member uses their own NVIDIA API key
- PDF extraction is the pipeline bottleneck — plan around this rate limit

## Respectful Scraping Rules

These are non-negotiable:

- **Rate limiting** — enforce delays between requests (minimum 2-3 seconds between court site hits)
- **Retry with backoff** — exponential backoff on 429/5xx responses, do not hammer the server
- **Respect robots.txt** — check and honor it
- **User-Agent** — identify as a research/academic scraper, not a generic bot
- **Session management** — reuse sessions, don't create excessive new ones
- **Error tracking** — log all failures, do not silently retry in tight loops
- **Daily caps** — configurable max requests per day to the court site
- **NVIDIA API key rotation** — support multiple keys, distribute load across them, track per-key usage against the ~500/day limit
- **Resume support** — track what's already been scraped so we never re-scrape or re-extract

## Output

- Raw PDFs stored in `data/raw/`
- Extracted text/structured data stored in `data/processed/`
- Scrape manifest/log tracking: date scraped, cases found, PDFs downloaded, extraction status

## Key Considerations

- Court websites can be fragile — handle malformed HTML, missing fields, and unexpected page layouts gracefully
- Store raw responses before parsing so we can re-parse without re-scraping
- Keep credentials (NVIDIA API keys) out of code — use environment variables
- The scraper is the entry point of the entire pipeline — data quality here cascades everywhere
