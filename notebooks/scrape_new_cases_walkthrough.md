# Walkthrough: `scrape_new_cases.ipynb`

## What This Notebook Does

`scrape_new_cases.ipynb` is a local helper notebook for discovering SF small-claims cases that are not available through the recent court calendar API. Instead of searching by hearing date, it probes case numbers directly against the court `GetROA` endpoint and records which cases exist.

Run this notebook on a local laptop, not in Colab. The SF court site is protected by Cloudflare, so the notebook requires a browser-generated `SessionID` that must be copied manually.

## Why It Uses Case-Number Probing

The court calendar endpoint only returns recent cases. Older dates usually come back empty, even when the case still exists. The `GetROA` endpoint works by case number, so the notebook brute-forces a configured numeric range such as:

```python
CSM25870000 -> CSM25879999
```

For this project, `CSM` identifies small-claims cases. The numeric prefix also encodes the year range, with examples like `CSM25...` for 2025 and `CSM26...` for 2026.

## Current Configuration

The notebook currently probes:

```python
PROBE_PREFIX    = "CSM"
PROBE_START_NUM = 25870000
PROBE_END_NUM   = 25879999
PROBE_DELAY     = 1.0
```

That means the full configured range contains `10,000` candidate case numbers, waiting one second between requests for newly probed cases. The output file is:

```text
scraper/state/valid_cases.json
```

The current local state file contains:

- `800` valid 2025 cases
- `800` probed 2025 cases through `CSM25870799`
- `238` not-found entries

Because the notebook skips cases already listed in `valid_cases.json`, the next run resumes with unprobed 2025 candidates.

## Cell-By-Cell Flow

### Cell 0: Notebook Purpose

The opening markdown cell explains the workflow:

- Load known case numbers from `valid_cases.json`
- Ask the user for a court `SessionID`
- Start a background keepalive request loop
- Probe each case number with `GetROA`
- Save progress incrementally

This cell also explains why date-based scraping is not enough for older cases.

### Cell 1: Configuration And Paths

This cell imports standard libraries, defines the case range, builds repo-relative paths, and prints a quick run summary.

Important values:

- `PROBE_PREFIX`: the court case prefix, currently `CSM`
- `PROBE_START_NUM` and `PROBE_END_NUM`: inclusive numeric range to probe
- `PROBE_DELAY`: delay between court requests
- `VALID_CASES_PATH`: where discovery results are stored
- `BASE_URL`, `CALENDAR_PATH`, and `CASE_PATH`: court API URL pieces

The path logic supports running from either the repo root or the `notebooks/` directory.

### Cell 2: Step 1 Header

This markdown cell introduces the resume behavior. The notebook should not re-fetch anything already listed in `valid_cases.json`.

### Cell 3: Load Existing State

This cell reads `scraper/state/valid_cases.json` if it exists.

It extracts:

- `valid`: cases confirmed to exist
- `probed`: cases already checked, whether found or not
- `not_found`: cases known not to exist

If the file does not exist, it starts with an empty structure:

```python
{"valid": {}, "not_found": [], "probed": []}
```

This is what makes the notebook resumable.

### Cell 4: Step 2 Header

This markdown cell explains how to get a `SessionID`.

Required user input:

1. Open `https://webapps.sftc.org/cc/CaseCalendar.dll`
2. Complete the Cloudflare browser check
3. Copy the hex string after `SessionID=` from the browser URL
4. Paste it into the notebook prompt

### Cell 5: Session Input And Keepalive

This cell imports `threading`, `requests`, and `date`, then defines `start_keepalive()`.

The keepalive loop sends periodic requests to the calendar endpoint every 60 seconds using the current session ID. This may extend idle sessions, but it cannot guarantee the session will never expire because the court site may enforce a hard time limit.

The cell then prompts for:

```text
Paste SessionID:
```

After a value is entered, it starts the keepalive thread.

### Cell 6: Step 3 Header

This markdown cell documents the `GetROA` request used for each case:

```text
GET /ci/CaseInfo.dll/datasnap/rest/TServerMethods1/GetROA/{case_num}/{SessionID}/
```

Expected response meanings:

- `count == -1`: session expired
- `count == 0`: case does not exist
- any other count: case exists

### Cell 7: Probe Loop And Saving

This is the main execution cell.

It defines:

- `probe_case(case_num, sid)`: calls `GetROA` and returns `expired`, `found`, `not_found`, or `error`
- `save_progress()`: writes the current in-memory store back to `valid_cases.json`

Then it builds a candidate list from the configured range, skipping any case already in `probed`.

For each candidate:

- Waits `PROBE_DELAY` seconds
- Calls `GetROA`
- If expired, saves progress and prompts for a fresh `SessionID`
- If found, writes the case number and ROA document count into `valid`
- If not found, records it in both `not_found` and `probed`
- If an error occurs, reports it but does not mark the case as successfully probed
- Prints one concise status line for each newly probed candidate

Progress is saved every 25 probes and once again at the end.

## What Changes When You Run It

Running the notebook makes live requests to the SF court site and mutates:

```text
scraper/state/valid_cases.json
```

The notebook is interruption-safe because it saves frequently and skips already-probed cases on the next run.

## Practical Notes

- Use a local machine because Colab IPs are blocked by Cloudflare.
- Keep the browser session open while the notebook runs.
- If the session expires, refresh the court page, copy the new `SessionID`, and paste it into the notebook prompt.
- This notebook is configured for the full 2025 range.
- Keep the delay and rate limits respectful when resuming or expanding future runs.
