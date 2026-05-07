# notebooks/

Four notebooks:
- `colab_gpu_extraction.ipynb` — full extraction pipeline on Colab GPU (steps 1–6)
- `local_pdf_pipeline.ipynb` — local-only steps 1–2 (download + PyMuPDF) on M4 via Drive Desktop sync, used when Cloudflare blocks the Colab IP range
- `ernesto_ML.ipynb` — exploratory ML training and analysis (local)
- `scrape_new_cases.ipynb` — local-only probing for new SF court case numbers

See `scrape_new_cases_walkthrough.md` for a step-by-step companion to the scraping notebook.

## ernesto_ML.ipynb

Local ML training notebook. Loads `dataset.csv`, trains classifier and regressor using the `v2 feat_*` feature set, and logs runs to the hosted MLflow server (`http://35.208.251.175:5000`). Use this for exploratory model work and validation before running `scripts/train_models.py` for the production registry run.

## colab_gpu_extraction.ipynb

End-to-end pipeline notebook designed for Google Colab (T4 GPU runtime). Handles:

1. **PDF download** from SF Superior Court API (`webapps.sftc.org`) using session IDs
2. **Text extraction** via tiered strategy:
   - PyMuPDF (free, instant) for PDFs with selectable text
   - Qwen2-VL-7B 4-bit on the Colab T4 GPU for scanned pages
3. **Label extraction** via GPT-4o-mini — win/loss/partial_win/dismissed/settled, amounts, etc.
   - Uses `response_format={"type": "json_object"}` for guaranteed valid JSON
   - Smart truncation keeps first 1/3 + last 2/3 of long docs (rulings are at the end)
   - Saves per-case JSON files to `labels/` for team deduplication, merges into `labels.json`
4. **Feature extraction** via GPT-4o-mini (Step 6 / Cell 15) — ~40 existence-based booleans per case using the v2 `FeatureExtractor` / `LLMFeatures` schema.
   - Builds `ProcessedCase` per case from the Drive-mounted extracted txts
   - Excludes label docs (`JUDGMENT`, `ORDER`, `DISMISSAL`, `STIPULATION`, `Notice_of_Entry_of_Judgment`, `COURT_JUDGMENT`) to preserve the leakage firewall
   - Auto-detects `user_side` per case: `"defendant"` if a `DEFENDANT_S_CLAIM` txt exists, else `"plaintiff"`
   - Writes per-case `FeatureVector` JSONs to `{DRIVE_DIR}/features_cache/` (content-hashed filenames)
   - Reuses the OpenAI key entered in Step 5, or re-prompts if Step 5 was skipped
   - Supports the same `MY_WORKER`/`TOTAL_WORKERS` stride sharding as earlier steps

### Key details

- Uses `valid_cases.json` from `scraper/state/` as the case list
- **800 total valid cases** in `valid_cases.json`
- Worker sharding (`MY_WORKER / TOTAL_WORKERS`) splits cases across team members
- All steps are idempotent — checks for existing outputs before re-processing
- Outputs: `.txt` files in `extracted/`, per-case JSONs in `labels/`, merged `labels.json`
- Google Drive mount is default for persistence between Colab sessions

### Worker assignments

| Person | `MY_WORKER` | `TOTAL_WORKERS` | Cases |
|--------|-------------|-----------------|-------|
| Ernesto | 0 | 3 | ~267 |
| Alexander | 1 | 3 | ~267 |
| Borna | 2 | 3 | ~266 |

Sharding applies to both PDF downloads and label extraction. Each person sets their
`MY_WORKER` and `TOTAL_WORKERS = 3` in the config cell. The shared Drive folder
prevents duplicates — per-case label JSONs act as locks so no case is labeled twice.

### Document type whitelist

Only these types are downloaded/extracted (procedural docs skipped):
`CLAIM_OF_PLAINTIFF`, `DEFENDANT_S_CLAIM`, `JUDGMENT`, `ORDER`, `DISMISSAL`,
`Notice_of_Entry_of_Judgment`, `DECLARATION_OF_APPEARANCE`, `STIPULATION`, `COURT_JUDGMENT`

### Dependencies

- `pymupdf` (fitz), `requests`, `openai`
- GPU path: `transformers`, `accelerate`, `bitsandbytes`, `qwen-vl-utils`
- Repo must be pip-installable (`pip install -e .`) for `scraper.config`, `scraper.court_api`, `features.labels`, etc. — the notebook imports from these directly

### Known issues

- GPU vision (Qwen2-VL-7B 4-bit via `bitsandbytes`) is the active fallback for scanned PDFs; PyMuPDF handles e-filed PDFs with a text layer. The earlier NVIDIA-API path has been removed.
- `bitsandbytes` 4-bit loading is CUDA-only — Step 3/4 will not run on Apple Silicon (MPS) without swapping the loader to MLX or `bfloat16` unquantized.
- Final zip download assumes Colab environment (`google.colab.files`).

## local_pdf_pipeline.ipynb

Local-only notebook that runs **Step 1 (PDF download)** and **Step 2 (PyMuPDF text extraction)** on the M4. Same code as `colab_gpu_extraction.ipynb` cells 5 + 7, but pointed at the Google Drive Desktop sync folder so PDFs/extracted text land in the same `litigation-pipeline/` directory Colab uses.

### Why this exists

Cloudflare blocks SF court (`webapps.sftc.org`) requests from Colab's IP range but not from the user's laptop (where the browser challenge was solved). Symptom: cell-5 in the Colab notebook returns 403 immediately for every case after Cloudflare flags the runtime's IP. Verified via `curl` test: M4 → 200 for both CSM24 and CSM25 cases; Colab → 403 for both.

### Workflow

1. Install **Google Drive for Desktop** on the M4 and sign in with the same Google account used by Colab. Confirm `~/Library/CloudStorage/GoogleDrive-<email>/My Drive/litigation-pipeline/` exists.
2. `conda activate ML` (the env where the repo is `pip install -e`'d).
3. Open `local_pdf_pipeline.ipynb`, edit `DRIVE_PATH` if your email differs, run cell-3 (setup) — it prints how many PDFs are already in Drive.
4. Run cell-5 (download). When prompted, paste a fresh SessionID from your browser. The skip-existing check covers cases already on Drive (no re-download of the 800 CSM25 cases from prior Colab runs).
5. Run cell-7 (PyMuPDF). Text files land in `<DRIVE_PATH>/extracted/`. Scanned PDFs (<100 chars) get queued for the GPU step.
6. Switch to `colab_gpu_extraction.ipynb` and run from **cell-9 (Step 3 — Load Vision Model)** onward — Colab sees the same files via Drive mount.

### Handoff handling

- Both result-code `-1` (API session expired) and HTTP 403 (Cloudflare reject) trigger a re-prompt for a fresh SessionID via the same retry loop.
- Corrupt PDFs (partial downloads / HTML error pages) are skipped with a log line; user can `.unlink()` them and re-run Step 1 to re-download.
- Interrupt-safe: the per-case skip check on `PDF_DIR.glob` lets you Ctrl+C any time and resume.

## scrape_new_cases.ipynb

Local-only notebook for discovering new SF small claims cases by **probing case-number ranges** and appending them to `scraper/state/valid_cases.json`. Must run on a laptop, not Colab — Cloudflare blocks Colab IPs.

Why this exists: the court calendar API (`GetCases2`) only returns recent dates, so older cases aren't reachable that way. The `GetROA` endpoint works for any case that exists, so this notebook brute-forces a numeric range of `CSM<year><num>` case numbers. Complements (does not replace) the date-based `scrape enumerate` / `scrape download-cases` CLI in `scraper/`.

### Workflow

1. Loads existing `valid_cases.json` so already-known cases are skipped
2. Prompts for a SessionID — manual one-time Cloudflare verification in a browser, then paste the hex `SessionID=` from the URL
3. Starts a 60s background keepalive thread to extend the session
4. Probes every case number in `[PROBE_START_NUM, PROBE_END_NUM]` via `GetROA`
5. Saves valid cases to `valid_cases.json` incrementally — interrupt-safe

Tunables in the config cell: `PROBE_PREFIX` (e.g. `CSM`), `PROBE_START_NUM`, `PROBE_END_NUM`, `PROBE_STEP` (1 = dense, larger = sparse sweep), `PROBE_DELAY` (default 1.0s between probes).

See `scrape_new_cases_walkthrough.md` for a narrative walkthrough of a typical run.
