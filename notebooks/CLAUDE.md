# notebooks/

Four notebooks:
- `colab_gpu_extraction.ipynb` — full extraction pipeline (steps 1–6). Auto-detects runtime: Colab T4 (CUDA + bitsandbytes), Apple Silicon Mac (MLX via `mlx-vlm`), or generic CUDA host like RunPod (CUDA + bitsandbytes + rclone-mounted Drive).
- `local_pdf_pipeline.ipynb` — local-only steps 1–2 (download + PyMuPDF) on M4 via Drive Desktop sync, used when Cloudflare blocks the Colab IP range
- `ernesto_ML.ipynb` — exploratory ML training and analysis (local)
- `scrape_new_cases.ipynb` — local-only probing for new SF court case numbers

See `scrape_new_cases_walkthrough.md` for a step-by-step companion to the scraping notebook.

## ernesto_ML.ipynb

Local ML training notebook. Loads `dataset.csv`, trains classifier and regressor using the `v2 feat_*` feature set, and logs runs to the hosted MLflow server (`http://35.208.251.175:5000`). Use this for exploratory model work and validation before running `scripts/train_models.py` for the production registry run.

## colab_gpu_extraction.ipynb

Tri-mode GPU + LLM extraction notebook. Cell 1 sets `BACKEND = "mlx"` on Apple Silicon (`platform.machine() == "arm64"`), or `BACKEND = "cuda"` on Colab and any generic Linux CUDA host (detected via `nvidia-smi`); Cells 3, 7, 9, and 15 branch on `BACKEND` and on the `IS_*` flags. PDF downloads and PyMuPDF text extraction are not in this notebook — they run on the laptop via `local_pdf_pipeline.ipynb` because Cloudflare blocks Colab IPs from `webapps.sftc.org`. This notebook reads from the Drive-synced `pdfs/` and `extracted/` folders and handles:

1. **Find scanned PDFs** — globs `PDF_DIR` for whitelisted doc types lacking a `.txt` companion in `OUTPUT_DIR` and builds `needs_gpu`.
2. **Vision OCR** via Qwen2-VL-7B 4-bit. On Colab uses `Qwen/Qwen2-VL-7B-Instruct` + bitsandbytes; on Mac uses `mlx-community/Qwen2-VL-7B-Instruct-4bit` + `mlx-vlm`. Targeted prompt for SC-100 claim docs that skips bilingual boilerplate; generic prompt otherwise. Pages with no case data are dropped.
3. **Label extraction** via GPT-4o-mini — win/loss/partial_win/dismissed/settled, amounts, dates. Output merged into `{DRIVE_DIR}/labels.json`.
4. **Feature extraction** via GPT-4o-mini — ~40 existence-based booleans per case using the v2 `FeatureExtractor` / `LLMFeatures` schema.
   - Builds `ProcessedCase` per case from the Drive-mounted extracted txts
   - Excludes label docs via `features.labels._is_label_doc` (`JUDGMENT`, `ORDER`, `DISMISSAL`, `STIPULATION`, `Notice_of_Entry_of_Judgment`, `COURT_JUDGMENT`) to preserve the leakage firewall
   - Auto-detects `user_side` per case: `"defendant"` if a `DEFENDANT_S_CLAIM` txt exists, else `"plaintiff"`
   - Writes per-case `FeatureVector` JSONs to `{DRIVE_DIR}/features_cache/` (content-hashed filenames)
   - Reuses the OpenAI key entered in Step 4, or re-prompts if Step 4 was skipped
   - Supports `MY_WORKER`/`TOTAL_WORKERS` stride sharding for the team
5. **Results** — prints summary, bundles `extracted/` to a zip, downloads it.

### Key details

- Reads PDFs + existing PyMuPDF txts from the shared Drive folder (populated by `local_pdf_pipeline.ipynb`)
- Worker sharding (`MY_WORKER / TOTAL_WORKERS`) splits feature-extraction work across team members
- All steps are idempotent — checks for existing outputs before re-processing
- Outputs: `.txt` files in `extracted/`, merged `labels.json` at `{DRIVE_DIR}/labels.json`, per-case feature JSONs in `features_cache/`
- Google Drive mount is required (no local-FS fallback)

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

- Common: `pymupdf` (fitz), `requests`, `openai`, `transformers>=4.45`, `Pillow`
- Colab/CUDA path: `accelerate`, `bitsandbytes>=0.46.1`, `qwen-vl-utils`
- Mac/MLX path: `mlx-vlm>=0.1.0` (pulls in `mlx`, `mlx-lm`)
- Repo must be pip-installable (`pip install -e .`) for `scraper.config`, `scraper.court_api`, `features.labels`, etc. — the notebook imports from these directly. On Mac the editable install lives in conda env `ML`; on Colab Cell 3 runs `pip install -e {REPO_DIR}` after cloning.

### Known issues

- GPU vision (Qwen2-VL-7B 4-bit) is the active fallback for scanned PDFs; PyMuPDF handles e-filed PDFs with a text layer. The earlier NVIDIA-API path has been removed.
- The Colab T4 has 16 GB VRAM; the M4 Pro has 24 GB unified memory; an RTX 4090 has 24 GB. All fit Qwen2-VL-7B 4-bit comfortably.
- `mlx-vlm.generate` returns a `GenerationResult` in newer versions and a string in older ones; `extract_page_mlx` normalizes both.
- Final zip download assumes Colab environment — wrapped in `if IS_COLAB:` so the Mac and RunPod paths skip it (outputs already live in the synced Drive folder).

## Generic CUDA / RunPod path

When Colab's monthly compute units are exhausted and the M4 MLX run is too slow, run on a rented RunPod GPU. Cell 1 detects this with `IS_GENERIC_CUDA` (Linux + `nvidia-smi` works + not Colab) and Cell 3 expects Drive to already be mounted at `/workspace/drive/litigation-pipeline` via rclone.

Cost reference (May 2026): RTX 4090 24 GB on Community Cloud is ~$0.34/hr. A full 2,425-PDF run takes ~2-4 hours = under $2. Per-second billing means you stop paying the moment the run ends.

### One-time rclone setup (do once on Mac)

1. `brew install rclone`
2. `rclone config` → `n` (new) → name `drive` → storage `drive` (Google Drive) → leave `client_id`/`client_secret` blank → scope `1` (full access) → leave `service_account_file` blank → `n` to advanced → `y` to auto config (browser OAuth as the team Drive account) → `n` to Shared Drive → `y` to confirm
3. Verify: `rclone lsd drive:litigation-pipeline` should list `pdfs`, `extracted`, etc.
4. The OAuth token now lives in `~/.config/rclone/rclone.conf` — keep this file safe; it grants Drive access until revoked from <https://myaccount.google.com/connections>.

### Per-pod setup (~5 min)

1. RunPod dashboard → Deploy a Pod
   - **GPU**: RTX 4090 24 GB (Community Cloud)
   - **Template**: `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`
   - **Container disk**: 30 GB
   - **Volume disk**: 0 GB (Drive mount handles persistence)
   - Expose port 8888 for Jupyter Lab
2. Click "Connect → Connect to Jupyter Lab" once the pod is running.
3. In Jupyter Lab terminal:
   ```bash
   curl https://rclone.org/install.sh | sudo bash
   ```
4. Upload your Mac's `~/.config/rclone/rclone.conf` via the Jupyter Lab file browser (drag-and-drop into `/workspace/`), then move it:
   ```bash
   mkdir -p ~/.config/rclone && mv /workspace/rclone.conf ~/.config/rclone/
   ```
5. Mount Drive in the background:
   ```bash
   mkdir -p /workspace/drive
   rclone mount drive: /workspace/drive --vfs-cache-mode writes --daemon --log-file /tmp/rclone.log
   ```
6. Verify the mount: `ls /workspace/drive/litigation-pipeline/pdfs | head -3`
7. Open `notebooks/colab_gpu_extraction.ipynb` in Jupyter Lab and run from Cell 1. Cell 1 prints `Backend: CUDA — NVIDIA GeForce RTX 4090 (24.0 GB) [Generic CUDA (e.g. RunPod)]`.

### Per-run teardown

- Outputs sync to Drive in real time via `rclone mount` writethrough — no manual upload needed.
- Stop or terminate the pod from the RunPod dashboard. **Terminate** is free (deletes the pod entirely); **Stop** keeps the disk billed at ~$0.10/GB/mo.
- Mac's Drive Desktop syncs the new `extracted/` files within ~1 minute.

### Notes

- rclone mount writes are throttled by Google Drive's API rate limits (~10 writes/sec sustained). Fine for the OCR loop's slow per-page output; would matter if a future change tries to batch-write thousands of small files.
- If `rclone mount` fails with "fuse: device not found", install fuse first: `apt-get update && apt-get install -y fuse3`.
- Don't commit `rclone.conf` to git — the OAuth refresh token is sensitive.

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
6. Switch to `colab_gpu_extraction.ipynb` and run all cells — Colab sees the same files via Drive mount.

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
