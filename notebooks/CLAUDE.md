# notebooks/

Colab notebooks for data pipeline stages that require GPU or external API access.

## colab_gpu_extraction.ipynb

End-to-end pipeline notebook designed for Google Colab (T4 GPU runtime). Handles:

1. **PDF download** from SF Superior Court API (`webapps.sftc.org`) using session IDs
2. **Text extraction** via tiered strategy:
   - PyMuPDF (free, instant) for PDFs with selectable text
   - Qwen2-VL-7B on GPU for scanned pages (currently commented out)
   - NVIDIA Llama 3.2 90B Vision API as fallback (500 calls/day/key)
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
- Outputs: `.txt` files in `extracted/` (or `extracted_nvidia/`), per-case JSONs in `labels/`, merged `labels.json`
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

- `pymupdf` (fitz), `requests`, `openai` (used for both NVIDIA and OpenAI APIs)
- GPU path (commented out): `transformers`, `accelerate`, `bitsandbytes`, `qwen-vl-utils`
- Repo must be pip-installable (`pip install -e .`) for `features.labels` import in label extraction step

### Known issues

- GPU extraction cells (Qwen2-VL) are fully commented out — NVIDIA API is the active path
- Prompts are duplicated between GPU and NVIDIA cells (GENERIC_PROMPT vs GENERIC_PROMPT_NV)
- Final zip download assumes Colab environment (`google.colab.files`)
