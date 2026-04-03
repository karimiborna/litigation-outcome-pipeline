# Data Module

Landing zone and preprocessing for all case data.

## Structure

```
data/raw/<case_number>/
    metadata.json      # CaseMetadata schema
    *.pdf              # raw court PDFs (immutable)

data/processed/<case_number>/
    *.txt              # extracted text (ExtractedText schema)

data/features_cache/   # SHA256-keyed feature extraction cache
data/retrieval_index/  # FAISS index + metadata store
data/scraped_dates.txt # scraper resume log
data/manifest.json     # full ScrapeManifest
```

## File Map

| File | Purpose |
|---|---|
| `schemas/case.py` | Pydantic models — Party, Attorney, CaseMetadata, ExtractedText, ProcessedCase |
| `cleaning.py` | Text preprocessing — normalize_unicode, remove_ocr_artifacts, collapse_whitespace |
| `storage.py` | Filesystem helpers — save/load metadata, PDFs, extracted text |
| `validation.py` | Schema validation — ValidationResult, validate_case_metadata, load_and_validate |

## Key Schema: ProcessedCase

The `ProcessedCase` model is what flows into the features pipeline. It combines:
- `CaseMetadata` (case number, title, parties, document list)
- All extracted text from PDFs (joined across pages)

## Key Rules

- **Raw data is immutable** — `data/raw/` is never modified after landing
- All transformations produce new files in `data/processed/`
- Pydantic validation runs at every stage boundary
- PII (party names, addresses) is preserved in raw — redaction decisions deferred to features pipeline

## Cleaning Pipeline

`clean_extracted_text()` runs:
1. NFKC unicode normalization
2. OCR artifact removal (stray characters, scanner noise)
3. Court stamp / header-footer removal
4. Whitespace collapse
5. Page break normalization
