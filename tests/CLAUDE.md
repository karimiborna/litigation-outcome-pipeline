# Tests Module

110 unit tests covering all pipeline modules. All passing.

## Structure

```
tests/
├── unit/
│   ├── test_api_schemas.py      # Pydantic request/response validation
│   ├── test_cleaning.py         # Text normalization and OCR artifact removal
│   ├── test_counterfactual.py   # Feature perturbation logic
│   ├── test_enumerator.py       # Case number range parsing and probing
│   ├── test_extractor.py        # PDF text extraction (pymupdf + NVIDIA)
│   ├── test_feature_schema.py   # LLMFeatures and FeatureVector models
│   ├── test_manifest.py         # ScrapeManifest persistence
│   ├── test_parser.py           # HTML parsing from court API responses
│   ├── test_prompts.py          # LLM prompt construction
│   ├── test_rate_limiter.py     # Rate limiting enforcement
│   ├── test_schemas.py          # Data schema validation
│   ├── test_session.py          # Session ID handling
│   ├── test_storage.py          # Filesystem read/write helpers
│   └── test_validation.py       # Pydantic validation pipeline
└── integration/                 # Integration tests (run on merge to main)
```

## Running Tests

```bash
# All unit tests
pytest tests/unit/

# Single file
pytest tests/unit/test_cleaning.py -v

# With coverage
pytest tests/unit/ --cov=. --cov-report=html
```

## Key Conventions

- LLM calls are mocked in all unit tests (no real API calls)
- Integration tests hit real services and run only on merge to main (GitHub Actions)
- Fixtures for sample case data live in `tests/unit/conftest.py`
- Tests mirror source structure: `scraper/court_api.py` → `tests/unit/test_parser.py`
