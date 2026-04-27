# Tests

Unit tests covering all pipeline modules.

## Structure

```
tests/
├── unit/
│   ├── test_api_app.py          # FastAPI endpoint behavior and health check
│   ├── test_api_schemas.py      # Pydantic request/response validation
│   ├── test_cleaning.py         # Text normalization and OCR artifact removal
│   ├── test_counterfactual.py   # Feature perturbation logic
│   ├── test_dataset.py          # models/dataset.py preprocessing logic
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

## Responsibilities

- Validate data ingestion and schema enforcement (`data/`)
- Test LLM prompt parsing and feature extraction output structure (`features/`)
- Test model training, loading, and prediction interfaces (`models/`)
- Test retrieval index building and query results (`retrieval/`)
- Test counterfactual perturbation logic and output format (`counterfactual/`)
- Test API endpoints, request validation, and error handling (`api/`)

## Key Considerations

- Unit tests should not require LLM API calls — mock the LLM responses
- Integration tests may hit real services — keep them in a separate directory
- Use fixtures for sample case data and expected feature outputs
- CI runs all unit tests on every push; integration tests run on merge to main
- Test data should be small, representative, and committed to the repo
