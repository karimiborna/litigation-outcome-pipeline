# Data Module

Handles ingestion, storage, validation, and preprocessing of small claims court case data.

## Structure

- `raw/` — original ingested data (never modified after landing)
- `processed/` — cleaned and transformed data ready for feature extraction
- `schemas/` — data validation schemas (e.g., JSON Schema, Pydantic models)

## Responsibilities

- Ingest case data from source (cloud storage: S3 or GCS)
- Validate incoming data against defined schemas
- Clean and normalize text fields (case descriptions, rulings)
- Produce structured records with metadata (claim amount, case type, dates) and raw text fields
- Maintain separation between raw and processed data stages

## Key Considerations

- Raw data is immutable — all transformations produce new files in `processed/`
- Schema validation should catch malformed records before they enter the pipeline
- Text fields need consistent encoding and normalization before passing to feature extraction
- Personally identifiable information (PII) in case text may need redaction
- Data versioning should be considered for reproducibility
