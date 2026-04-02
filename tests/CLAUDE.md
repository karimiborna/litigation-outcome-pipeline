# Tests

Unit and integration tests mirroring the source module structure.

## Structure

- `unit/` — fast, isolated tests for individual functions and classes
- `integration/` — tests that exercise cross-module interactions (e.g., feature extraction → model prediction)

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
