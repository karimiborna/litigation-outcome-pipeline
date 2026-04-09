# Features Module

LLM-based feature extraction — converts raw case text into structured ML signals.

## File Map

| File | Purpose |
|---|---|
| `extraction.py` | FeatureExtractor class — extract(), extract_batch(), LLM calls, caching |
| `prompts.py` | LLM prompt templates — FEATURE_EXTRACTION_SYSTEM, build_extraction_prompt() |
| `schema.py` | LLMFeatures, FeatureVector, to_model_input() |
| `config.py` | FeaturesConfig (pydantic-settings) — LLM provider, model, caching |

## What Gets Extracted

- LLM is used strictly for feature extraction, NOT for prediction
- Prompts should produce consistent, parseable structured output (e.g., JSON)
- Feature extraction should be idempotent — same input always produces same features
- Cost and latency of LLM calls need to be managed (batching, caching)
- Feature definitions should be versioned so model training is reproducible
- Extracted features must be logged/stored for audit and debugging

## Related modules

- **`labels.py`** — separate LLM pipeline for **supervision**: structured labels (outcome, amounts, attorney flags) from judgment / order text, not the same as training-feature extraction in `extraction.py`.
