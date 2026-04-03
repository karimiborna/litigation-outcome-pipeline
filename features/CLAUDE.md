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

| Feature | Type | Scale |
|---|---|---|
| evidence_strength | int | 1–5 |
| contract_present | bool | — |
| argument_clarity | int | 1–5 |
| witness_count | int | raw count |
| timeline_clarity | int | 1–5 |
| legal_representation | bool | has attorney |
| counterclaim_present | bool | — |
| claim_category | str | e.g. "property damage" |
| claim_amount | float | dollars |
| ... | | |

Missing info → `null` (never guessed). Nulls → `-1` sentinel when converting to model input.

## Caching

Features are cached by SHA256 hash of the case text. Cache lives in `data/features_cache/`. Re-running on the same case text costs 0 LLM calls.

## LLM Role

The LLM is used **strictly for extraction, not prediction**. It reads case text and fills in a JSON schema. The JSON schema is defined in `schema.py` (LLMFeatures). Temperature is set to 0 for deterministic output.

## Usage

```python
from features.extraction import FeatureExtractor
from features.config import FeaturesConfig

extractor = FeatureExtractor(FeaturesConfig())
vector = extractor.extract(processed_case)
model_input = vector.to_model_input()  # flat dict of floats
```
