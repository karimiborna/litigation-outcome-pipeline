# Features Module

LLM-based feature extraction and preprocessing pipeline that converts raw case data into structured ML-ready features.

## Responsibilities

- Use an LLM to extract structured signals from unstructured case text:
  - **Evidence strength** — quality and quantity of supporting evidence
  - **Contract presence** — whether a written contract is involved
  - **Argument clarity** — coherence and specificity of legal arguments
  - Other domain-relevant textual signals
- Combine LLM-extracted features with structured metadata (claim amount, case type, etc.)
- Output a unified feature matrix suitable for model training and inference
- Handle prompt engineering and LLM response parsing

## Key Considerations

- LLM is used strictly for feature extraction, NOT for prediction
- Prompts should produce consistent, parseable structured output (e.g., JSON)
- Feature extraction should be idempotent — same input always produces same features
- Cost and latency of LLM calls need to be managed (batching, caching)
- Feature definitions should be versioned so model training is reproducible
- Extracted features must be logged/stored for audit and debugging
