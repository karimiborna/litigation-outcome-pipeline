# Counterfactual Module

Shows users which case features most improve their win probability ("what-if" analysis).

## File Map

| File | Purpose |
|---|---|
| `analyzer.py` | CounterfactualAnalyzer, CounterfactualResult, FEATURE_CONSTRAINTS |

## How It Works

1. Takes a case's feature vector as baseline
2. Perturbs one feature at a time using `_auto_perturbations()`:
   - Booleans → flipped
   - Integers → +1
   - Floats → ×1.5
3. Re-runs classifier + regressor on each perturbed vector
4. Records delta in win probability and monetary outcome
5. Returns results sorted by impact (largest delta first)

Feature constraints (`FEATURE_CONSTRAINTS`) enforce valid ranges — no impossible values like evidence_strength=6.

## Usage

```python
from counterfactual.analyzer import CounterfactualAnalyzer

analyzer = CounterfactualAnalyzer(classifier, regressor)
results = analyzer.analyze(feature_vector)
# results[0] is the single feature change with biggest win prob improvement
```

## Purpose in Pipeline

Exposed via API `/counterfactual` endpoint. Tells self-represented litigants: "If you had a written contract, your win probability goes from 45% to 72%." This is the key differentiator from pure-RAG competitors.
