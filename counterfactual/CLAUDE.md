# Counterfactual Analysis Module

Simulates changes in key features to show how they would affect predicted outcomes.

## Responsibilities

- Accept a case's feature vector and a set of feature perturbations
- Run the trained model(s) on the modified feature vectors
- Compare original vs. counterfactual predictions
- Present results as "if X were different, the outcome would change by Y"
- Support both classification (win probability shift) and regression (monetary change) counterfactuals

## Key Considerations

- Perturbations must respect feature constraints (e.g., claim amount can't be negative)
- Only perturb features that are meaningful and actionable. v2 perturbable set is defined in `FEATURE_CONSTRAINTS` in `analyzer.py` and focuses on evidence-existence booleans (`has_*`), argument-content booleans (`argument_*`), procedural booleans (`sent_*` / `attempted_*` / `gave_*`), representation booleans (`user_has_attorney`, `opposing_party_has_attorney`), and the two numerics (`monetary_amount_claimed`, `witness_count`). Fields that aren't meaningfully changeable by the user (contract detail — only meaningful if a contract exists; damages breakdown — describes the claim; jurisdictional; counts; `user_is_plaintiff`) are intentionally excluded.
- Depends on trained models from `models/` — loads from MLflow registry
- Output should be human-readable and useful for non-technical users
- Feature interaction effects may be important — changing one feature may affect interpretation of others
- Bool perturbations flip 0↔1; numeric perturbations step by 1 (int) or ×1.5 (float) within the declared min/max
