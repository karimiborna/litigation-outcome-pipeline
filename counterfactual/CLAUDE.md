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
- Only perturb features that are meaningful and actionable
- Depends on trained models from `models/` — loads from MLflow registry
- Output should be human-readable and useful for non-technical users
- Feature interaction effects may be important — changing one feature may affect interpretation of others
