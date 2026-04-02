# Models Module

Classification and regression model training for predicting case outcomes.

## Responsibilities

- Train a **classification model** to predict plaintiff win probability
- Train a **regression model** to predict expected monetary outcome
- Consume the unified feature matrix produced by the features module
- Hyperparameter tuning and model selection
- Log all experiments, metrics, and artifacts to MLflow
- Export trained models for registration in the MLflow model registry

## Key Considerations

- Two distinct prediction targets: binary outcome (win/lose) and continuous outcome (monetary amount)
- Models should be compared across experiments using MLflow tracking
- Feature importance / interpretability is important for downstream explanation generation
- Train/validation/test splits must be consistent and reproducible
- Model artifacts are never committed to git — always tracked via MLflow
