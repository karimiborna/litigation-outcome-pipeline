 
# Deep Dive: Model Training Pipeline

**Files covered:**
- `scripts/train_classifier_real.py`
- `models/dataset.py`
- `models/trainer.py`

---

## Overview

This pipeline takes a `dataset.csv` of real small claims court cases and trains two models:
1. **Classifier** — predicts win/loss (binary)
2. **Regressor** — predicts dollar amount awarded (continuous)

Both models are tracked in MLflow and promoted to Production so the API can load them at startup.

---

## Code Walkthrough

### `models/dataset.py` — Data Preprocessing

#### The `PreparedDataset` dataclass

```python
@dataclass(frozen=True)
class PreparedDataset:
    x: pd.DataFrame
    y: pd.Series
    raw_rows: int
    target_rows: int
    model_rows: int
    ...
```

**What:** A frozen dataclass that bundles the feature matrix (`x`), target (`y`), and audit metadata.

**Why `frozen=True`:** Prevents accidental mutation after creation. Once you've prepared your dataset, nothing should change it. This is a safety rail — if code tries to reassign a field, Python raises a `FrozenInstanceError` immediately instead of silently corrupting your training data.

**Why bundle audit metadata:** `raw_rows` vs `target_rows` vs `model_rows` tells you exactly how many rows were dropped at each stage (missing labels, missing features). This makes debugging data quality issues much easier — you can see "I started with 500 rows but only 312 made it to training" without digging through logs.

---

#### One-hot encoding `claim_category`

```python
category = pd.Categorical(
    selected["feat_claim_category"],
    categories=list(CLAIM_CATEGORIES),
)
category_dummies = pd.get_dummies(category, prefix="feat_claim_category", dtype=float)
```

**What:** Converts a text column like `"security_deposit"` into 8 binary columns: `feat_claim_category_security_deposit=1`, all others `=0`.

**Why:** Machine learning models are math. They can't compute "security_deposit > property_damage". By converting categories to separate 0/1 columns, the model can learn that security deposit cases have a different win rate than property damage cases — independently, without assuming any ordering.

**Why `pd.Categorical` with explicit `categories` list first:** This guarantees the same 8 columns always appear in the same order, even if a category is missing from a batch of data. Without this, `pd.get_dummies` on a dataset that happens to have no fraud cases would silently produce 7 columns instead of 8 — your model's feature count would be wrong at inference time.

**Alternatives:**
- `OrdinalEncoder` — assigns integers (1, 2, 3...). Bad here because it implies `fraud > breach_of_contract` numerically, which is meaningless.
- `TargetEncoder` — replaces category with its mean target value. Can be powerful but leaks label info; needs careful cross-validation.

---

#### Dropping rows with missing features

```python
selected = selected.dropna(axis=0, how="any")
```

**What:** Removes any row that has a `null` in any of the 28 feature columns.

**Why no imputation:** The project explicitly chose to not impute (fill in missing values). This is the more conservative choice — a null in `has_photos_or_physical_evidence` means the LLM couldn't determine this from the text, not that the answer is `False`. Filling it with `False` or the column mean would inject false signal. Dropping the row loses data but keeps what remains trustworthy.

**The tradeoff:** You lose rows. If 30% of your data has missing features, your effective training set shrinks by 30%. The alternative (imputation) keeps rows but risks training the model on made-up values.

---

#### Dataset SHA-256

```python
def dataset_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
```

**What:** Computes a fingerprint of the CSV file and logs it to MLflow alongside each run.

**Why:** If you retrain 3 months later and the metrics change, you can check whether you used the same data. This is "data versioning lite" — not a full DVC setup, just a cheap reproducibility anchor. If the hash matches a previous run, the data was identical.

---

### `models/trainer.py` — Model Training

#### Gradient Boosting

```python
self._model = GradientBoostingClassifier(
    n_estimators=200, max_depth=5, learning_rate=0.1
)
```

**What:** An ensemble of 200 shallow decision trees built sequentially. Each tree corrects the errors of the previous one ("boosting").

**Why Gradient Boosting over other options:**
| Model | Why not |
|---|---|
| Logistic Regression | Too simple; can't capture feature interactions (e.g., having a contract AND a demand letter together matters more than either alone) |
| Random Forest | Good alternative, but GBM typically wins on tabular data with ~35 features |
| Neural Network | Overkill for 35 features; needs far more data to beat GBM; harder to interpret |
| GBM | Strong on tabular data, handles mixed feature types, gives feature importances |

**Key hyperparameters:**
- `n_estimators=200` — number of trees. More = better fit, but slower and risks overfitting.
- `max_depth=5` — how deep each tree goes. Shallow trees reduce overfitting.
- `learning_rate=0.1` — how much each new tree contributes. Lower = more trees needed, but more robust. The "shrinkage" parameter.

---

#### Stratified train/test split (classifier only)

```python
x_train, x_test, y_train, y_test = train_test_split(
    features, labels, test_size=0.2, random_state=42, stratify=labels
)
```

**What:** Splits 80% train / 20% test, with `stratify=labels` ensuring the win/loss ratio is the same in both splits.

**Why `stratify`:** If 70% of cases are losses, without stratification a random split might put 90% losses in the test set by chance. The model's test metrics would be misleading. `stratify` enforces that both splits reflect the real class distribution.

**Why the regressor doesn't stratify:** `stratify` only works for classification targets. For continuous dollar amounts you can't stratify — you'd use `KFold` or `ShuffleSplit` if you wanted more robust validation.

**Why `random_state=42`:** Any fixed integer works. The point is reproducibility — running the script twice gives the same split. `42` is a convention (Hitchhiker's Guide joke), not magic.

---

#### Feature importance logging

```python
importances = dict(zip(features.columns, self._model.feature_importances_))
top_features = sorted(importances.items(), key=lambda x: x[1], reverse=True)[:10]
for feat_name, importance in top_features:
    mlflow.log_metric(f"importance_{feat_name}", importance)
```

**What:** After training, logs the top 10 most predictive features to MLflow as individual metrics.

**Why:** Feature importance tells you which case signals the model relies on most. If `feat_has_written_communications` is the top feature, that's something you can communicate to users ("get your emails in order"). It also catches data leakage — if some unexpected feature dominates, something may be wrong.

**What GBM feature importance measures:** How much each feature reduces the loss function across all 200 trees. Higher = more splits were made on this feature = it's more predictive. This is "impurity-based" importance and can be biased toward high-cardinality features; for more rigorous importance you'd use SHAP values.

---

### `scripts/train_classifier_real.py` — Orchestration

#### MLflow model registration

```python
log_model_artifact(
    self._model,
    artifact_path="classifier",
    registered_name=self._config.classifier_model_name,
)
```

**What:** Saves the trained sklearn model to MLflow and registers it in the Model Registry under a versioned name like `litigation-win-classifier`.

**Why a registry:** Separates "this run produced a model" from "this model is production-ready". A model can be trained and registered but stay in `None` / `Staging` stage. Only after `promote_models_to_production.py` runs does the API see it. This prevents accidentally serving an undertrained model.

**The full lifecycle:**
```
train → registered (version 1, stage=None)
      → promote → stage=Production
      → API loads models:/litigation-win-classifier/Production
```

---

## Concepts Explained

### Gradient Boosting — the core idea

Imagine you're trying to predict who wins in court. You start with a dumb baseline (everyone wins 60% of the time). Then you build a small tree that identifies which cases the baseline got most wrong. Then another tree that fixes *those* errors. Repeat 200 times. Each tree is weak on its own, but the ensemble is strong.

This is the "boosting" in gradient boosting — you're boosting performance by focusing on failures.

### The Bias-Variance Tradeoff

- `max_depth` too high → model memorizes training data (low bias, high variance → bad on new cases)
- `max_depth` too low → model can't capture real patterns (high bias, low variance → bad everywhere)
- `n_estimators` too low → underfitting; too high → overfitting + slow

The current values (depth=5, 200 trees, lr=0.1) are reasonable defaults for a dataset of a few hundred to a few thousand rows.

### MLflow Experiment Tracking

Every training run logs: parameters, metrics, artifacts (the model file, the dataset, the feature column list). This means you can compare run #5 (200 trees) vs run #6 (500 trees) in the MLflow UI and see which had better AUC — without keeping notes manually.

Think of it as "git for model experiments".

---

## Learning Resources

**Gradient Boosting:**
- [Scikit-learn GBM user guide](https://scikit-learn.org/stable/modules/ensemble.html#gradient-boosting) — official docs with math
- [StatQuest: Gradient Boost](https://www.youtube.com/watch?v=3CC4N4z3GJc) — best visual explanation on YouTube
- [XGBoost paper](https://arxiv.org/abs/1603.02754) — the production-grade version of what's used here

**MLflow:**
- [MLflow Tracking quickstart](https://mlflow.org/docs/latest/tracking.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)

**Feature Engineering:**
- [Scikit-learn preprocessing guide](https://scikit-learn.org/stable/modules/preprocessing.html) — covers one-hot encoding, scaling, imputation
- [SHAP library](https://shap.readthedocs.io/) — better feature importance than impurity-based; next step for this project

**Train/Test Split:**
- [Scikit-learn cross-validation](https://scikit-learn.org/stable/modules/cross_validation.html) — goes beyond a single split

---

## Related Code

- [features/schema.py](../litigation-outcome-pipeline/features/schema.py) — defines `FeatureVector`, the input shape the LLM produces
- [models/tracking.py](../litigation-outcome-pipeline/models/tracking.py) — MLflow helper functions called by the trainers
- [models/config.py](../litigation-outcome-pipeline/models/config.py) — model names, experiment names, tracking URI
- [api/dependencies.py](../litigation-outcome-pipeline/api/dependencies.py) — loads the Production models at server startup
- [scripts/promote_models_to_production.py](../litigation-outcome-pipeline/scripts/promote_models_to_production.py) — moves registered models to Production stage
