"""Pre-promotion / startup validation for the classifier and regressor.

Both `scripts/promote_models_to_production.py` (gate) and
`api/dependencies.py:load_models` (startup smoke test) call into this module so
they enforce identical contracts. If the API can load a model that the
promotion script would have rejected, the gate is meaningless.
"""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from models.dataset import MODEL_FEATURE_COLUMNS


class ModelValidationError(RuntimeError):
    """Raised when a model fails a pre-promotion or startup check."""


def _check_feature_columns(model: Any, label: str) -> None:
    if not hasattr(model, "feature_names_in_"):
        raise ModelValidationError(
            f"{label} has no feature_names_in_ — was it fit on a DataFrame? "
            "Train via scripts/train_models.py to ensure column names are recorded."
        )
    expected = set(MODEL_FEATURE_COLUMNS)
    actual = set(model.feature_names_in_)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise ModelValidationError(
            f"{label} feature_names_in_ does not match MODEL_FEATURE_COLUMNS. "
            f"Missing from model: {missing or '(none)'}. "
            f"Extra in model: {extra or '(none)'}."
        )


def validate_classifier(model: Any) -> None:
    if not hasattr(model, "predict_proba"):
        raise ModelValidationError("classifier has no predict_proba — wrong model type?")
    _check_feature_columns(model, "classifier")
    classes = list(getattr(model, "classes_", []))
    if classes != [0, 1]:
        raise ModelValidationError(
            f"classifier classes_ must be [0, 1] (got {classes}). "
            "predict_proba(...)[:, 1] is hard-coded to mean 'win probability'; "
            "any other order silently inverts predictions."
        )


def validate_regressor(model: Any) -> None:
    if not hasattr(model, "predict"):
        raise ModelValidationError("regressor has no predict — wrong model type?")
    _check_feature_columns(model, "regressor")


def smoke_predict(classifier: Any, regressor: Any) -> None:
    """Run a canned prediction; raise if outputs are out-of-shape or non-finite."""
    row = pd.DataFrame([{c: 0.0 for c in MODEL_FEATURE_COLUMNS}])[list(MODEL_FEATURE_COLUMNS)]

    proba = classifier.predict_proba(row)
    if proba.shape != (1, 2):
        raise ModelValidationError(
            f"classifier.predict_proba shape was {proba.shape}, expected (1, 2)"
        )
    win_prob = float(proba[0, 1])
    if math.isnan(win_prob) or not (0.0 <= win_prob <= 1.0):
        raise ModelValidationError(
            f"classifier returned out-of-range win probability: {win_prob}"
        )

    monetary = float(regressor.predict(row)[0])
    if not math.isfinite(monetary):
        raise ModelValidationError(f"regressor returned non-finite value: {monetary}")


def validate_all(classifier: Any, regressor: Any) -> None:
    """Run every check. Raises ModelValidationError on first failure."""
    validate_classifier(classifier)
    validate_regressor(regressor)
    smoke_predict(classifier, regressor)
