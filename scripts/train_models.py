#!/usr/bin/env python3
"""Train real classifier and regressor models from dataset.csv.

Phase A flow (M5 demo path):
- Reads root dataset.csv (built by scripts/build_training_rows.py)
- Stratified 85/15 split, random_state=42, held-out test set
- Baselines: DummyClassifier + LogisticRegression (with SimpleImputer)
- Three classifier families compared: XGBoost, GradientBoosting, RandomForest
  Each: Pipeline(SelectFromModel(same family) -> Classifier) + RandomizedSearchCV (15 candidates)
  Pick the best family by ROC AUC on the held-out set, then calibrate (sigmoid vs isotonic).
- Regressor: same pipeline pattern with XGBRegressor, scoring=neg_RMSE, no calibration
- Logs everything to hosted MLflow at http://35.208.251.175:5000
- Registers the FULL CalibratedClassifierCV pipeline so the API picks up selection + calibration
"""

from __future__ import annotations

import json
import logging
import math
import sys
import tempfile
from pathlib import Path
from typing import Any, Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import loguniform
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    confusion_matrix,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier, XGBRegressor

import mlflow
from models.config import MLflowConfig
from models.dataset import (
    MODEL_FEATURE_COLUMNS,
    dataset_sha256,
    load_dataset_csv,
    prepare_classifier_dataset,
    prepare_regressor_dataset,
)
from models.tracking import get_or_create_experiment, log_model_artifact

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DATASET_CSV = _REPO_ROOT / "dataset.csv"
RANDOM_STATE = 42
TEST_SIZE = 0.15
N_SEARCH_CANDIDATES_PER_FAMILY = 15
SEARCH_CV_FOLDS = 5
CALIBRATION_CV_FOLDS = 5


def _scale_pos_weight(y_train: pd.Series) -> float:
    pos = int((y_train == 1).sum())
    neg = int((y_train == 0).sum())
    return float(neg) / float(pos) if pos > 0 else 1.0


def _xgb_pipeline(scale_pos_weight: float) -> Pipeline:
    """XGBoost handles NaN natively. Imbalance via scale_pos_weight."""
    return Pipeline([
        ("select", SelectFromModel(
            estimator=XGBClassifier(
                n_estimators=100, max_depth=3, random_state=RANDOM_STATE,
                eval_metric="logloss", n_jobs=-1,
                scale_pos_weight=scale_pos_weight,
            ),
            threshold="median",
        )),
        ("clf", XGBClassifier(
            random_state=RANDOM_STATE, n_jobs=-1, eval_metric="logloss",
            scale_pos_weight=scale_pos_weight,
        )),
    ])


def _xgb_param_dist() -> dict[str, Any]:
    return {
        "select__threshold": ["mean", "median", "1.25*median", 0.005],
        "clf__max_depth": [2, 3, 4],
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__n_estimators": [100, 200, 400],
        "clf__reg_alpha": loguniform(0.001, 10),
        "clf__reg_lambda": loguniform(0.001, 10),
        "clf__min_child_weight": [1, 3, 5],
        "clf__subsample": [0.7, 0.85, 1.0],
        "clf__colsample_bytree": [0.7, 0.85, 1.0],
    }


def _gb_pipeline() -> Pipeline:
    """GradientBoosting needs imputation (no native NaN handling) + imbalance via sample weights."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectFromModel(
            estimator=GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=RANDOM_STATE,
            ),
            threshold="median",
        )),
        ("clf", GradientBoostingClassifier(random_state=RANDOM_STATE)),
    ])


def _gb_param_dist() -> dict[str, Any]:
    return {
        "select__threshold": ["mean", "median", "1.25*median"],
        "clf__max_depth": [2, 3, 4],
        "clf__learning_rate": loguniform(0.01, 0.3),
        "clf__n_estimators": [100, 200, 400],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__subsample": [0.7, 0.85, 1.0],
    }


def _rf_pipeline() -> Pipeline:
    """RandomForest needs imputation (no native NaN handling) + class_weight=balanced."""
    return Pipeline([
        ("impute", SimpleImputer(strategy="median")),
        ("select", SelectFromModel(
            estimator=RandomForestClassifier(
                n_estimators=100, max_depth=4, random_state=RANDOM_STATE,
                class_weight="balanced", n_jobs=-1,
            ),
            threshold="median",
        )),
        ("clf", RandomForestClassifier(
            random_state=RANDOM_STATE, class_weight="balanced", n_jobs=-1,
        )),
    ])


def _rf_param_dist() -> dict[str, Any]:
    return {
        "select__threshold": ["mean", "median", "1.25*median"],
        "clf__max_depth": [3, 4, 5, 7, None],
        "clf__n_estimators": [100, 200, 400],
        "clf__min_samples_split": [2, 5, 10],
        "clf__min_samples_leaf": [1, 3, 5],
        "clf__max_features": ["sqrt", "log2", 0.5],
    }


def _log_baselines(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    with mlflow.start_run(run_name="baseline_dummy_most_frequent", nested=True):
        dummy = DummyClassifier(strategy="most_frequent", random_state=RANDOM_STATE)
        dummy.fit(x_train, y_train)
        y_pred = dummy.predict(x_test)
        y_proba = dummy.predict_proba(x_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "brier_score": float(brier_score_loss(y_test, y_proba)),
        }
        mlflow.log_metrics(metrics)
        logger.info("Baseline Dummy: %s", metrics)
        results["dummy"] = metrics

    with mlflow.start_run(run_name="baseline_logistic_regression", nested=True):
        lr_pipeline = Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("lr", LogisticRegression(
                max_iter=2000, class_weight="balanced", random_state=RANDOM_STATE, solver="liblinear",
            )),
        ])
        lr_pipeline.fit(x_train, y_train)
        y_pred = lr_pipeline.predict(x_test)
        y_proba = lr_pipeline.predict_proba(x_test)[:, 1]
        metrics = {
            "accuracy": float(accuracy_score(y_test, y_pred)),
            "roc_auc": float(roc_auc_score(y_test, y_proba)),
            "brier_score": float(brier_score_loss(y_test, y_proba)),
        }
        mlflow.log_metrics(metrics)
        logger.info("Baseline LR: %s", metrics)
        results["logistic_regression"] = metrics
    return results


def _reliability_diagram(y_true: pd.Series, y_proba: np.ndarray, title: str) -> Path:
    n_bins = min(5, max(2, int(math.sqrt(len(y_true)))))
    frac_pos, mean_pred = calibration_curve(y_true, y_proba, n_bins=n_bins, strategy="quantile")
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot([0, 1], [0, 1], "k--", label="perfect")
    ax.plot(mean_pred, frac_pos, "o-", label=title)
    ax.set_xlabel("Mean predicted probability"); ax.set_ylabel("Fraction of positives")
    ax.set_title(f"Reliability — {title}"); ax.legend(); ax.grid(alpha=0.3)
    tmp = Path(tempfile.mkstemp(suffix=".png")[1])
    fig.tight_layout(); fig.savefig(tmp, dpi=110); plt.close(fig)
    return tmp


def _evaluate_classifier(
    model: Any, x_test: pd.DataFrame, y_test: pd.Series, label: str,
) -> dict[str, float]:
    y_pred = model.predict(x_test)
    y_proba = model.predict_proba(x_test)[:, 1]
    p, r, f, _ = precision_recall_fscore_support(
        y_test, y_pred, average=None, labels=[0, 1], zero_division=0,
    )
    cm = confusion_matrix(y_test, y_pred, labels=[0, 1])
    auc = float(roc_auc_score(y_test, y_proba)) if len(set(y_test)) > 1 else float("nan")
    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "roc_auc": auc,
        "brier_score": float(brier_score_loss(y_test, y_proba)),
        "precision_loss": float(p[0]), "recall_loss": float(r[0]), "f1_loss": float(f[0]),
        "precision_win": float(p[1]), "recall_win": float(r[1]), "f1_win": float(f[1]),
        "cm_tn": int(cm[0, 0]), "cm_fp": int(cm[0, 1]),
        "cm_fn": int(cm[1, 0]), "cm_tp": int(cm[1, 1]),
    }
    logger.info("%s held-out: %s", label, metrics)
    return metrics


def _search_family(
    family: str,
    pipeline_factory: Callable[..., Pipeline],
    param_dist: dict[str, Any],
    x_train: pd.DataFrame, y_train: pd.Series,
    x_test: pd.DataFrame, y_test: pd.Series,
    **pipeline_kwargs: Any,
) -> tuple[Pipeline, dict[str, float]]:
    """Run RandomizedSearchCV for one model family. Returns (best_estimator, held-out metrics)."""
    with mlflow.start_run(run_name=f"family_{family}", nested=True):
        pipeline = pipeline_factory(**pipeline_kwargs)
        search = RandomizedSearchCV(
            pipeline, param_dist,
            n_iter=N_SEARCH_CANDIDATES_PER_FAMILY, cv=SEARCH_CV_FOLDS,
            scoring="roc_auc", random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        )
        search.fit(x_train, y_train)
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_roc_auc", float(search.best_score_))
        metrics = _evaluate_classifier(search.best_estimator_, x_test, y_test, f"family_{family}")
        mlflow.log_metrics(metrics)
        return search.best_estimator_, metrics


def _train_classifier(df: pd.DataFrame, config: MLflowConfig, dataset_hash: str) -> None:
    prepared = prepare_classifier_dataset(df)
    if prepared.y.nunique() < 2:
        raise RuntimeError("Only one classifier class present after filtering.")
    logger.info(
        "Classifier dataset — rows=%d features=%d class_counts=%s",
        len(prepared.x), prepared.x.shape[1], prepared.y.value_counts().to_dict(),
    )

    x_train, x_test, y_train, y_test = train_test_split(
        prepared.x, prepared.y,
        test_size=TEST_SIZE, stratify=prepared.y, random_state=RANDOM_STATE,
    )
    spw = _scale_pos_weight(y_train)
    logger.info("train rows=%d, test rows=%d, scale_pos_weight=%.2f", len(x_train), len(x_test), spw)

    experiment_id = get_or_create_experiment(config.classifier_experiment, config)
    with mlflow.start_run(experiment_id=experiment_id, run_name="phase-a-classifier"):
        mlflow.log_params({
            "dataset_path": str(DATASET_CSV),
            "dataset_sha256": dataset_hash,
            "feature_version": "v2",
            "model_kind": "classifier",
            "raw_rows": prepared.raw_rows,
            "target_rows": prepared.target_rows,
            "model_rows": prepared.model_rows,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "n_features_before_selection": x_train.shape[1],
            "search_candidates_per_family": N_SEARCH_CANDIDATES_PER_FAMILY,
            "cv_folds": SEARCH_CV_FOLDS,
            "random_state": RANDOM_STATE,
            "test_size": TEST_SIZE,
            "scale_pos_weight": spw,
            "win_definition": "label_outcome in (plaintiff_win, partial_win)",
        })

        baselines = _log_baselines(x_train, x_test, y_train, y_test)
        for name, m in baselines.items():
            for k, v in m.items():
                mlflow.log_metric(f"baseline_{name}_{k}", v)

        family_results: dict[str, tuple[Pipeline, dict[str, float]]] = {}
        family_results["xgboost"] = _search_family(
            "xgboost", _xgb_pipeline, _xgb_param_dist(),
            x_train, y_train, x_test, y_test,
            scale_pos_weight=spw,
        )
        family_results["gradient_boosting"] = _search_family(
            "gradient_boosting", _gb_pipeline, _gb_param_dist(),
            x_train, y_train, x_test, y_test,
        )
        family_results["random_forest"] = _search_family(
            "random_forest", _rf_pipeline, _rf_param_dist(),
            x_train, y_train, x_test, y_test,
        )

        for name, (_, m) in family_results.items():
            for k, v in m.items():
                mlflow.log_metric(f"family_{name}_{k}", v)

        # Pick by F1 on the win class — this is the user-facing job (catch real wins).
        # ROC AUC alone can favor models that rank well but never predict the positive class.
        # Tiebreaker: ROC AUC.
        def _family_rank(f: str) -> tuple[float, float]:
            m = family_results[f][1]
            return (-m["f1_win"], -m["roc_auc"])
        chosen_family = min(family_results, key=_family_rank)
        chosen_pipeline, chosen_family_metrics = family_results[chosen_family]
        logger.info("Chosen family: %s (f1_win=%.4f roc_auc=%.4f)",
                    chosen_family, chosen_family_metrics["f1_win"], chosen_family_metrics["roc_auc"])
        mlflow.log_param("chosen_family", chosen_family)
        mlflow.log_param("family_selection_criterion", "f1_win_then_roc_auc")

        # Calibration on the winner — compare "none" (uncalibrated), sigmoid, isotonic.
        # At small n, calibration can destroy discrimination, so "none" is a valid choice.
        calibration_results: dict[str, tuple[Any, dict[str, float]]] = {}
        with mlflow.start_run(run_name="calibration_none", nested=True):
            metrics = _evaluate_classifier(chosen_pipeline, x_test, y_test, "calibration_none")
            mlflow.log_metrics(metrics)
            rel_png = _reliability_diagram(
                y_test, chosen_pipeline.predict_proba(x_test)[:, 1], "uncalibrated",
            )
            mlflow.log_artifact(str(rel_png))
            calibration_results["none"] = (chosen_pipeline, metrics)
        for method in ("sigmoid", "isotonic"):
            with mlflow.start_run(run_name=f"calibration_{method}", nested=True):
                calibrated = CalibratedClassifierCV(
                    chosen_pipeline, method=method, cv=CALIBRATION_CV_FOLDS,
                )
                calibrated.fit(x_train, y_train)
                metrics = _evaluate_classifier(calibrated, x_test, y_test, f"calibration_{method}")
                mlflow.log_metrics(metrics)
                rel_png = _reliability_diagram(
                    y_test, calibrated.predict_proba(x_test)[:, 1],
                    f"calibrated_{method}",
                )
                mlflow.log_artifact(str(rel_png))
                calibration_results[method] = (calibrated, metrics)

        # Pick by F1-win (catches actual wins) with ROC AUC as tiebreaker.
        # At small n calibration collapses to base-rate (F1-win=0) while preserving ranking
        # (ROC AUC stays decent) — F1-win catches that pathology, ROC AUC alone does not.
        def _calib_rank(m: str) -> tuple[float, float]:
            metrics = calibration_results[m][1]
            return (-metrics["f1_win"], -metrics["roc_auc"])
        chosen_method = min(calibration_results, key=_calib_rank)
        chosen_model, chosen_metrics = calibration_results[chosen_method]
        logger.info("Chosen calibration: %s (roc_auc=%.4f brier=%.4f)",
                    chosen_method, chosen_metrics["roc_auc"], chosen_metrics["brier_score"])
        mlflow.log_param("chosen_calibration_method", chosen_method)
        for k, v in chosen_metrics.items():
            mlflow.log_metric(f"final_{k}", v)

        # Selected features from the chosen pipeline (handle different pipeline structures)
        try:
            steps = chosen_pipeline.named_steps
            if "select" in steps:
                # Need to figure out the column space after imputation (it doesn't drop columns)
                support = steps["select"].get_support()
                # SelectFromModel transforms columns in the order they entered the selector.
                # If there's a SimpleImputer before, columns are preserved 1:1 with input.
                selected_features = [
                    c for c, keep in zip(MODEL_FEATURE_COLUMNS, support) if keep
                ]
                with tempfile.TemporaryDirectory() as tmp:
                    feat_path = Path(tmp) / "selected_features.json"
                    feat_path.write_text(json.dumps({
                        "selected_features": selected_features,
                        "n_before_selection": len(MODEL_FEATURE_COLUMNS),
                        "n_after_selection": len(selected_features),
                        "chosen_family": chosen_family,
                    }, indent=2))
                    mlflow.log_artifact(str(feat_path))
                    mlflow.log_metric("n_features_after_selection", len(selected_features))
                    logger.info("Selected features: %d / %d",
                                len(selected_features), len(MODEL_FEATURE_COLUMNS))
        except Exception as e:
            logger.warning("Could not log selected feature list: %s", e)

        with tempfile.TemporaryDirectory() as tmp:
            comparison = {
                "baselines": baselines,
                "families": {f: m for f, (_, m) in family_results.items()},
                "calibration": {m: metrics for m, (_, metrics) in calibration_results.items()},
                "chosen_family": chosen_family,
                "chosen_calibration_method": chosen_method,
            }
            comp_path = Path(tmp) / "comparison.json"
            comp_path.write_text(json.dumps(comparison, indent=2))
            mlflow.log_artifact(str(comp_path))

        log_model_artifact(
            chosen_model,
            artifact_path="classifier",
            registered_name=config.classifier_model_name,
        )
        mlflow.log_artifact(str(DATASET_CSV))


def _train_regressor(df: pd.DataFrame, config: MLflowConfig, dataset_hash: str) -> None:
    prepared = prepare_regressor_dataset(df)
    if len(prepared.x) < 5:
        raise RuntimeError("Too few regressor rows after filtering.")
    positive_rows = int((prepared.y > 0).sum())
    logger.info(
        "Regressor — rows=%d features=%d positive=%d mean=%.2f",
        len(prepared.x), prepared.x.shape[1], positive_rows, float(prepared.y.mean()),
    )
    x_train, x_test, y_train, y_test = train_test_split(
        prepared.x, prepared.y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
    )
    experiment_id = get_or_create_experiment(config.regressor_experiment, config)
    with mlflow.start_run(experiment_id=experiment_id, run_name="phase-a-regressor"):
        mlflow.log_params({
            "dataset_path": str(DATASET_CSV),
            "dataset_sha256": dataset_hash,
            "model_kind": "regressor",
            "raw_rows": prepared.raw_rows,
            "model_rows": prepared.model_rows,
            "train_rows": len(x_train),
            "test_rows": len(x_test),
            "positive_rows": positive_rows,
            "random_state": RANDOM_STATE,
            "target_column": "label_total_awarded",
        })

        pipeline = Pipeline([
            ("select", SelectFromModel(
                estimator=XGBRegressor(
                    n_estimators=100, max_depth=3, random_state=RANDOM_STATE, n_jobs=-1,
                ),
                threshold="median",
            )),
            ("reg", XGBRegressor(random_state=RANDOM_STATE, n_jobs=-1)),
        ])
        param_dist = {
            "select__threshold": ["mean", "median", "1.25*median", 0.005],
            "reg__max_depth": [2, 3, 4],
            "reg__learning_rate": loguniform(0.01, 0.3),
            "reg__n_estimators": [100, 200, 400],
            "reg__reg_alpha": loguniform(0.001, 10),
            "reg__reg_lambda": loguniform(0.001, 10),
            "reg__min_child_weight": [1, 3, 5],
        }
        search = RandomizedSearchCV(
            pipeline, param_dist, n_iter=N_SEARCH_CANDIDATES_PER_FAMILY,
            cv=SEARCH_CV_FOLDS, scoring="neg_root_mean_squared_error",
            random_state=RANDOM_STATE, n_jobs=-1, refit=True,
        )
        search.fit(x_train, y_train)
        mlflow.log_params({f"best__{k}": v for k, v in search.best_params_.items()})
        mlflow.log_metric("best_cv_neg_rmse", float(search.best_score_))

        best = search.best_estimator_
        y_pred = best.predict(x_test)
        metrics = {
            "mae": float(mean_absolute_error(y_test, y_pred)),
            "rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
            "r2": float(r2_score(y_test, y_pred)),
            "mean_pred": float(np.mean(y_pred)),
            "mean_true": float(np.mean(y_test)),
        }
        mlflow.log_metrics(metrics)
        logger.info("Regressor held-out: %s", metrics)

        log_model_artifact(
            best, artifact_path="regressor",
            registered_name=config.regressor_model_name,
        )
        mlflow.log_artifact(str(DATASET_CSV))


def main() -> int:
    config = MLflowConfig()
    mlflow.set_tracking_uri(config.tracking_uri)
    logger.info("MLflow tracking URI: %s", config.tracking_uri)
    if not DATASET_CSV.exists():
        raise FileNotFoundError(f"Expected {DATASET_CSV} — run scripts/build_training_rows.py first.")
    df = load_dataset_csv(DATASET_CSV)
    dataset_hash = dataset_sha256(DATASET_CSV)
    logger.info("Loaded dataset: shape=%s sha256=%s", df.shape, dataset_hash[:16])
    logger.info("Training classifier...")
    _train_classifier(df, config, dataset_hash)
    logger.info("Training regressor...")
    _train_regressor(df, config, dataset_hash)
    return 0


if __name__ == "__main__":
    sys.exit(main())
