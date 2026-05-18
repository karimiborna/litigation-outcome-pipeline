#!/usr/bin/env python3
"""Join extracted features with labels and report trainable coverage.

Inputs (Drive Desktop sync — single source of truth across team):
- {DRIVE}/labels.json
- {DRIVE}/features_cache/*.json

Outputs:
- data/processed/training_rows.jsonl (unprefixed, for inspection)
- data/processed/training_rows.csv (unprefixed, for inspection)
- dataset.csv (root, prefixed feat_*/label_* columns — what train_models.py reads)
"""

from __future__ import annotations

import csv
import json
import logging
import sys
from collections.abc import Iterable
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from features.schema import FeatureVector
from models.dataset import feature_vector_to_raw_row

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _find_drive_dir() -> Path:
    """Locate the team's Drive Desktop sync — works for any team member."""
    cloud = Path.home() / "Library/CloudStorage"
    for gd in cloud.glob("GoogleDrive-*/My Drive/litigation-pipeline"):
        return gd
    raise FileNotFoundError(
        "litigation-pipeline not found in Drive Desktop sync — "
        "install Google Drive for Desktop and set the team folder to Available offline."
    )


DRIVE_DIR = _find_drive_dir()
LABELS_PATH = DRIVE_DIR / "labels.json"
FEATURES_DIR = DRIVE_DIR / "features_cache"
OUT_JSONL = _REPO_ROOT / "data/processed/training_rows.jsonl"
OUT_CSV = _REPO_ROOT / "data/processed/training_rows.csv"
DATASET_CSV = _REPO_ROOT / "dataset.csv"


def _load_labels(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be an object keyed by case_number")
    return raw


def _load_features(features_dir: Path) -> dict[str, dict]:
    """Load latest feature vector per case (collapses multiple cache hashes to one)."""
    by_case: dict[str, dict] = {}
    by_case_mtime: dict[str, float] = {}
    for fp in sorted(features_dir.glob("*.json")):
        try:
            feature_row = json.loads(fp.read_text(encoding="utf-8"))
        except Exception:
            logger.warning("Skipping unreadable feature file: %s", fp.name)
            continue
        case_number = feature_row.get("case_number")
        if not isinstance(case_number, str) or not case_number:
            logger.warning("Skipping feature file missing case_number: %s", fp.name)
            continue
        mtime = fp.stat().st_mtime
        if case_number not in by_case or mtime > by_case_mtime[case_number]:
            by_case[case_number] = feature_row
            by_case_mtime[case_number] = mtime
    return by_case


def _build_rows(labels_by_case: dict[str, dict], features_by_case: dict[str, dict]) -> list[dict]:
    """Build the unprefixed training_rows.csv format (legacy inspection artifact)."""
    rows: list[dict] = []
    for case_number in sorted(labels_by_case):
        feature_row = dict(features_by_case.get(case_number, {}))
        feature_row["missing_features"] = case_number not in features_by_case
        row = {
            "case_number": case_number,
            **feature_row,
            **labels_by_case[case_number],
        }
        rows.append(row)
    return rows


def _build_trainable_rows(
    labels_by_case: dict[str, dict], features_by_case: dict[str, dict]
) -> list[dict]:
    """Build the prefixed dataset.csv format that train_models.py expects.

    Uses feature_vector_to_raw_row so training and inference produce identical
    feature shapes. Computes label_total_awarded as sum of principal+costs+interest
    since LabelExtractor doesn't populate total_awarded directly.
    """
    rows: list[dict] = []
    skipped_no_features = 0
    skipped_validation = 0
    for case_number in sorted(labels_by_case):
        feature_dict = features_by_case.get(case_number)
        if not feature_dict:
            skipped_no_features += 1
            continue
        try:
            fv = FeatureVector.model_validate(feature_dict)
        except Exception as e:
            logger.warning("Skipping %s — FeatureVector validation failed: %s", case_number, e)
            skipped_validation += 1
            continue

        feat_columns = feature_vector_to_raw_row(fv)
        label = labels_by_case[case_number]
        principal = label.get("amount_awarded_principal") or 0.0
        costs = label.get("amount_awarded_costs") or 0.0
        interest = label.get("amount_awarded_interest") or 0.0
        total_awarded = float(principal) + float(costs) + float(interest)

        row = {
            "case_number": case_number,
            "feat_feature_version": fv.feature_version,
            **feat_columns,
            "label_outcome": label.get("outcome"),
            "label_dismissal_type": label.get("dismissal_type"),
            "label_amount_awarded_principal": label.get("amount_awarded_principal"),
            "label_amount_awarded_costs": label.get("amount_awarded_costs"),
            "label_amount_awarded_interest": label.get("amount_awarded_interest"),
            "label_total_awarded": total_awarded,
            "label_defendant_appeared": label.get("defendant_appeared"),
            "label_has_attorney_plaintiff": label.get("has_attorney_plaintiff"),
            "label_has_attorney_defendant": label.get("has_attorney_defendant"),
            "label_judgment_date": label.get("judgment_date"),
        }
        rows.append(row)

    logger.info("Trainable rows: %d (skipped %d w/o features, %d failed validation)",
                len(rows), skipped_no_features, skipped_validation)
    return rows


def _write_jsonl(rows: Iterable[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def _write_csv(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = sorted({k for row in rows for k in row.keys()})
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    logger.info("Drive source: %s", DRIVE_DIR)
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")
    if not FEATURES_DIR.exists():
        raise FileNotFoundError(f"Missing features directory: {FEATURES_DIR}")

    labels_by_case = _load_labels(LABELS_PATH)
    features_by_case = _load_features(FEATURES_DIR)

    legacy_rows = _build_rows(labels_by_case, features_by_case)
    trainable_rows = _build_trainable_rows(labels_by_case, features_by_case)

    label_cases = set(labels_by_case)
    feature_cases = set(features_by_case)
    joined_cases = set(row["case_number"] for row in legacy_rows)
    rows_missing_features = sum(1 for row in legacy_rows if row.get("missing_features"))

    _write_jsonl(legacy_rows, OUT_JSONL)
    _write_csv(legacy_rows, OUT_CSV)
    _write_csv(trainable_rows, DATASET_CSV)

    labels_with_outcome = sum(1 for v in labels_by_case.values() if v.get("outcome") is not None)
    joined_with_outcome = sum(1 for row in legacy_rows if row.get("outcome") is not None)
    trainable_with_outcome = sum(1 for r in trainable_rows if r.get("label_outcome") is not None)

    logger.info("Labels: %d", len(label_cases))
    logger.info("Features (distinct cases): %d", len(feature_cases))
    logger.info("Legacy joined rows: %d", len(joined_cases))
    logger.info("Trainable dataset rows: %d", len(trainable_rows))
    logger.info("Labels missing features: %d", len(label_cases - feature_cases))
    logger.info("Features missing labels: %d", len(feature_cases - label_cases))
    logger.info("Rows flagged missing_features=True (legacy): %d", rows_missing_features)
    logger.info("Labels with non-null outcome: %d", labels_with_outcome)
    logger.info("Joined rows with non-null outcome: %d", joined_with_outcome)
    logger.info("Trainable rows with non-null outcome: %d", trainable_with_outcome)
    logger.info("Wrote: %s (legacy unprefixed)", OUT_JSONL)
    logger.info("Wrote: %s (legacy unprefixed)", OUT_CSV)
    logger.info("Wrote: %s (trainable feat_*/label_* prefixed)", DATASET_CSV)


if __name__ == "__main__":
    main()
