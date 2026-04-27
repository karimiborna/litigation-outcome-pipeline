#!/usr/bin/env python3
"""Join extracted features with labels and report trainable coverage.

Inputs:
- data/processed/labels.json
- data/features_cache/*.json

Outputs:
- data/processed/training_rows.jsonl
- data/processed/training_rows.csv
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

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

LABELS_PATH = Path("data/processed/labels.json")
FEATURES_DIR = Path("data/features_cache")
OUT_JSONL = Path("data/processed/training_rows.jsonl")
OUT_CSV = Path("data/processed/training_rows.csv")


def _load_labels(path: Path) -> dict[str, dict]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must be an object keyed by case_number")
    return raw


def _load_features(features_dir: Path) -> dict[str, dict]:
    by_case: dict[str, dict] = {}
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
        by_case[case_number] = feature_row
    return by_case


def _build_rows(labels_by_case: dict[str, dict], features_by_case: dict[str, dict]) -> list[dict]:
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
    if not LABELS_PATH.exists():
        raise FileNotFoundError(f"Missing labels file: {LABELS_PATH}")
    if not FEATURES_DIR.exists():
        raise FileNotFoundError(f"Missing features directory: {FEATURES_DIR}")

    labels_by_case = _load_labels(LABELS_PATH)
    features_by_case = _load_features(FEATURES_DIR)
    rows = _build_rows(labels_by_case, features_by_case)

    label_cases = set(labels_by_case)
    feature_cases = set(features_by_case)
    joined_cases = set(row["case_number"] for row in rows)
    rows_missing_features = sum(1 for row in rows if row.get("missing_features"))

    _write_jsonl(rows, OUT_JSONL)
    _write_csv(rows, OUT_CSV)

    labels_with_outcome = sum(1 for v in labels_by_case.values() if v.get("outcome") is not None)
    joined_with_outcome = sum(1 for row in rows if row.get("outcome") is not None)

    logger.info("Labels: %d", len(label_cases))
    logger.info("Features: %d", len(feature_cases))
    logger.info("Training rows: %d", len(joined_cases))
    logger.info("Labels missing features: %d", len(label_cases - feature_cases))
    logger.info("Features missing labels: %d", len(feature_cases - label_cases))
    logger.info("Rows flagged missing_features=True: %d", rows_missing_features)
    logger.info("Labels with non-null outcome: %d", labels_with_outcome)
    logger.info("Joined rows with non-null outcome: %d", joined_with_outcome)
    logger.info("Wrote: %s", OUT_JSONL)
    logger.info("Wrote: %s", OUT_CSV)


if __name__ == "__main__":
    main()
