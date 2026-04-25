#!/usr/bin/env python3
"""Extract labels only for cases that are still unlabeled.

Reads case text from ``scraper/data/processed`` and writes/updates a merged labels
file at ``data/processed/labels.json``. Existing labels are preserved; only
missing case numbers are sent through the label extraction pipeline.
"""

from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

# Allow `python scripts/extract_missing_labels.py` without editable install.
_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from features.config import FeaturesConfig
from features.labels import LabelExtractor, _is_label_doc

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TXT_DIR = Path("scraper/data/processed")
LABELS_PATH = Path("data/processed/labels.json")


def _load_existing_labels(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a JSON object keyed by case_number")
    return raw


def _case_numbers_with_outcome_docs(txt_dir: Path) -> list[str]:
    case_numbers: set[str] = set()
    for txt_path in txt_dir.glob("CSM*_*.txt"):
        if _is_label_doc(txt_path.name):
            case_number = txt_path.name.split("_", 1)[0]
            case_numbers.add(case_number)
    return sorted(case_numbers)


def main() -> None:
    cfg = FeaturesConfig()
    if not cfg.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set. Add it to .env or your shell environment.")

    if not TXT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {TXT_DIR}")

    all_candidates = _case_numbers_with_outcome_docs(TXT_DIR)
    existing = _load_existing_labels(LABELS_PATH)
    missing = [c for c in all_candidates if c not in existing]

    logger.info("Cases with outcome docs: %d", len(all_candidates))
    logger.info("Already labeled: %d", len(existing))
    logger.info("Missing labels: %d", len(missing))

    if not missing:
        logger.info("No unlabeled cases found. Nothing to do.")
        return

    extractor = LabelExtractor(config=cfg)
    new_labels = extractor.extract_batch(missing, TXT_DIR)
    merged = {**existing, **{k: v.model_dump() for k, v in new_labels.items()}}

    LABELS_PATH.parent.mkdir(parents=True, exist_ok=True)
    LABELS_PATH.write_text(json.dumps(merged, indent=2, sort_keys=True), encoding="utf-8")

    logger.info("New labels written: %d", len(new_labels))
    logger.info("Total labels in %s: %d", LABELS_PATH, len(merged))


if __name__ == "__main__":
    main()
