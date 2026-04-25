#!/usr/bin/env python3
"""Run LLM-based feature extraction for locally processed cases.

Builds `ProcessedCase` objects from `scraper/data/processed/*.txt` while excluding
outcome documents (judgments/orders/dismissals) to avoid leakage. Then runs
`FeatureExtractor.extract_batch`, writing per-case feature JSONs to
`data/features_cache/` (or `FEATURES_CACHE_DIR`).

Existing cache entries are reused; only missing cases trigger LLM calls.
"""

from __future__ import annotations

import asyncio
import logging
import sys
from datetime import date
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from data.schemas.case import ProcessedCase
from features.config import FeaturesConfig
from features.extraction import FeatureExtractor
from features.labels import _is_label_doc

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

TXT_DIR = Path("scraper/data/processed")


def _build_processed_cases() -> list[ProcessedCase]:
    """Build minimal ProcessedCase instances from non-outcome local txts."""
    docs_by_case: dict[str, list[Path]] = {}
    for txt_path in sorted(TXT_DIR.glob("CSM*_*.txt")):
        case_number = txt_path.name.split("_", 1)[0]
        if _is_label_doc(txt_path.name):
            continue
        docs_by_case.setdefault(case_number, []).append(txt_path)

    cases: list[ProcessedCase] = []
    for case_number, paths in sorted(docs_by_case.items()):
        chunks: list[str] = []
        for txt_path in paths:
            text = txt_path.read_text(encoding="utf-8").strip()
            if not text:
                continue
            doc_type = txt_path.stem.replace(f"{case_number}_", "")
            chunks.append(f"[Document: {doc_type}]\n{text}")
        if not chunks:
            continue
        full_text = "\n\n---\n\n".join(chunks)
        user_side = (
            "defendant"
            if any("DEFENDANT_S_CLAIM" in p.name.upper() for p in paths)
            else "plaintiff"
        )
        pc = ProcessedCase(
            case_number=case_number,
            case_title=case_number,
            cause_of_action=None,
            filing_date=date(2020, 1, 1),
            full_text=full_text,
            user_side=user_side,
        )
        cases.append(pc)
    return cases


async def _run() -> None:
    cfg = FeaturesConfig()
    if not cfg.llm_api_key:
        raise RuntimeError("LLM_API_KEY is not set. Add it to .env or your shell environment.")

    if not TXT_DIR.exists():
        raise FileNotFoundError(f"Input directory not found: {TXT_DIR}")

    cases = _build_processed_cases()
    logger.info("Built %d ProcessedCase objects from local txts", len(cases))
    if not cases:
        logger.info("No cases found; nothing to do.")
        return

    extractor = FeatureExtractor(config=cfg)
    vectors = await extractor.extract_batch(cases)
    logger.info("Feature extraction completed for %d cases", len(vectors))
    logger.info("Cache directory: %s", cfg.cache_dir)


def main() -> None:
    asyncio.run(_run())


if __name__ == "__main__":
    main()

